"""COLMAP sparse reconstruction pipeline."""
import subprocess
import os
import re
from typing import Optional, Callable


def _get_colmap_version() -> tuple:
    """Get COLMAP version as (major, minor)."""
    result = subprocess.run(
        ["colmap", "--version"], capture_output=True, text=True,
    )
    output = result.stdout + result.stderr
    match = re.search(r"(\d+)\.(\d+)", output)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return (3, 0)


def _parse_progress(line, step_name):
    """Extract user-friendly progress from COLMAP output lines."""
    s = line.strip()
    # Feature extraction: "Processed file [23/47]"
    m = re.search(r"Processed file \[(\d+)/(\d+)\]", s)
    if m:
        return f"  Extracting features: image {m.group(1)}/{m.group(2)}"
    # Matching: "Matching block [3/10]" or processed pairs
    m = re.search(r"Matching block \[(\d+)/(\d+)\]", s)
    if m:
        return f"  Matching: block {m.group(1)}/{m.group(2)}"
    # Mapper: "Registering image #N (M)" or registered images count
    m = re.search(r"Registering image #(\d+) \((\d+)\)", s)
    if m:
        return f"  Mapping: registered image {m.group(1)} ({m.group(2)} total)"
    m = re.search(r"=>\s+Registered images:\s+(\d+)", s)
    if m:
        return f"  Registered images: {m.group(1)}"
    m = re.search(r"=>\s+Points:\s+(\d+)", s)
    if m:
        return f"  3D points: {m.group(1)}"
    # Undistortion progress
    m = re.search(r"Undistorting image \[(\d+)/(\d+)\]", s)
    if m:
        return f"  Undistorting: image {m.group(1)}/{m.group(2)}"
    return None


def _run_cmd(cmd, on_output=None, check_cancel=None, step_name=""):
    """Run a subprocess command with live progress output."""
    if on_output:
        on_output(f"$ {' '.join(cmd)}")

    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )

    last_progress = ""
    for line in iter(process.stdout.readline, ""):
        if check_cancel and check_cancel():
            process.terminate()
            process.wait()
            return -1
        if on_output and line.strip():
            progress = _parse_progress(line, step_name)
            if progress and progress != last_progress:
                on_output(progress)
                last_progress = progress

    process.wait()
    return process.returncode


def run_colmap(
    project_dir: str,
    camera_model: str = "SIMPLE_RADIAL",
    use_gpu: bool = True,
    matcher: str = "exhaustive",
    undistort: bool = True,
    single_camera: bool = False,
    max_image_size: int = 3200,
    max_num_features: int = 8192,
    on_output: Optional[Callable[[str], None]] = None,
    check_cancel: Optional[Callable[[], bool]] = None,
) -> dict:
    """Run full COLMAP pipeline: extract → match → map → undistort."""
    images_dir = os.path.join(project_dir, "images")
    sparse_dir = os.path.join(project_dir, "sparse")
    database_path = os.path.join(project_dir, "database.db")

    if not os.path.isdir(images_dir):
        return {"status": "error", "message": f"Images not found: {images_dir}"}

    num_images = len([
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    if num_images == 0:
        return {"status": "error", "message": "No images found in images/"}

    os.makedirs(sparse_dir, exist_ok=True)

    # Detect COLMAP version — v4.x renamed SIFT* flags
    version = _get_colmap_version()
    if on_output:
        on_output(f"COLMAP {version[0]}.{version[1]} detected, {num_images} images")

    if version[0] >= 4:
        extract_gpu = "FeatureExtraction.use_gpu"
        match_gpu = "FeatureMatching.use_gpu"
    else:
        extract_gpu = "SiftExtraction.use_gpu"
        match_gpu = "SiftMatching.use_gpu"

    gpu_val = "1" if use_gpu else "0"

    import time

    # --- Step 1: Feature extraction ---
    if on_output:
        on_output(f"━━━ Step 1/4: Feature Extraction ({num_images} images) ━━━")
        on_output("  Finding keypoints in each image...")
    if single_camera and on_output:
        on_output("  Single camera mode — sharing intrinsics across all images")

    t0 = time.monotonic()
    extract_cmd = [
        "colmap", "feature_extractor",
        "--database_path", database_path,
        "--image_path", images_dir,
        "--ImageReader.camera_model", camera_model,
        f"--{extract_gpu}", gpu_val,
        "--SiftExtraction.max_image_size", str(max_image_size),
        "--SiftExtraction.max_num_features", str(max_num_features),
    ]
    if single_camera:
        extract_cmd.extend(["--ImageReader.single_camera", "1"])
    rc = _run_cmd(extract_cmd, on_output, check_cancel, "extract")

    if rc != 0:
        return {"status": "error", "step": "feature_extraction", "returncode": rc}
    if on_output:
        on_output(f"  ✓ Features extracted in {time.monotonic()-t0:.0f}s")

    # --- Step 2: Feature matching ---
    pairs = num_images * (num_images - 1) // 2 if matcher == "exhaustive" else num_images - 1
    if on_output:
        on_output(f"━━━ Step 2/4: {matcher.title()} Matching ━━━")
        on_output(f"  Comparing ~{pairs} image pairs — this is the slow step...")

    t0 = time.monotonic()
    matcher_cmd = "exhaustive_matcher" if matcher == "exhaustive" else "sequential_matcher"
    rc = _run_cmd([
        "colmap", matcher_cmd,
        "--database_path", database_path,
        f"--{match_gpu}", gpu_val,
    ], on_output, check_cancel, "match")

    if rc != 0:
        return {"status": "error", "step": "matching", "returncode": rc}
    if on_output:
        on_output(f"  ✓ Matching done in {time.monotonic()-t0:.0f}s")

    # --- Step 3: Sparse mapping ---
    if on_output:
        on_output("━━━ Step 3/4: Sparse Mapping ━━━")
        on_output("  Triangulating 3D points from matched features...")

    t0 = time.monotonic()
    rc = _run_cmd([
        "colmap", "mapper",
        "--database_path", database_path,
        "--image_path", images_dir,
        "--output_path", sparse_dir,
    ], on_output, check_cancel, "map")

    if rc != 0:
        return {"status": "error", "step": "mapper", "returncode": rc}
    if on_output:
        on_output(f"  ✓ Mapping done in {time.monotonic()-t0:.0f}s")

    sparse_model = os.path.join(sparse_dir, "0")
    if not os.path.isdir(sparse_model):
        return {
            "status": "error", "step": "mapper",
            "message": "No reconstruction produced (sparse/0/ missing)",
        }

    # --- Step 4: Undistortion (optional) ---
    train_source = project_dir
    if undistort:
        if on_output:
            on_output("━━━ Step 4/4: Image Undistortion ━━━")
            on_output("  Removing lens distortion from images...")

        t0 = time.monotonic()
        undistorted_dir = os.path.join(project_dir, "undistorted")
        rc = _run_cmd([
            "colmap", "image_undistorter",
            "--image_path", images_dir,
            "--input_path", sparse_model,
            "--output_path", undistorted_dir,
            "--output_type", "COLMAP",
        ], on_output, check_cancel, "undistort")

        if rc != 0:
            if on_output:
                on_output("  ⚠ Undistortion failed — training will use raw images")
        else:
            if on_output:
                on_output(f"  ✓ Undistortion done in {time.monotonic()-t0:.0f}s")
            # Original 3DGS expects sparse/0/  — restructure if needed
            undist_sparse = os.path.join(undistorted_dir, "sparse")
            undist_sparse_0 = os.path.join(undist_sparse, "0")
            if os.path.isdir(undist_sparse) and not os.path.isdir(undist_sparse_0):
                os.makedirs(undist_sparse_0, exist_ok=True)
                for fname in os.listdir(undist_sparse):
                    fpath = os.path.join(undist_sparse, fname)
                    if os.path.isfile(fpath):
                        os.rename(fpath, os.path.join(undist_sparse_0, fname))
            train_source = undistorted_dir
            if on_output:
                on_output(f"Undistorted data → {undistorted_dir}")
    else:
        if on_output:
            on_output("Skipping undistortion")

    if on_output:
        on_output(f"COLMAP complete. Training source: {train_source}")

    return {
        "status": "success",
        "num_images": num_images,
        "sparse_path": sparse_model,
        "train_source": train_source,
    }
