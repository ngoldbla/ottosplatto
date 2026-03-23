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


def _run_cmd(cmd, on_output=None, check_cancel=None):
    """Run a subprocess command with live output."""
    if on_output:
        on_output(f"$ {' '.join(cmd)}")

    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )

    for line in iter(process.stdout.readline, ""):
        if check_cancel and check_cancel():
            process.terminate()
            process.wait()
            return -1
        if on_output and line.strip():
            on_output(line.strip())

    process.wait()
    return process.returncode


def run_colmap(
    project_dir: str,
    camera_model: str = "SIMPLE_RADIAL",
    use_gpu: bool = True,
    matcher: str = "exhaustive",
    undistort: bool = True,
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

    # --- Step 1: Feature extraction ---
    if on_output:
        on_output("━━━ Step 1/4: Feature Extraction ━━━")

    rc = _run_cmd([
        "colmap", "feature_extractor",
        "--database_path", database_path,
        "--image_path", images_dir,
        "--ImageReader.camera_model", camera_model,
        f"--{extract_gpu}", gpu_val,
    ], on_output, check_cancel)

    if rc != 0:
        return {"status": "error", "step": "feature_extraction", "returncode": rc}

    # --- Step 2: Feature matching ---
    if on_output:
        on_output(f"━━━ Step 2/4: {matcher.title()} Matching ━━━")

    matcher_cmd = "exhaustive_matcher" if matcher == "exhaustive" else "sequential_matcher"
    rc = _run_cmd([
        "colmap", matcher_cmd,
        "--database_path", database_path,
        f"--{match_gpu}", gpu_val,
    ], on_output, check_cancel)

    if rc != 0:
        return {"status": "error", "step": "matching", "returncode": rc}

    # --- Step 3: Sparse mapping ---
    if on_output:
        on_output("━━━ Step 3/4: Sparse Mapping ━━━")

    rc = _run_cmd([
        "colmap", "mapper",
        "--database_path", database_path,
        "--image_path", images_dir,
        "--output_path", sparse_dir,
    ], on_output, check_cancel)

    if rc != 0:
        return {"status": "error", "step": "mapper", "returncode": rc}

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

        undistorted_dir = os.path.join(project_dir, "undistorted")
        rc = _run_cmd([
            "colmap", "image_undistorter",
            "--image_path", images_dir,
            "--input_path", sparse_model,
            "--output_path", undistorted_dir,
            "--output_type", "COLMAP",
        ], on_output, check_cancel)

        if rc != 0:
            if on_output:
                on_output("Undistortion failed — training will use raw images")
        else:
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
