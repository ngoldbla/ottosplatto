"""Frame extraction from video using FFmpeg."""
import subprocess
import os
import shutil
from typing import Optional, Callable


def extract_frames(
    video_path: str,
    project_dir: str,
    fps: int = 2,
    resolution: Optional[str] = None,
    on_output: Optional[Callable[[str], None]] = None,
    check_cancel: Optional[Callable[[], bool]] = None,
) -> dict:
    """Extract frames from video using FFmpeg."""
    images_dir = os.path.join(project_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    vf_filters = [f"fps={fps}"]
    if resolution:
        vf_filters.append(f"scale={resolution}")

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", ",".join(vf_filters),
        "-q:v", "1",
        os.path.join(images_dir, "%04d.jpg"),
    ]

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
            return {"status": "cancelled"}
        if on_output and line.strip():
            on_output(line.strip())

    process.wait()

    num_frames = len([
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    if on_output:
        on_output(f"Extracted {num_frames} frames to {images_dir}")

    return {
        "status": "success" if process.returncode == 0 else "error",
        "returncode": process.returncode,
        "num_frames": num_frames,
        "path": images_dir,
    }


def copy_images(
    source_dir: str,
    project_dir: str,
    on_output: Optional[Callable[[str], None]] = None,
) -> dict:
    """Copy images from a directory into the project."""
    images_dir = os.path.join(project_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")
    count = 0
    for f in sorted(os.listdir(source_dir)):
        if f.lower().endswith(exts):
            shutil.copy2(os.path.join(source_dir, f), images_dir)
            count += 1

    if on_output:
        on_output(f"Copied {count} images to {images_dir}")

    return {"status": "success", "num_frames": count, "path": images_dir}
