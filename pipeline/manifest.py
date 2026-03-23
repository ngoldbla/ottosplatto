"""Export Manifest — generates manifest.json with pipeline metadata for a PLY file."""
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from typing import Optional, Callable


def _read_ply_vertex_count(ply_path: str) -> int:
    """Read the vertex count from a PLY file header."""
    with open(ply_path, "rb") as f:
        for raw_line in f:
            line = raw_line.decode("ascii", errors="replace").strip()
            if line.startswith("element vertex"):
                return int(line.split()[2])
            if line == "end_header":
                break
    return 0


def _detect_gpu() -> dict:
    """Detect GPU name and CUDA version via nvidia-smi."""
    info = {"gpu_name": "unknown", "cuda_version": "unknown"}
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            info["gpu_name"] = result.stdout.strip().split("\n")[0]
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            info["driver_version"] = result.stdout.strip().split("\n")[0]
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if "release" in line.lower():
                    # e.g. "Cuda compilation tools, release 12.4, V12.4.131"
                    parts = line.split("release")
                    if len(parts) > 1:
                        info["cuda_version"] = parts[1].strip().rstrip(",").split(",")[0].strip()
                        break
    except Exception:
        pass

    return info


def generate_manifest(
    project_dir: str,
    ply_path: str,
    on_output: Optional[Callable[[str], None]] = None,
) -> dict:
    """Generate manifest.json alongside a PLY file with pipeline metadata.

    Reads project.json for pipeline settings, stats the PLY, detects GPU info.
    Returns dict with status, manifest data, and manifest_path.
    """
    log = on_output or (lambda _: None)

    if not os.path.isfile(ply_path):
        return {"status": "error", "message": f"PLY not found: {ply_path}"}

    log("Generating manifest…")

    # ── Read project config ───────────────────────────────────
    project_config = {}
    config_path = os.path.join(project_dir, "project.json")
    if os.path.isfile(config_path):
        with open(config_path) as f:
            project_config = json.load(f)

    # ── PLY stats ─────────────────────────────────────────────
    gaussian_count = _read_ply_vertex_count(ply_path)
    ply_size_bytes = os.path.getsize(ply_path)
    ply_size_mb = round(ply_size_bytes / (1024 * 1024), 2)

    log(f"  PLY: {gaussian_count} gaussians, {ply_size_mb} MB")

    # ── Check for cleaned PLY ─────────────────────────────────
    cleanup_stats = None
    base, ext = os.path.splitext(ply_path)
    # If current ply is the cleaned version, look for original
    if base.endswith("_cleaned"):
        original_path = base.replace("_cleaned", "") + ext
        if os.path.isfile(original_path):
            original_count = _read_ply_vertex_count(original_path)
            removed = original_count - gaussian_count
            cleanup_stats = {
                "original_count": original_count,
                "cleaned_count": gaussian_count,
                "removed": removed,
                "removed_pct": round(100 * removed / max(original_count, 1), 1),
            }
            log(f"  Cleanup: {removed} removed ({cleanup_stats['removed_pct']}%)")
    else:
        # Check if a cleaned version exists
        cleaned_path = f"{base}_cleaned{ext}"
        if os.path.isfile(cleaned_path):
            cleaned_count = _read_ply_vertex_count(cleaned_path)
            removed = gaussian_count - cleaned_count
            cleanup_stats = {
                "original_count": gaussian_count,
                "cleaned_count": cleaned_count,
                "removed": removed,
                "removed_pct": round(100 * removed / max(gaussian_count, 1), 1),
            }

    # ── Device info ───────────────────────────────────────────
    device_info = _detect_gpu()
    log(f"  GPU: {device_info['gpu_name']}, CUDA {device_info.get('cuda_version', '?')}")

    # ── Count source images ───────────────────────────────────
    source_images = project_config.get("num_frames", 0)
    if source_images == 0:
        # Try counting images directory
        images_dir = os.path.join(project_dir, "images")
        if os.path.isdir(images_dir):
            exts = {".jpg", ".jpeg", ".png"}
            source_images = sum(
                1 for f in os.listdir(images_dir)
                if os.path.splitext(f)[1].lower() in exts
            )

    # ── Build manifest ────────────────────────────────────────
    manifest = {
        "software": "OttoSplatto",
        "created": datetime.now(timezone.utc).isoformat(),
        "project_name": project_config.get("name", os.path.basename(project_dir)),
        "source_images": source_images,
        "colmap": {
            "camera_model": project_config.get("camera_model", "SIMPLE_RADIAL"),
            "matcher": project_config.get("matcher", "exhaustive"),
        },
        "training": {
            "iterations": project_config.get("iterations", 30000),
            "sh_degree": project_config.get("sh_degree", 3),
        },
        "output": {
            "gaussian_count": gaussian_count,
            "ply_size_mb": ply_size_mb,
            "ply_path": os.path.basename(ply_path),
        },
        "device": device_info,
    }

    if cleanup_stats:
        manifest["cleanup"] = cleanup_stats

    # ── Write manifest.json ───────────────────────────────────
    manifest_path = os.path.join(os.path.dirname(ply_path), "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    log(f"  Manifest written to {manifest_path}")

    return {
        "status": "success",
        "manifest_path": manifest_path,
        "manifest": manifest,
    }
