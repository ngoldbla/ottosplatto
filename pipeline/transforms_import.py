"""Import nerfstudio-compatible transforms.json — bypass COLMAP with app-provided poses."""
import json
import os
import struct
import shutil
import numpy as np
from typing import Optional, Callable


# COLMAP camera model IDs
COLMAP_MODELS = {
    "SIMPLE_PINHOLE": 0,
    "PINHOLE": 1,
    "SIMPLE_RADIAL": 2,
    "RADIAL": 3,
    "OPENCV": 4,
    "OPENCV_FISHEYE": 5,
}

# Number of params per model
COLMAP_MODEL_PARAMS = {
    0: 3,   # SIMPLE_PINHOLE: f, cx, cy
    1: 4,   # PINHOLE: fx, fy, cx, cy
    2: 4,   # SIMPLE_RADIAL: f, cx, cy, k1
    3: 5,   # RADIAL: f, cx, cy, k1, k2
    4: 8,   # OPENCV: fx, fy, cx, cy, k1, k2, p1, p2
    5: 8,   # OPENCV_FISHEYE: fx, fy, cx, cy, k1, k2, k3, k4
}


def detect_transforms(input_dir: str) -> Optional[str]:
    """Check if a directory contains a nerfstudio-compatible transforms.json."""
    tf_path = os.path.join(input_dir, "transforms.json")
    if not os.path.isfile(tf_path):
        return None
    try:
        with open(tf_path) as f:
            data = json.load(f)
        # Must have frames with transform_matrix and either fl_x or camera_angle_x
        if "frames" in data and len(data["frames"]) > 0:
            frame = data["frames"][0]
            if "transform_matrix" in frame and ("fl_x" in data or "camera_angle_x" in data):
                return tf_path
    except (json.JSONDecodeError, KeyError):
        pass
    return None


def import_transforms(
    input_dir: str,
    project_dir: str,
    on_output: Optional[Callable[[str], None]] = None,
) -> dict:
    """Convert a nerfstudio transforms.json + images into a COLMAP-compatible project.

    Creates the directory structure expected by 3DGS/2DGS trainers:
        project_dir/
            images/          (copied from input)
            sparse/0/
                cameras.bin  (intrinsics)
                images.bin   (extrinsics — poses)
                points3D.bin (empty or from PLY)
    """
    tf_path = os.path.join(input_dir, "transforms.json")
    if not os.path.isfile(tf_path):
        return {"status": "error", "message": f"transforms.json not found in {input_dir}"}

    with open(tf_path) as f:
        tf = json.load(f)

    frames = tf.get("frames", [])
    if not frames:
        return {"status": "error", "message": "No frames in transforms.json"}

    if on_output:
        on_output(f"Importing {len(frames)} frames from transforms.json")
        on_output(f"  Camera model: {tf.get('camera_model', 'PINHOLE')}")

    # ── Extract camera intrinsics ──
    # Nerfstudio format supports fl_x/fl_y or camera_angle_x
    if "fl_x" in tf:
        fl_x = tf["fl_x"]
        fl_y = tf.get("fl_y", fl_x)
    elif "camera_angle_x" in tf:
        w = tf.get("w", 1920)
        fl_x = 0.5 * w / np.tan(0.5 * tf["camera_angle_x"])
        fl_y = fl_x
    else:
        return {"status": "error", "message": "No focal length in transforms.json"}

    w = tf.get("w", 1920)
    h = tf.get("h", 1080)
    cx = tf.get("cx", w / 2.0)
    cy = tf.get("cy", h / 2.0)

    # Distortion params
    k1 = tf.get("k1", 0.0)
    k2 = tf.get("k2", 0.0)
    p1 = tf.get("p1", 0.0)
    p2 = tf.get("p2", 0.0)

    # Determine COLMAP camera model
    camera_model_name = tf.get("camera_model", "PINHOLE").upper()
    if camera_model_name not in COLMAP_MODELS:
        camera_model_name = "PINHOLE"

    has_distortion = any(abs(v) > 1e-8 for v in [k1, k2, p1, p2])
    if has_distortion and camera_model_name == "PINHOLE":
        camera_model_name = "OPENCV"

    model_id = COLMAP_MODELS[camera_model_name]

    if on_output:
        on_output(f"  Focal length: {fl_x:.1f}, {fl_y:.1f}")
        on_output(f"  Image size: {w}×{h}")
        on_output(f"  COLMAP model: {camera_model_name} (id={model_id})")

    # ── Set up project directories ──
    images_dir = os.path.join(project_dir, "images")
    sparse_dir = os.path.join(project_dir, "sparse", "0")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(sparse_dir, exist_ok=True)

    # ── Copy images ──
    copied = 0
    image_names = []
    for frame in frames:
        file_path = frame.get("file_path", "")
        src = os.path.join(input_dir, file_path)
        if not os.path.isfile(src):
            # Try without leading path components
            src = os.path.join(input_dir, os.path.basename(file_path))
        if os.path.isfile(src):
            dst_name = os.path.basename(file_path)
            dst = os.path.join(images_dir, dst_name)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
            image_names.append(dst_name)
            copied += 1
        else:
            if on_output:
                on_output(f"  ⚠ Missing image: {file_path}")
            image_names.append(os.path.basename(file_path))

    if on_output:
        on_output(f"  Copied {copied}/{len(frames)} images")

    # ── Write cameras.bin ──
    # Single camera shared across all frames
    camera_id = 1
    if model_id == 1:  # PINHOLE
        params = [fl_x, fl_y, cx, cy]
    elif model_id == 4:  # OPENCV
        params = [fl_x, fl_y, cx, cy, k1, k2, p1, p2]
    elif model_id == 2:  # SIMPLE_RADIAL
        params = [fl_x, cx, cy, k1]
    elif model_id == 0:  # SIMPLE_PINHOLE
        params = [fl_x, cx, cy]
    else:
        params = [fl_x, fl_y, cx, cy]

    cameras_path = os.path.join(sparse_dir, "cameras.bin")
    with open(cameras_path, "wb") as f:
        f.write(struct.pack("<Q", 1))  # num_cameras
        f.write(struct.pack("<I", camera_id))  # camera_id
        f.write(struct.pack("<i", model_id))   # model_id
        f.write(struct.pack("<Q", int(w)))      # width
        f.write(struct.pack("<Q", int(h)))      # height
        for p in params:
            f.write(struct.pack("<d", float(p)))

    # ── Write images.bin ──
    # Convert 4×4 camera-to-world matrices to COLMAP format (world-to-camera quaternion + translation)
    images_path = os.path.join(sparse_dir, "images.bin")
    with open(images_path, "wb") as f:
        f.write(struct.pack("<Q", len(frames)))  # num_images

        for idx, frame in enumerate(frames):
            image_id = idx + 1
            c2w = np.array(frame["transform_matrix"], dtype=np.float64)

            # Ensure 4×4
            if c2w.shape == (3, 4):
                c2w = np.vstack([c2w, [0, 0, 0, 1]])

            # Invert to get world-to-camera
            w2c = np.linalg.inv(c2w)

            # Extract rotation and translation
            R = w2c[:3, :3]
            t = w2c[:3, 3]

            # Convert rotation matrix to quaternion (COLMAP uses w, x, y, z order)
            quat = _rotation_matrix_to_quaternion(R)

            # Write image entry
            f.write(struct.pack("<I", image_id))   # image_id
            for q in quat:
                f.write(struct.pack("<d", q))      # qw, qx, qy, qz
            for ti in t:
                f.write(struct.pack("<d", ti))     # tx, ty, tz
            f.write(struct.pack("<I", camera_id))  # camera_id

            # Image name as null-terminated string
            name = image_names[idx] if idx < len(image_names) else f"{idx:04d}.jpg"
            f.write(name.encode("utf-8") + b"\x00")

            # num_points2D = 0 (no feature points)
            f.write(struct.pack("<Q", 0))

    # ── Write points3D.bin ──
    # Check for a point cloud PLY
    ply_path = tf.get("ply_file_path", "")
    ply_src = os.path.join(input_dir, ply_path) if ply_path else None
    points3d_path = os.path.join(sparse_dir, "points3D.bin")

    if ply_src and os.path.isfile(ply_src):
        # Copy the PLY and also write a minimal points3D.bin
        shutil.copy2(ply_src, os.path.join(sparse_dir, "points3D.ply"))
        if on_output:
            size_mb = os.path.getsize(ply_src) / (1024 * 1024)
            on_output(f"  LiDAR point cloud: {size_mb:.1f} MB")

    # Write empty points3D.bin (trainers will initialize from PLY or random)
    with open(points3d_path, "wb") as f:
        f.write(struct.pack("<Q", 0))  # num_points = 0

    if on_output:
        on_output(f"  ✓ COLMAP files written to {sparse_dir}")
        on_output(f"  ✓ Ready for training — COLMAP reconstruction skipped")

    return {
        "status": "success",
        "num_images": len(frames),
        "copied": copied,
        "project_dir": project_dir,
        "sparse_dir": sparse_dir,
        "train_source": project_dir,
        "has_pointcloud": bool(ply_src and os.path.isfile(ply_src)),
    }


def _rotation_matrix_to_quaternion(R: np.ndarray) -> list:
    """Convert a 3×3 rotation matrix to quaternion [w, x, y, z] (COLMAP convention)."""
    # Shepperd's method for numerical stability
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    # Normalize
    norm = np.sqrt(w*w + x*x + y*y + z*z)
    return [w/norm, x/norm, y/norm, z/norm]
