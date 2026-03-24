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

            # Convert from OpenGL convention (Y-up, -Z forward) to OpenCV (Y-down, Z-forward)
            # This is what 3DGS/2DGS do internally when reading transforms.json:
            #   c2w[:3, 1:3] *= -1  (negate Y and Z columns)
            c2w[:3, 1] *= -1  # Negate Y column
            c2w[:3, 2] *= -1  # Negate Z column

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
        if on_output:
            size_mb = os.path.getsize(ply_src) / (1024 * 1024)
            on_output(f"  LiDAR point cloud: {size_mb:.1f} MB")

        # Convert PLY to the format 3DGS/2DGS expects: x,y,z,nx,ny,nz,red,green,blue
        _convert_pointcloud_ply(ply_src, os.path.join(sparse_dir, "points3D.ply"), on_output)
    else:
        # No point cloud — write a minimal random one so training can initialize
        if on_output:
            on_output("  No point cloud provided — generating random initialization")
        _write_random_pointcloud(os.path.join(sparse_dir, "points3D.ply"), frames)

    # Write points3D.bin from the PLY so all trainers (including gsplat) can read it
    _write_points3d_bin_from_ply(os.path.join(sparse_dir, "points3D.ply"), points3d_path, on_output)

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


def _convert_pointcloud_ply(src_ply: str, dst_ply: str, on_output=None):
    """Convert a LiDAR PLY to the format expected by 3DGS/2DGS trainers.

    Trainers expect: x, y, z, nx, ny, nz, red, green, blue.
    LiDAR PLYs may be missing normals — we add zero normals if needed.
    """
    from plyfile import PlyData, PlyElement
    try:
        plydata = PlyData.read(src_ply)
    except ImportError:
        # Fallback: manual PLY parsing if plyfile not installed
        _convert_pointcloud_manual(src_ply, dst_ply, on_output)
        return
    except Exception as e:
        if on_output:
            on_output(f"  ⚠ Could not read PLY: {e}")
        _convert_pointcloud_manual(src_ply, dst_ply, on_output)
        return

    verts = plydata['vertex']
    n = len(verts)

    x = np.array(verts['x'], dtype=np.float32)
    y = np.array(verts['y'], dtype=np.float32)
    z = np.array(verts['z'], dtype=np.float32)

    # Normals — use existing or zeros
    has_normals = all(p in verts.data.dtype.names for p in ('nx', 'ny', 'nz'))
    if has_normals:
        nx = np.array(verts['nx'], dtype=np.float32)
        ny = np.array(verts['ny'], dtype=np.float32)
        nz = np.array(verts['nz'], dtype=np.float32)
    else:
        nx = ny = nz = np.zeros(n, dtype=np.float32)

    # Colors — use existing or white
    has_colors = all(p in verts.data.dtype.names for p in ('red', 'green', 'blue'))
    if has_colors:
        red = np.array(verts['red'], dtype=np.uint8)
        green = np.array(verts['green'], dtype=np.uint8)
        blue = np.array(verts['blue'], dtype=np.uint8)
    else:
        red = green = blue = np.full(n, 200, dtype=np.uint8)

    # Write in the expected format
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    arr = np.zeros(n, dtype=dtype)
    arr['x'] = x; arr['y'] = y; arr['z'] = z
    arr['nx'] = nx; arr['ny'] = ny; arr['nz'] = nz
    arr['red'] = red; arr['green'] = green; arr['blue'] = blue

    el = PlyElement.describe(arr, 'vertex')
    PlyData([el], text=False).write(dst_ply)
    if on_output:
        on_output(f"  Point cloud: {n} points → {dst_ply}")


def _convert_pointcloud_manual(src_ply: str, dst_ply: str, on_output=None):
    """Fallback PLY conversion without plyfile library."""
    with open(src_ply, 'rb') as f:
        header = b''
        while True:
            line = f.readline()
            header += line
            if line.strip() == b'end_header':
                break
        raw = f.read()

    hdr = header.decode('ascii')
    lines = hdr.strip().split('\n')
    props = [l.split() for l in lines if l.startswith('property')]
    n_verts = int([l for l in lines if l.startswith('element vertex')][0].split()[-1])

    # Build property map
    prop_names = [p[2] for p in props]
    prop_types = [p[1] for p in props]

    # Calculate stride
    type_sizes = {'float': 4, 'double': 8, 'uchar': 1, 'int': 4, 'short': 2}
    stride = sum(type_sizes.get(t, 4) for t in prop_types)

    has_normals = 'nx' in prop_names
    has_colors = 'red' in prop_names

    # Write new PLY with required format
    new_header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n_verts}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )

    import struct as st
    fmt_map = {'float': '<f', 'double': '<d', 'uchar': '<B', 'int': '<i', 'short': '<h'}

    with open(dst_ply, 'wb') as out:
        out.write(new_header.encode('ascii'))

        offset = 0
        for i in range(n_verts):
            vals = {}
            pos = offset
            for pname, ptype in zip(prop_names, prop_types):
                sz = type_sizes.get(ptype, 4)
                fmt = fmt_map.get(ptype, '<f')
                vals[pname] = st.unpack_from(fmt, raw, pos)[0]
                pos += sz
            offset += stride

            out.write(st.pack('<f', vals.get('x', 0)))
            out.write(st.pack('<f', vals.get('y', 0)))
            out.write(st.pack('<f', vals.get('z', 0)))
            out.write(st.pack('<f', vals.get('nx', 0) if has_normals else 0))
            out.write(st.pack('<f', vals.get('ny', 0) if has_normals else 0))
            out.write(st.pack('<f', vals.get('nz', 0) if has_normals else 0))
            out.write(st.pack('<B', int(vals.get('red', 200)) if has_colors else 200))
            out.write(st.pack('<B', int(vals.get('green', 200)) if has_colors else 200))
            out.write(st.pack('<B', int(vals.get('blue', 200)) if has_colors else 200))

    if on_output:
        on_output(f"  Point cloud: {n_verts} points → {dst_ply}")


def _write_random_pointcloud(dst_ply: str, frames: list):
    """Generate a minimal random point cloud from camera positions for initialization."""
    # Extract camera centers from transform matrices
    centers = []
    for frame in frames:
        c2w = np.array(frame["transform_matrix"], dtype=np.float64)
        if c2w.shape == (3, 4):
            c2w = np.vstack([c2w, [0, 0, 0, 1]])
        centers.append(c2w[:3, 3])

    centers = np.array(centers)
    centroid = centers.mean(axis=0)
    spread = np.linalg.norm(centers - centroid, axis=1).max()

    # Generate random points around the centroid
    n = 1000
    pts = centroid + np.random.randn(n, 3) * spread * 0.5

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )

    with open(dst_ply, 'wb') as f:
        f.write(header.encode('ascii'))
        for i in range(n):
            f.write(struct.pack('<fff', pts[i, 0], pts[i, 1], pts[i, 2]))
            f.write(struct.pack('<fff', 0.0, 0.0, 0.0))  # normals
            f.write(struct.pack('<BBB', 200, 200, 200))   # colors


def _write_points3d_bin_from_ply(ply_path: str, bin_path: str, on_output=None):
    """Convert a points3D.ply into COLMAP's points3D.bin format.

    COLMAP binary format per point:
        point3D_id (uint64), x y z (double×3), r g b (uint8×3),
        error (double), track_length (uint64),
        [image_id (uint32), point2D_idx (uint32)] × track_length
    """
    try:
        from plyfile import PlyData
        plydata = PlyData.read(ply_path)
        verts = plydata['vertex']
        n = len(verts)
        x = np.array(verts['x'], dtype=np.float64)
        y = np.array(verts['y'], dtype=np.float64)
        z = np.array(verts['z'], dtype=np.float64)
        has_colors = all(p in verts.data.dtype.names for p in ('red', 'green', 'blue'))
        if has_colors:
            red = np.array(verts['red'], dtype=np.uint8)
            green = np.array(verts['green'], dtype=np.uint8)
            blue = np.array(verts['blue'], dtype=np.uint8)
        else:
            red = green = blue = np.full(n, 200, dtype=np.uint8)
    except Exception:
        # Fallback: write empty bin if we can't read the PLY
        with open(bin_path, "wb") as f:
            f.write(struct.pack("<Q", 0))
        return

    with open(bin_path, "wb") as f:
        f.write(struct.pack("<Q", n))
        for i in range(n):
            f.write(struct.pack("<Q", i + 1))              # point3D_id
            f.write(struct.pack("<ddd", x[i], y[i], z[i])) # xyz
            f.write(struct.pack("<BBB", red[i], green[i], blue[i]))  # rgb
            f.write(struct.pack("<d", 0.0))                 # error
            f.write(struct.pack("<Q", 0))                   # track_length = 0

    if on_output:
        on_output(f"  Wrote {n} points to points3D.bin")
