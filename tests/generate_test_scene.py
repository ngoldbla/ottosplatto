#!/usr/bin/env python3
"""Generate a synthetic 3D test scene for validating the OttoSplatto pipeline.

Creates 20 images of a textured room (checkerboard floor, patterned walls,
scattered 3D objects) rendered from cameras orbiting on a circle. This scene
reliably reconstructs in COLMAP and trains in 3DGS — use it to verify the
full pipeline works end-to-end without needing real video footage.

Usage:
    python tests/generate_test_scene.py                    # default output
    python tests/generate_test_scene.py --output ~/my_test # custom path
    python tests/generate_test_scene.py --num-cameras 30   # more viewpoints
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np


def generate(output_dir: str, num_cameras: int = 20, width: int = 800, height: int = 600):
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    np.random.seed(42)

    fx, fy = 500.0, 500.0
    cx, cy = width / 2, height / 2

    # ── Build 3D scene ───────────────────────────────────────

    points = []
    colors = []

    # Checkerboard floor (y=0)
    for x in np.arange(-3, 3, 0.15):
        for z in np.arange(-3, 3, 0.15):
            points.append([x, 0, z])
            c = 200 if (int(x * 2) + int(z * 2)) % 2 == 0 else 50
            colors.append([c, c, c])

    # Patterned walls
    for x in np.arange(-3, 3, 0.1):
        for y in np.arange(0, 2, 0.1):
            # Back wall (z=-3)
            points.append([x, y, -3])
            colors.append([
                int(80 + 40 * np.sin(x * 5)),
                int(120 + 40 * np.cos(y * 5)),
                150,
            ])
            # Front wall (z=3)
            points.append([x, y, 3])
            colors.append([
                150,
                int(80 + 40 * np.sin(x * 3)),
                int(100 + 40 * np.cos(y * 4)),
            ])

    for z in np.arange(-3, 3, 0.1):
        for y in np.arange(0, 2, 0.1):
            # Left wall (x=-3)
            points.append([-3, y, z])
            colors.append([
                int(100 + 50 * np.sin(z * 4)),
                80,
                int(120 + 40 * np.cos(y * 3)),
            ])
            # Right wall (x=3)
            points.append([3, y, z])
            colors.append([
                80,
                int(100 + 50 * np.sin(z * 3)),
                int(140 + 40 * np.cos(y * 4)),
            ])

    # Scattered objects in the room
    for _ in range(200):
        points.append([
            np.random.uniform(-2, 2),
            np.random.uniform(0.2, 1.5),
            np.random.uniform(-2, 2),
        ])
        colors.append([
            np.random.randint(50, 255),
            np.random.randint(50, 255),
            np.random.randint(50, 255),
        ])

    points = np.array(points)
    colors = np.array(colors, dtype=np.uint8)

    print(f"Scene: {len(points)} points, {num_cameras} cameras")

    # ── Render from each camera ──────────────────────────────

    for i in range(num_cameras):
        angle = 2 * np.pi * i / num_cameras
        radius = 4.0
        cam_pos = np.array([
            radius * np.cos(angle),
            1.2,
            radius * np.sin(angle),
        ])

        # Look-at matrix (cameras point at origin)
        forward = -cam_pos / np.linalg.norm(cam_pos)
        right = np.cross(forward, [0, 1, 0])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)

        R = np.array([right, -up, forward])
        t = -R @ cam_pos

        # Project all points
        pts_cam = (R @ points.T).T + t
        mask = pts_cam[:, 2] > 0.1
        pts_vis = pts_cam[mask]
        cols_vis = colors[mask]

        u = (fx * pts_vis[:, 0] / pts_vis[:, 2] + cx).astype(int)
        v = (fy * pts_vis[:, 1] / pts_vis[:, 2] + cy).astype(int)

        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:] = [30, 25, 40]

        # Render far-to-near for correct occlusion
        for idx in np.argsort(-pts_vis[:, 2]):
            px, py = u[idx], v[idx]
            if 2 <= px < width - 2 and 2 <= py < height - 2:
                img[py - 1 : py + 2, px - 1 : px + 2] = cols_vis[idx]

        # Save via PPM → JPEG (avoids PIL dependency)
        ppm = os.path.join(tempfile.gettempdir(), f"otto_{i:04d}.ppm")
        jpg = os.path.join(images_dir, f"{i + 1:04d}.jpg")
        with open(ppm, "wb") as f:
            f.write(f"P6\n{width} {height}\n255\n".encode())
            f.write(img.tobytes())

        subprocess.run(
            ["ffmpeg", "-y", "-i", ppm, "-q:v", "2", jpg],
            capture_output=True,
        )
        os.remove(ppm)

    print(f"Images saved to {images_dir}")
    return images_dir


def main():
    parser = argparse.ArgumentParser(
        description="Generate a synthetic 3D test scene for OttoSplatto validation",
    )
    parser.add_argument(
        "--output", default="/tmp/ottosplatto_validation/synth3d",
        help="Output directory (default: /tmp/ottosplatto_validation/synth3d)",
    )
    parser.add_argument("--num-cameras", type=int, default=20)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=600)
    args = parser.parse_args()

    generate(args.output, args.num_cameras, args.width, args.height)


if __name__ == "__main__":
    main()
