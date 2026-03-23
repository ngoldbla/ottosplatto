"""Splat Cleanup Filter — removes bad gaussians from a 3DGS PLY file."""
import os
import struct
from typing import Optional, Callable

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def cleanup_splat(
    input_ply: str,
    output_ply: str = None,
    min_opacity: float = 0.005,
    needle_opacity: float = 0.1,
    needle_ratio: float = 20.0,
    outlier_sigma: float = 4.0,
    giant_percentile: float = 99.5,
    on_output: Optional[Callable[[str], None]] = None,
) -> dict:
    """Read a 3DGS PLY, remove bad gaussians, write cleaned PLY.

    Cleanup criteria (applied as AND for needle+opacity, OR for outliers):
    1. Remove if opacity < min_opacity (nearly invisible, always remove)
    2. Remove if opacity < needle_opacity AND scale_ratio > needle_ratio
    3. Remove if distance from centroid > mean + outlier_sigma * std
    4. Remove if max_scale > giant_percentile-th percentile

    Returns dict with status, counts, breakdown, and output_path.
    """
    if not os.path.isfile(input_ply):
        return {"status": "error", "message": f"PLY not found: {input_ply}"}

    if output_ply is None:
        base, ext = os.path.splitext(input_ply)
        output_ply = f"{base}_cleaned{ext}"

    log = on_output or (lambda _: None)

    # ── Read PLY ──────────────────────────────────────────────
    log(f"Reading {input_ply}…")

    with open(input_ply, "rb") as f:
        raw = f.read()

    # Parse header
    header_end = raw.index(b"end_header\n") + len(b"end_header\n")
    header_str = raw[:header_end].decode("ascii")
    header_lines = header_str.strip().split("\n")

    vertex_count = 0
    props = []
    for line in header_lines:
        line = line.strip()
        if line.startswith("element vertex"):
            vertex_count = int(line.split()[2])
        if line.startswith("property float"):
            props.append(line.split()[2])

    num_props = len(props)
    if num_props == 0:
        return {"status": "error", "message": "No float properties found in PLY header"}

    log(f"PLY: {vertex_count} gaussians, {num_props} properties each")

    # Read vertex data as a numpy array
    data_bytes = raw[header_end:]
    expected_size = vertex_count * num_props * 4
    if len(data_bytes) < expected_size:
        return {
            "status": "error",
            "message": f"Data too short: expected {expected_size} bytes, got {len(data_bytes)}",
        }

    vertices = np.frombuffer(data_bytes[:expected_size], dtype=np.float32).reshape(
        vertex_count, num_props
    )

    # ── Property indices ──────────────────────────────────────
    def pidx(name):
        return props.index(name) if name in props else -1

    ix, iy, iz = pidx("x"), pidx("y"), pidx("z")
    iopacity = pidx("opacity")
    iscale0, iscale1, iscale2 = pidx("scale_0"), pidx("scale_1"), pidx("scale_2")

    if any(i < 0 for i in (ix, iy, iz, iopacity, iscale0)):
        return {"status": "error", "message": "Missing required properties (x/y/z/opacity/scale)"}

    # ── Compute derived values ────────────────────────────────
    positions = vertices[:, [ix, iy, iz]]
    opacity_raw = vertices[:, iopacity]
    opacity = _sigmoid(opacity_raw)

    # Support both 3DGS (3 scales) and 2DGS (2 scales)
    scale_indices = [i for i in (iscale0, iscale1, iscale2) if i >= 0]
    scale_raw = vertices[:, scale_indices]
    scale = np.exp(scale_raw)

    max_scale = np.max(scale, axis=1)
    min_scale = np.min(scale, axis=1)
    # Avoid divide-by-zero
    scale_ratio = max_scale / np.maximum(min_scale, 1e-12)

    # ── Build removal masks ───────────────────────────────────

    # 1. Nearly invisible (always remove)
    mask_invisible = opacity < min_opacity
    count_invisible = int(np.sum(mask_invisible))
    log(f"  Invisible (opacity < {min_opacity}): {count_invisible}")

    # 2. Semi-transparent needles (opacity < threshold AND ratio > threshold)
    mask_needle = (opacity < needle_opacity) & (scale_ratio > needle_ratio)
    count_needle = int(np.sum(mask_needle & ~mask_invisible))  # unique to this criterion
    log(f"  Needles (opacity < {needle_opacity} & ratio > {needle_ratio}): {count_needle}")

    # 3. Spatial outliers (distance from centroid > mean + sigma * std)
    centroid = np.mean(positions, axis=0)
    dists = np.linalg.norm(positions - centroid, axis=1)
    dist_mean = np.mean(dists)
    dist_std = np.std(dists)
    outlier_threshold = dist_mean + outlier_sigma * dist_std
    mask_outlier = dists > outlier_threshold
    count_outlier = int(np.sum(mask_outlier & ~mask_invisible & ~mask_needle))
    log(f"  Outliers (> {outlier_sigma}σ from centroid): {count_outlier}")

    # 4. Giant degenerate gaussians (max_scale > percentile)
    giant_threshold = np.percentile(max_scale, giant_percentile)
    mask_giant = max_scale > giant_threshold
    count_giant = int(
        np.sum(mask_giant & ~mask_invisible & ~mask_needle & ~mask_outlier)
    )
    log(f"  Giants (max_scale > {giant_percentile}th pctl = {giant_threshold:.4f}): {count_giant}")

    # ── Combine: invisible OR needle OR outlier OR giant ──────
    mask_remove = mask_invisible | mask_needle | mask_outlier | mask_giant
    total_removed = int(np.sum(mask_remove))
    keep = ~mask_remove
    output_count = int(np.sum(keep))

    log(f"Removing {total_removed} / {vertex_count} gaussians ({100*total_removed/max(vertex_count,1):.1f}%)")
    log(f"Keeping {output_count} gaussians")

    # ── Write cleaned PLY ─────────────────────────────────────
    # Rebuild header with updated vertex count
    new_header_lines = []
    for line in header_lines:
        stripped = line.strip()
        if stripped.startswith("element vertex"):
            new_header_lines.append(f"element vertex {output_count}")
        else:
            new_header_lines.append(stripped)
    new_header = "\n".join(new_header_lines) + "\n"

    clean_vertices = vertices[keep]

    log(f"Writing {output_ply}…")
    with open(output_ply, "wb") as f:
        f.write(new_header.encode("ascii"))
        f.write(clean_vertices.tobytes())

    output_size_mb = os.path.getsize(output_ply) / (1024 * 1024)
    log(f"Done — {output_ply} ({output_size_mb:.1f} MB)")

    return {
        "status": "success",
        "input_path": input_ply,
        "output_path": output_ply,
        "input_count": vertex_count,
        "output_count": output_count,
        "removed": {
            "total": total_removed,
            "invisible": count_invisible,
            "needles": count_needle,
            "outliers": count_outlier,
            "giants": count_giant,
        },
    }
