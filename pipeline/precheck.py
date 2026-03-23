"""Pre-check input images for COLMAP readiness."""
import os
import numpy as np
from PIL import Image
from typing import Optional, Callable


def precheck_images(
    images_dir: str,
    on_output: Optional[Callable[[str], None]] = None,
) -> dict:
    """Scan images and report quality issues before COLMAP."""
    from PIL.ExifTags import TAGS

    image_files = sorted([
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    if not image_files:
        return {"status": "error", "message": "No images found", "score": "POOR"}

    issues = []
    num_images = len(image_files)
    sizes = []
    blur_scores = []
    exif_count = 0

    if on_output:
        on_output(f"Pre-checking {num_images} images…")

    for i, fname in enumerate(image_files):
        path = os.path.join(images_dir, fname)
        try:
            img = Image.open(path)
            sizes.append(img.size)

            # Check EXIF
            exif = img._getexif()
            if exif:
                exif_count += 1

            # Blur detection via Laplacian variance
            gray = np.array(img.convert("L"), dtype=np.float64)
            # Laplacian kernel applied manually (no cv2 dependency)
            kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
            # Use stride tricks or simple convolution
            h, w = gray.shape
            if h > 100 and w > 100:
                # Sample center crop for speed
                ch, cw = h // 2, w // 2
                crop = gray[ch-50:ch+50, cw-50:cw+50]
                lap = np.zeros_like(crop)
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        lap += kernel[dy+1, dx+1] * np.roll(np.roll(crop, dy, 0), dx, 1)
                blur_scores.append(lap.var())
            img.close()
        except Exception:
            issues.append(f"Could not read {fname}")

        if on_output and (i + 1) % 10 == 0:
            on_output(f"  Checked {i+1}/{num_images}")

    # Analysis
    # Image count
    if num_images < 10:
        issues.append(f"⚠ Only {num_images} images — need at least 20 for reliable reconstruction")
    elif num_images < 20:
        issues.append(f"⚠ {num_images} images — 20+ recommended for good coverage")

    # Resolution consistency
    unique_sizes = set(sizes)
    if len(unique_sizes) > 1:
        issues.append(f"⚠ Mixed resolutions detected: {len(unique_sizes)} different sizes — COLMAP handles this but quality may vary")

    # EXIF presence
    exif_pct = exif_count / num_images * 100 if num_images > 0 else 0
    if exif_pct < 50:
        issues.append(f"⚠ Only {exif_pct:.0f}% of images have EXIF data — COLMAP will estimate camera intrinsics (slower, less accurate)")

    # Blur detection
    blur_threshold = None
    if blur_scores:
        blur_arr = np.array(blur_scores)
        threshold = np.percentile(blur_arr, 10)  # bottom 10%
        blur_threshold = float(threshold)
        blurry_count = (blur_arr < threshold).sum()
        very_blurry = (blur_arr < threshold * 0.3).sum()
        if very_blurry > 0:
            issues.append(f"⚠ {very_blurry} images appear very blurry — consider removing them")

    # Overall score
    if len(issues) == 0:
        score = "GOOD"
    elif any("Only" in i and "images" in i for i in issues) or any("very blurry" in i for i in issues):
        score = "POOR"
    else:
        score = "FAIR"

    if on_output:
        on_output(f"  Readiness: {score}")
        if issues:
            for issue in issues:
                on_output(f"  {issue}")
        else:
            on_output(f"  ✓ All checks passed — images look good for reconstruction")

    return {
        "status": "success",
        "score": score,
        "num_images": num_images,
        "exif_pct": exif_pct,
        "issues": issues,
        "blur_scores": blur_scores,
        "blur_threshold": blur_threshold,
    }


def cull_blurry_frames(
    images_dir: str,
    threshold_percentile: int = 10,
    on_output: Optional[Callable[[str], None]] = None,
) -> dict:
    """Remove the bottom N% of images by blur score (Laplacian variance).

    Blurry frames are moved to images_dir/../culled/ (not deleted).
    """
    image_files = sorted([
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    if not image_files:
        return {"culled": 0, "kept": 0, "threshold": 0.0}

    # Compute blur scores for all images
    scores = {}
    for fname in image_files:
        path = os.path.join(images_dir, fname)
        try:
            img = Image.open(path)
            gray = np.array(img.convert("L"), dtype=np.float64)
            h, w = gray.shape
            if h > 100 and w > 100:
                ch, cw = h // 2, w // 2
                crop = gray[ch-50:ch+50, cw-50:cw+50]
                kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
                lap = np.zeros_like(crop)
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        lap += kernel[dy+1, dx+1] * np.roll(np.roll(crop, dy, 0), dx, 1)
                scores[fname] = lap.var()
            else:
                scores[fname] = 0.0
            img.close()
        except Exception:
            scores[fname] = 0.0

    if not scores:
        return {"culled": 0, "kept": 0, "threshold": 0.0}

    all_scores = np.array(list(scores.values()))
    threshold = float(np.percentile(all_scores, threshold_percentile))

    culled_dir = os.path.join(os.path.dirname(images_dir.rstrip("/")), "culled")
    os.makedirs(culled_dir, exist_ok=True)

    culled = 0
    for fname, score in scores.items():
        if score < threshold:
            src = os.path.join(images_dir, fname)
            dst = os.path.join(culled_dir, fname)
            os.rename(src, dst)
            culled += 1
            if on_output:
                on_output(f"  Culled {fname} (blur score: {score:.1f} < {threshold:.1f})")

    kept = len(scores) - culled
    if on_output:
        on_output(f"  Culled {culled} blurry frames, kept {kept} (threshold: {threshold:.1f})")

    return {"culled": culled, "kept": kept, "threshold": threshold}
