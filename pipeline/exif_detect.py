"""Auto-detect camera type from EXIF metadata."""
from PIL import Image
from PIL.ExifTags import TAGS
import os
from typing import Optional, Callable


# Camera model recommendations based on device
CAMERA_PROFILES = {
    "iphone": {"camera_model": "SIMPLE_RADIAL", "reason": "iPhone — standard lens with mild radial distortion"},
    "pixel": {"camera_model": "SIMPLE_RADIAL", "reason": "Pixel — standard lens with mild radial distortion"},
    "samsung": {"camera_model": "SIMPLE_RADIAL", "reason": "Samsung — standard lens with mild radial distortion"},
    "gopro": {"camera_model": "OPENCV", "reason": "GoPro — wide-angle lens with significant distortion"},
    "hero": {"camera_model": "OPENCV", "reason": "GoPro Hero — wide-angle lens with significant distortion"},
    "insta360": {"camera_model": "OPENCV_FISHEYE", "reason": "Insta360 — fisheye lens"},
    "dji": {"camera_model": "SIMPLE_RADIAL", "reason": "DJI drone — calibrated lens with mild distortion"},
    "mavic": {"camera_model": "SIMPLE_RADIAL", "reason": "DJI Mavic — calibrated lens"},
    "phantom": {"camera_model": "SIMPLE_RADIAL", "reason": "DJI Phantom — calibrated lens"},
    "air": {"camera_model": "SIMPLE_RADIAL", "reason": "DJI Air — calibrated lens"},
    "mini": {"camera_model": "SIMPLE_RADIAL", "reason": "DJI Mini — calibrated lens"},
    "canon": {"camera_model": "PINHOLE", "reason": "Canon DSLR/mirrorless — calibrated lens, minimal distortion"},
    "nikon": {"camera_model": "PINHOLE", "reason": "Nikon DSLR/mirrorless — calibrated lens"},
    "sony": {"camera_model": "PINHOLE", "reason": "Sony mirrorless — calibrated lens"},
    "fuji": {"camera_model": "PINHOLE", "reason": "Fujifilm — calibrated lens"},
}


def detect_camera(images_dir: str, on_output: Optional[Callable[[str], None]] = None) -> dict:
    """Detect camera type from EXIF and recommend COLMAP settings."""
    image_files = [
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".heic", ".heif"))
    ]

    if not image_files:
        return {"status": "no_images", "camera_model": "SIMPLE_RADIAL", "matcher": "exhaustive"}

    # Sample up to 5 images for EXIF
    samples = image_files[:5]
    makes = []
    models = []
    focal_lengths = []
    has_sequence = _check_sequential_naming(image_files)

    for fname in samples:
        try:
            img = Image.open(os.path.join(images_dir, fname))
            exif = img._getexif()
            if not exif:
                continue
            exif_dict = {TAGS.get(k, k): v for k, v in exif.items()}
            make = str(exif_dict.get("Make", "")).strip()
            model = str(exif_dict.get("Model", "")).strip()
            fl = exif_dict.get("FocalLength")
            if make:
                makes.append(make.lower())
            if model:
                models.append(model.lower())
            if fl:
                focal_lengths.append(float(fl))
        except Exception:
            continue

    # Determine camera model from make/model strings
    camera_model = "SIMPLE_RADIAL"
    reason = "Default — no EXIF camera info detected"

    all_text = " ".join(makes + models)
    for keyword, profile in CAMERA_PROFILES.items():
        if keyword in all_text:
            camera_model = profile["camera_model"]
            reason = profile["reason"]
            break

    # Determine matcher: sequential if filenames suggest video frames or ordered capture
    matcher = "sequential" if has_sequence else "exhaustive"
    matcher_reason = (
        "Sequential — images appear to be ordered (sequential filenames)"
        if has_sequence
        else "Exhaustive — unordered photos, comparing all pairs"
    )

    result = {
        "status": "detected",
        "camera_model": camera_model,
        "camera_reason": reason,
        "matcher": matcher,
        "matcher_reason": matcher_reason,
        "make": makes[0] if makes else None,
        "model": models[0] if models else None,
        "focal_length": focal_lengths[0] if focal_lengths else None,
        "num_images": len(image_files),
    }

    if on_output:
        on_output(f"Camera: {result.get('model', 'Unknown')} ({result.get('make', 'Unknown')})")
        on_output(f"  Recommended: {camera_model} — {reason}")
        on_output(f"  Matcher: {matcher} — {matcher_reason}")
        if focal_lengths:
            on_output(f"  Focal length: {focal_lengths[0]:.1f}mm")
        on_output(f"  Images: {len(image_files)}")

    return result


def _check_sequential_naming(filenames: list) -> bool:
    """Check if filenames suggest sequential/ordered capture (e.g. frame_0001.jpg)."""
    import re
    numbers = []
    for f in sorted(filenames)[:20]:
        m = re.search(r"(\d+)", os.path.splitext(f)[0])
        if m:
            numbers.append(int(m.group(1)))
    if len(numbers) < 5:
        return False
    # Check if numbers are roughly sequential (diff of ~1 between consecutive)
    diffs = [numbers[i+1] - numbers[i] for i in range(len(numbers)-1)]
    avg_diff = sum(diffs) / len(diffs) if diffs else 0
    return 0.5 <= avg_diff <= 5  # sequential with possible gaps
