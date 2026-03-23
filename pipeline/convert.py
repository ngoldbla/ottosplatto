"""Image format conversion — HEIC/HEIF to JPEG with EXIF preservation."""
import os
from typing import Optional, Callable


def convert_heic_to_jpeg(
    images_dir: str,
    quality: int = 95,
    on_output: Optional[Callable[[str], None]] = None,
) -> dict:
    """Convert all HEIC/HEIF files in a directory to JPEG, preserving EXIF.

    COLMAP requires JPEG/PNG and reads EXIF (focal length, camera model)
    for camera intrinsics. HEIC is the default iPhone format but unsupported
    by COLMAP. This converts in-place, removing originals after successful
    conversion.
    """
    try:
        from PIL import Image
        from pillow_heif import register_heif_opener
        register_heif_opener()
    except ImportError:
        return {
            "status": "error",
            "message": "Install pillow-heif: pip install Pillow pillow-heif",
        }

    heic_files = [
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".heic", ".heif"))
    ]

    if not heic_files:
        return {"status": "success", "converted": 0}

    if on_output:
        on_output(f"Converting {len(heic_files)} HEIC images to JPEG (preserving EXIF)…")

    converted = 0
    for i, filename in enumerate(sorted(heic_files)):
        src = os.path.join(images_dir, filename)
        dst = os.path.join(images_dir, os.path.splitext(filename)[0] + ".jpg")

        try:
            img = Image.open(src)
            exif = img.info.get("exif", b"")
            if exif:
                img.save(dst, "JPEG", quality=quality, exif=exif)
            else:
                img.save(dst, "JPEG", quality=quality)
            img.close()
            os.remove(src)
            converted += 1
            if on_output and (converted % 10 == 0 or converted == len(heic_files)):
                on_output(f"  Converted {converted}/{len(heic_files)}")
        except Exception as e:
            if on_output:
                on_output(f"  ⚠ Failed to convert {filename}: {e}")

    if on_output:
        on_output(f"  ✓ {converted} images converted to JPEG")

    return {"status": "success", "converted": converted}
