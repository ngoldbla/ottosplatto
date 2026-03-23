# Tier 2 Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add auto camera detection from EXIF, image quality pre-check, 2DGS training support, and training loss curves to OttoSplatto.

**Architecture:** Four independent features that each touch pipeline/ (new module) + tui/app.py (integration). EXIF detection and quality pre-check run before COLMAP. 2DGS adds an alternative training backend. Loss curves add a TUI widget.

**Tech Stack:** Python, Pillow (EXIF), numpy (blur detection), Textual 8.x (TUI), conda (2DGS env)

---

## Feature A: Auto Camera Detection from EXIF

### Task A1: pipeline/exif_detect.py

**Files:**
- Create: `pipeline/exif_detect.py`

**Purpose:** Read EXIF from input images to auto-detect camera type and recommend COLMAP camera model + matcher.

```python
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
```

- [ ] Create `pipeline/exif_detect.py` with the code above
- [ ] Verify: `python3 -c "from pipeline.exif_detect import detect_camera; print(detect_camera('/home/dylan/uploads/truck', on_output=print))"`

### Task A2: TUI Integration — auto-fill camera settings on project creation

**Files:**
- Modify: `tui/app.py` — `_do_create_project` method

After images are copied into the project, run `detect_camera` and auto-fill the Reconstruct tab:

```python
# In _do_create_project, after copying images:
from pipeline.exif_detect import detect_camera
detection = detect_camera(os.path.join(self.project_dir, "images"), on_output=self._log)
# Auto-fill Reconstruct tab
if detection["status"] == "detected":
    self.app.call_later(
        lambda: self.query_one("#camera-model", Select).value = detection["camera_model"]
    )
    self.app.call_later(
        lambda: self.query_one("#matcher", Select).value = detection["matcher"]
    )
```

- [ ] Add EXIF detection call in `_do_create_project` after image copy
- [ ] Auto-fill camera model and matcher via `call_later`
- [ ] Verify by relaunching TUI with truck project

---

## Feature B: Image Quality Pre-Check

### Task B1: pipeline/precheck.py

**Files:**
- Create: `pipeline/precheck.py`

**Purpose:** Scan input images before COLMAP and warn about issues.

Checks:
1. **Blur detection** — Laplacian variance per image, flag bottom 10% as blurry
2. **Resolution consistency** — warn if images have different sizes
3. **EXIF presence** — warn if missing (degrades COLMAP)
4. **Image count** — warn if < 20 (likely insufficient overlap)
5. **Overall readiness score** — GOOD / FAIR / POOR

```python
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
    if blur_scores:
        blur_arr = np.array(blur_scores)
        threshold = np.percentile(blur_arr, 10)  # bottom 10%
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
    }
```

- [ ] Create `pipeline/precheck.py` with the code above
- [ ] Verify: `python3 -c "from pipeline.precheck import precheck_images; precheck_images('/home/dylan/uploads/truck', on_output=print)"`

### Task B2: TUI Integration — run pre-check before COLMAP

**Files:**
- Modify: `tui/app.py` — `_do_colmap` method

Insert pre-check at the start of `_do_colmap`, before HEIC conversion:

```python
# At start of _do_colmap:
from pipeline.precheck import precheck_images
check = precheck_images(os.path.join(self.project_dir, "images"), on_output=self._log)
if check.get("score") == "POOR":
    self._log("[yellow]⚠ Image quality is POOR — reconstruction may fail. Consider adding more images or removing blurry ones.[/]")
```

- [ ] Add precheck call in `_do_colmap` before HEIC conversion
- [ ] Verify by relaunching TUI

---

## Feature C: 2DGS Training Support

### Task C1: Update pipeline/train.py — multi-method support

**Files:**
- Modify: `pipeline/train.py`

The 2DGS repo is at `~/gaussian-splat-pipeline/methods/06_2dgs/repo/train.py` with conda env `gs_2dgs`. Its CLI is nearly identical to original 3DGS (`-s`, `--model_path`, `--iterations`, `--sh_degree`, `--save_iterations`).

Add to SEARCH_PATHS:
```python
SEARCH_PATHS_2DGS = [
    os.path.expanduser("~/gaussian-splat-pipeline/methods/06_2dgs/repo"),
    os.path.expanduser("~/2d-gaussian-splatting"),
    os.path.expanduser("~/.ottosplatto/2d-gaussian-splatting"),
]
```

Add `method` parameter to `train()`:
```python
def train(
    source_dir, output_dir, iterations=30000, sh_degree=3,
    save_iterations=None, conda_env=DEFAULT_CONDA_ENV,
    method="original",  # NEW: "original" or "2dgs"
    trainer_path=None, env_override=None,
    on_output=None, check_cancel=None,
) -> dict:
```

When `method="2dgs"`, use `SEARCH_PATHS_2DGS`, `conda_env="gs_2dgs"`, and the 2DGS train.py.

- [ ] Add `SEARCH_PATHS_2DGS` and `find_trainer_2dgs()` to train.py
- [ ] Add `method` parameter to `train()` function
- [ ] Route to correct trainer path and conda env based on method
- [ ] Verify: test 2DGS training on truck dataset

### Task C2: TUI — add method selector to Train tab

**Files:**
- Modify: `tui/app.py` — Train tab compose + `_do_train`

Add a Select widget for training method before the iterations input:

```python
yield Label("Training Method", classes="form-label")
yield Select[str](
    [("Original 3DGS (fast, good quality)", "original"),
     ("2D Gaussian Splatting (better surfaces, fewer artifacts)", "2dgs")],
    value="2dgs", id="train-method",
)
yield Static(
    "Original 3DGS — Fast training, good general quality. May produce floaters and needles.\n"
    "2D Gaussian Splatting — Better surface reconstruction, fewer artifacts. "
    "Recommended for emergency response scenes with structures and terrain.",
    classes="help-text",
)
```

In `_do_train`, read the method and pass it:
```python
method = self.query_one("#train-method", Select).value
conda_env = "gs_2dgs" if method == "2dgs" else self.query_one("#conda-env", Input).value.strip()
```

- [ ] Add method Select widget to Train tab
- [ ] Pass method to train() in _do_train
- [ ] Default to "2dgs" for better quality

---

## Feature D: Training Loss Curves

### Task D1: Capture loss history in train.py

**Files:**
- Modify: `pipeline/train.py`

In `_parse_training_line`, when loss is detected, append to a list. Return the loss history from `train()`.

```python
# In the training loop, collect losses:
loss_history = []
# When a loss is parsed:
loss_history.append((cur_iter, float(loss_value)))
# Return in result dict:
return {"status": "success", "ply_path": final_ply, "loss_history": loss_history}
```

- [ ] Add loss_history collection to train()
- [ ] Return loss_history in result dict

### Task D2: TUI — display loss sparkline

**Files:**
- Modify: `tui/app.py`

After training completes, display a text-based loss curve in the log:

```python
if result.get("loss_history"):
    losses = [l for _, l in result["loss_history"]]
    # Simple ASCII sparkline
    min_l, max_l = min(losses), max(losses)
    if max_l > min_l:
        chars = "▁▂▃▄▅▆▇█"
        sparkline = ""
        for l in losses:
            idx = int((l - min_l) / (max_l - min_l) * (len(chars) - 1))
            sparkline += chars[idx]
        self._log(f"  Loss curve: {sparkline}")
        self._log(f"  Start: {losses[0]:.4f} → Final: {losses[-1]:.4f}")
```

- [ ] Add sparkline rendering after training completes in _do_train
- [ ] Verify display in log panel
