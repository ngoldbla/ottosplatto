# OttoSplatto — Improvement Roadmap

Prioritized improvements identified after MVP completion. Each item includes rationale, estimated complexity, and implementation notes.

## Tier 1: High Impact, Achievable Now

### 1. Splat Cleanup / Post-Processing ⭐ PRIORITY
**Status:** Not started
**Complexity:** Medium (~150 lines Python)

Vanilla 3DGS produces three types of visual artifacts:
- **Floaters** — gaussians in empty space from reconstruction noise
- **Needles** — extremely elongated gaussians (degenerate geometry, scale ratio >10:1)
- **Peripheral blur** — low-opacity gaussians at scene edges with few observations

**Implementation:** `pipeline/cleanup.py` — post-training filter that reads PLY, removes bad gaussians, writes cleaned PLY:
- Remove opacity < threshold (default 0.05)
- Remove needles where max_scale / min_scale > threshold (default 10)
- Remove outliers far from scene centroid (>N standard deviations)
- Optional: remove gaussians with very large scale (>99th percentile)
- Report stats: "Removed 23,400/142,000 gaussians (16%) — 12,100 low-opacity, 8,200 needles, 3,100 outliers"

**References:** Mini-Splatting, LightGaussian pruning approaches

---

### 2. WASD Camera Controls in Viewer ⭐ PRIORITY
**Status:** Not started
**Complexity:** Low (~50 lines JS)

Current OrbitControls only support mouse-based orbit/zoom/pan. Users expect FPS-style WASD+mouse traversal for walking through scenes.

**Implementation:** Add PointerLockControls (Three.js addon) alongside OrbitControls. Toggle between modes with a key (e.g., Tab or F). WASD moves camera position, mouse rotates view direction. Shift = faster movement, space = up, ctrl = down.

**UX:** Show control mode in the HUD overlay. Default to OrbitControls (familiar), let user switch to FPS mode.

---

### 3. Export & Reproducibility Manifest ⭐ PRIORITY
**Status:** Not started
**Complexity:** Low (~80 lines Python + JS)

After training, generate a `manifest.json` alongside the PLY capturing all pipeline settings and results. Enables reproducibility and provenance tracking.

**Manifest fields:**
- Timestamps (created, training duration)
- Source info (image count, resolution, camera model detected)
- COLMAP settings (camera model, matcher, version)
- Training settings (iterations, SH degree, loss history)
- Output stats (gaussian count, PLY size, cleanup stats if applied)
- Device info (GPU model, CUDA version, OttoSplatto version)

**Viewer integration:** "Export" button that downloads PLY + manifest as a zip. "Info" panel showing manifest data.

---

## Tier 2: Significant Quality Improvements

### 4. Auto Camera Detection from EXIF
**Status:** Not started
**Complexity:** Low (~60 lines Python)

Read EXIF from input images to auto-select camera model:
- iPhone/Android → SIMPLE_RADIAL
- GoPro/action cam → OPENCV
- DSLR/mirrorless → PINHOLE
- Unknown → SIMPLE_RADIAL with warning

Also extract focal length to pre-populate COLMAP intrinsics, improving reconstruction speed and accuracy.

---

### 5. Image Quality Pre-Check
**Status:** Not started
**Complexity:** Medium (~120 lines Python)

Before COLMAP, scan input images and warn about:
- Blurry frames (Laplacian variance < threshold)
- Inconsistent exposure (histogram variance across images)
- Mixed resolutions (flag mismatched sizes)
- Missing EXIF (warn about degraded COLMAP performance)
- Insufficient overlap (too few images, too little coverage)

Display results as a "readiness score" before reconstruction.

---

### 6. Better Training Methods (2DGS, Scaffold-GS)
**Status:** Not started
**Complexity:** High (new conda envs + training wrappers)

Vanilla 3DGS (2023) is outdated. Modern alternatives produce better quality:
- **2D Gaussian Splatting (2DGS)** — better surfaces, fewer floaters
- **Scaffold-GS** — anchor-based structure, eliminates needles
- **Mip-Splatting** — anti-aliased, less zoom-dependent blur
- **GOF (Gaussian Opacity Fields)** — enables mesh extraction

Add a "Method" dropdown in the Train tab. Each method = separate conda env and training script.

---

## Tier 3: Polish & Professional Features

### 7. Training Loss Curves in TUI
**Status:** Not started
**Complexity:** Low

Use Textual's `Sparkline` widget to show loss over time in the log panel. Helps users see if training has converged or needs more iterations.

---

### 8. Comparison Mode in Viewer
**Status:** Not started
**Complexity:** Medium

Side-by-side or slider comparison of:
- iteration_7000 vs iteration_30000 (convergence)
- Original vs cleaned (cleanup effectiveness)
- Different training methods

---

### 9. Multi-Method Pipeline
**Status:** Not started
**Complexity:** High

Integrate with existing `gaussian-splat-pipeline` methods:
- VicaSplat (COLMAP-free, works on textureless scenes)
- G3Splat
- DN-Splatter

Offer method selection in TUI with pros/cons for each.

---

### 10. Compressed Export & Cloud Viewer
**Status:** Not started
**Complexity:** Medium

- Export to `.splat` format (SuperSplat compatible, ~5x smaller)
- Export to SPZ format (Spark.js native, ~10-20x smaller)
- Generate standalone HTML viewer file (PLY embedded, shareable)
- SOGS compression for web delivery

---

### 11. COLMAP-Free Reconstruction
**Status:** Not started
**Complexity:** High

For scenes where COLMAP fails (textureless, repetitive patterns):
- DUSt3R / MASt3R — learned stereo matching
- VicaSplat — direct image-to-splat without SfM
- Auto-detect COLMAP failure and suggest alternatives

---

### 12. Monitoring Dashboard
**Status:** Not started
**Complexity:** Medium

Persistent monitoring during long operations:
- GPU utilization, memory, temperature (via nvidia-smi)
- Disk usage tracking
- Estimated time remaining for each step
- Historical run comparisons
