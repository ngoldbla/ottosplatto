# OttoSplatto — Improvement Roadmap

Prioritized improvements for emergency response 3D reconstruction. Each item includes status, rationale, and implementation notes.

---

## ✅ Completed (Session 2026-03-23)

| # | Feature | Module |
|---|---------|--------|
| 1 | Splat cleanup (floaters, needles, outliers) | pipeline/cleanup.py |
| 2 | WASD + orbit camera controls in viewer | pipeline/viewer.py |
| 3 | Export manifest + viewer info panel | pipeline/manifest.py |
| 4 | Auto camera detection from EXIF | pipeline/exif_detect.py |
| 5 | Image quality pre-check | pipeline/precheck.py |
| 6a | 2DGS training method | pipeline/train.py (method="2dgs") |
| 7 | Training loss curves (sparkline) | pipeline/train.py + tui/app.py |
| — | Spark.js viewer fix + Wayland workaround | pipeline/viewer.py |
| — | Copyparty upload + QR code | tui/app.py |
| — | HEIC auto-conversion | pipeline/convert.py |
| — | transforms.json import (skip COLMAP) | pipeline/transforms_import.py |
| — | LiDAR point cloud conversion | pipeline/transforms_import.py |
| — | COLMAP progress parsing + timing | pipeline/reconstruct.py |
| — | Training progress bar + ETA | pipeline/train.py |
| — | TUI help text (camera model, matcher, training) | tui/app.py |
| — | Capture guidance panel | tui/app.py |
| — | File/directory browse buttons | tui/app.py |
| — | QR code full-screen modal | tui/app.py |
| — | Conda env dropdown | tui/app.py |
| — | --no-browser CLI flag | main.py |
| — | Responsive TUI layout | tui/app.py |

---

## 🔧 In Progress / Partially Done

### 6b. Scaffold-GS Training Method
**Status:** Not started (2DGS done, Scaffold-GS next)
**Complexity:** Medium

Scaffold-GS uses anchor-based structure that eliminates needles entirely. Better than 2DGS for scenes with fine geometric detail.

**Needs:** Check if conda env exists, find/clone repo, add to train.py method routing.

---

### 13. Tab Auto-Advance Reliability
**Status:** Bug fix attempted, needs validation
**Complexity:** Low

`_switch_tab` rewritten to detect thread context and use `call_from_thread` properly. Needs real-world testing across all transitions: Create→Reconstruct, Reconstruct→Train, Train→View, and the transforms.json shortcut (Create→Train).

---

### 14. TUI Log Copy/Paste Support
**Status:** Not started
**Complexity:** Medium

Textual's RichLog widget doesn't support text selection. Options:
- Add a "Copy Log" button that copies full log to clipboard via `pyperclip`
- Add a "Save Log" button that writes log to a file
- Investigate Textual's `TextArea` widget as an alternative (supports selection)

---

### 15. SOTA Training Optimization
**Status:** Research in progress
**Complexity:** High

Potential improvements:
- Exposure compensation flags for auto-exposed smartphone footage
- Better densification strategies (adaptive density control)
- Depth-supervised loss when LiDAR depth is available
- Anti-aliasing (Mip-Splatting / Mip-Gaussian)
- Learning rate tuning for different scene scales

---

## 📋 Remaining Planned Features

### 8. Comparison Mode in Viewer
**Status:** Not started
**Complexity:** Medium

Side-by-side or slider comparison of:
- iteration_7000 vs iteration_30000 (convergence check)
- Original vs cleaned splat (cleanup effectiveness)
- 3DGS vs 2DGS (method comparison)
- Before/after post-processing

---

### 9. Multi-Method Pipeline
**Status:** Partially done (2DGS integrated)
**Complexity:** High

Remaining methods from `~/gaussian-splat-pipeline/methods/`:
- **VicaSplat** (gs_vicasplat) — COLMAP-free, works on textureless scenes
- **G3Splat** (gs_g3splat) — another COLMAP-free approach
- **DN-Splatter** (gs_dn_splatter) — depth-normal supervision
- **EFA-GS** (gs_efa) — efficient feature-aware splatting

Each needs: find trainer script, verify CLI compatibility, add to method dropdown.

---

### 10. Compressed Export & Cloud Viewer
**Status:** Not started
**Complexity:** Medium

- Export to `.splat` format (SuperSplat compatible, ~5x smaller)
- Export to SPZ format (Spark.js native, ~10-20x smaller via Scaniverse)
- Generate standalone HTML viewer file (PLY + viewer embedded, shareable)
- SOGS compression for web delivery

Critical for emergency response: sharing 3D scenes with command centers over limited bandwidth.

---

### 11. COLMAP-Free Reconstruction
**Status:** Partially done (transforms.json import works)
**Complexity:** High

Already working: LiDAR apps that export transforms.json + pointcloud.ply bypass COLMAP.

Still needed:
- **DUSt3R / MASt3R** — learned stereo matching, works with as few as 15 images
- **VicaSplat** — direct image-to-splat without any SfM
- Auto-detect COLMAP failure and suggest alternatives
- Fallback chain: try COLMAP → if fails, try DUSt3R → if fails, try VicaSplat

---

### 12. Monitoring Dashboard
**Status:** Not started
**Complexity:** Medium

Real-time monitoring during long operations:
- GPU utilization, VRAM usage, temperature (nvidia-smi polling)
- Disk usage tracking (PLY files can be large)
- Estimated time remaining for each pipeline step
- Historical run comparisons
- Alert if GPU temp exceeds threshold

---

### 16. Record3D / LiDAR App UX Polish
**Status:** Core import done, UX needs work
**Complexity:** Low-Medium

The transforms.json converter works. Remaining polish:
- Better auto-detection of capture app (show app name in logs)
- Validate transforms.json integrity (check all images exist, matrices are valid)
- Support Record3D's native format (quaternion+translation, not 4x4 matrix)
- Guide user through Record3D capture in the TUI

---

### 17. Capture Quality Feedback Loop
**Status:** Pre-check done, live feedback not started
**Complexity:** High

Move beyond pre-check warnings to active capture guidance:
- Coverage heatmap showing which areas have sufficient views
- Real-time overlap estimation from camera poses
- "You need more images from this angle" guidance
- Post-reconstruction quality map (where reconstruction is weak)

---

### 18. Mesh Extraction from Gaussians
**Status:** Not started
**Complexity:** High

Convert gaussian splats to triangle meshes for:
- CAD/BIM integration (emergency response planning)
- 3D printing structural assessments
- Volumetric measurements (debris volume, structural damage extent)

Methods: Poisson reconstruction from gaussian centers, TSDF fusion, GOF (Gaussian Opacity Fields).
