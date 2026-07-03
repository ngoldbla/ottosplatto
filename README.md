# OttoSplatto

Gaussian Splatting pipeline for Linux + NVIDIA GPUs. Orchestrates the full workflow from video to trained 3D Gaussian Splats.

Supports NVIDIA GPUs from GTX 10-series (Pascal) through RTX 50-series and DGX Spark (Grace Blackwell), including low-VRAM (4–8 GB) cards and multi-GPU workstations. Auto-detects hardware and adapts training defaults.

## Pipeline

```
Video → FFmpeg → Frames → COLMAP → Sparse Reconstruction → 3DGS Training → PLY Viewer
```

## Requirements

### System
- Linux (x86_64 or aarch64)
- NVIDIA GPU with CUDA (toolkit version must match PyTorch's CUDA version)
- Python 3.10+
- [FFmpeg](https://ffmpeg.org/)
- [COLMAP](https://colmap.github.io/) (v3.x or v4.x — version differences handled automatically)
- conda (for managing training environments)

### Training backends (at least one)
- **gsplat MCMC** (recommended) — `gs_gsplat` conda env. See `docs/gsplat-setup-prompt.md` for setup.
- **Original 3DGS** — `gs_original` conda env with [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- **2DGS** — `gs_2dgs` conda env with [2d-gaussian-splatting](https://github.com/hbb1/2d-gaussian-splatting)

## Quick Start

```bash
git clone https://github.com/ngoldbla/ottosplatto.git
cd ottosplatto
./run.sh              # installs Python deps, launches TUI
```

## Usage

### TUI (interactive terminal interface)

```bash
python main.py        # or ./run.sh
```

Tabbed interface: **Project → Extract → Reconstruct → Train → View**

### CLI (scriptable)

```bash
# Full pipeline in one shot
python main.py run --name my_scene --input /path/to/video.mp4 --output ~/splats

# Step by step
python main.py create      --name my_scene --input video.mp4 --output ~/splats
python main.py extract      --project ~/splats/my_scene --fps 2
python main.py reconstruct  --project ~/splats/my_scene --matcher sequential
python main.py train        --project ~/splats/my_scene --iterations 30000
python main.py view         --ply ~/splats/my_scene/output/point_cloud/iteration_30000/point_cloud.ply
```

### Device detection

```bash
python main.py device       # print GPU profile and recommended settings
```

## Project Structure

```
ottosplatto/
├── main.py                 # Entry point (CLI + TUI dispatch)
├── run.sh                  # Launcher (auto-installs deps)
├── CLAUDE.md               # Claude Code project instructions
├── .claude/
│   └── commands/
│       └── setup-and-test.md   # /setup-and-test slash command
├── pipeline/
│   ├── device.py           # GPU/arch detection and adaptive defaults
│   ├── extract.py          # FFmpeg frame extraction
│   ├── reconstruct.py      # COLMAP sparse reconstruction
│   ├── train.py            # 3DGS training wrapper
│   └── viewer.py           # Web-based Spark.js gaussian splat viewer
├── tui/
│   └── app.py              # Textual TUI application
└── tests/
    ├── generate_test_scene.py  # Synthetic 3D scene generator (test rabbit)
    └── run_validation.sh       # End-to-end pipeline validation
```

## Device Support

| Device | Arch | Detected As | Training Defaults |
|--------|------|-------------|-------------------|
| DGX Spark (128 GB) | aarch64 | Blackwell, sm_10.x | 30K iters, SH 4 |
| RTX 4090 (24 GB) | x86_64 | Ada, sm_8.9 | 30K iters, SH 3 |
| RTX 3090 (24 GB) | x86_64 | Ampere, sm_8.6 | 30K iters, SH 3 |
| RTX 3060 (12 GB) | x86_64 | Ampere, sm_8.6 | 30K iters, SH 3, 1.5M gaussian cap |
| GTX 1070/1080 (8 GB) | x86_64 | Pascal, sm_6.1 | 30K iters, SH 3, images in RAM, 1M cap, packed |
| GTX 1060 (6 GB) | x86_64 | Pascal, sm_6.1 | 20K iters, SH 2, half-res, 600K cap |
| ≤4 GB cards | any | auto | 15K iters, SH 2, quarter-res, 350K cap |

Architectures from Kepler through Blackwell are recognized (GTX 900/10-series, RTX 20/30/40/50-series, Titan, Tesla/datacenter). **Pre-Volta cards (Pascal and older)** need an older PyTorch/CUDA stack — OttoSplatto detects them, defaults to the original 3DGS trainer, and preflight-checks that your torch build has kernels for the card. See [docs/legacy-gpu-setup.md](docs/legacy-gpu-setup.md).

## Test Rabbit

A built-in synthetic 3D test scene that proves the full pipeline works without needing real video. Generates a textured room (checkerboard floor, patterned walls, scattered objects) rendered from 20 cameras on a circle — a scene that reliably reconstructs in COLMAP and trains in 3DGS.

```bash
# One command — generates test data, runs full pipeline, reports pass/fail
./tests/run_validation.sh

# Or generate the test scene separately
python tests/generate_test_scene.py --output /tmp/my_test

# Then run the pipeline on it
python main.py run --name rabbit --input /tmp/my_test/images --output /tmp/my_test --iterations 500
```

The validation script checks every component: device detection, frame extraction, COLMAP reconstruction, 3DGS training, PLY output, and the web viewer. Runs in under a minute on a 4090.

## Development with Claude Code

This repo includes built-in [Claude Code](https://docs.anthropic.com/en/docs/claude-code) support for development and testing.

### CLAUDE.md

Loaded automatically when Claude Code opens this repo. Contains project architecture, conventions, known limitations, and system dependency information. Claude Code uses this to understand the codebase without re-exploration each session.

### /setup-and-test

A slash command that bootstraps and validates OttoSplatto on any new machine. Run it in Claude Code after cloning:

```
/setup-and-test
```

It performs a 9-step validation:

1. **Check system deps** — ffmpeg, colmap, nvidia-smi, conda
2. **Install Python deps** — textual, rich
3. **Verify imports** — all pipeline modules and TUI
4. **Check conda env** — gs_original with PyTorch + CUDA + OpenCV
5. **Generate test data** — synthetic 3D scene (textured room, 20 cameras on a circle)
6. **Run full pipeline** — create → extract → COLMAP reconstruct → 3DGS train
7. **Test viewer** — HTTP server + Three.js PLY rendering
8. **Test TUI** — headless widget rendering
9. **Report results** — pass/fail summary table per component

This is the primary way to verify the application works end-to-end on a new device.

## Viewer

The built-in viewer renders gaussian splats in-browser using [Spark.js](https://sparkjs.dev/) + Three.js. It requires a browser with **WebGL2** and GPU acceleration.

### Wayland + NVIDIA

Chrome/Chromium on Wayland with NVIDIA drivers has a known issue where the GPU process crashes, disabling WebGL entirely. The viewer auto-detects this and launches Chrome with X11 fallback. If you open the URL manually, launch Chrome with:

```bash
chromium-browser --ozone-platform=x11 http://127.0.0.1:8765/?ply=FILE.ply
```

Or add `--no-browser` to skip auto-open and use your own browser:

```bash
python main.py view --ply path/to/point_cloud.ply --no-browser
```

### Fallback

If WebGL is completely unavailable (e.g. SSH/headless), the viewer falls back to a Three.js point cloud renderer that shows splat positions with SH DC colors.

## COLMAP Notes

- COLMAP 4.x renamed `SiftExtraction.use_gpu` → `FeatureExtraction.use_gpu` — OttoSplatto detects the version and uses the correct flags.
- Textureless scenes (e.g., empty interiors with smooth walls) will fail COLMAP reconstruction. Use COLMAP-free methods like [G3Splat](https://github.com/SZU-AdvTech-2023/174-G3Splat) or [VicaSplat](https://github.com/Vicassa/VicaSplat) for those scenes.
- For video input, **sequential matcher** is faster. For unordered photos, use **exhaustive**.
