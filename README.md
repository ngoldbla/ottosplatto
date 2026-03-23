# OttoSplatto

Gaussian Splatting pipeline for Linux + NVIDIA GPUs. Orchestrates the full workflow from video to trained 3D Gaussian Splats.

Supports RTX consumer GPUs (4090, 3090, …), DGX Spark (Grace Blackwell), and multi-GPU workstations. Auto-detects hardware and adapts training defaults.

## Pipeline

```
Video → FFmpeg → Frames → COLMAP → Sparse Reconstruction → 3DGS Training → PLY Viewer
```

## Requirements

- Linux (x86_64 or aarch64)
- NVIDIA GPU with CUDA
- Python 3.11+
- [FFmpeg](https://ffmpeg.org/)
- [COLMAP](https://colmap.github.io/) (v3.x or v4.x — version differences handled automatically)
- [Original 3DGS](https://github.com/graphdeco-inria/gaussian-splatting) in a conda environment

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
│   └── viewer.py           # Web-based PLY point cloud viewer
├── tui/
│   └── app.py              # Textual TUI application
└── tests/
    ├── generate_test_scene.py  # Synthetic 3D scene generator (test rabbit)
    └── run_validation.sh       # End-to-end pipeline validation
```

## Device Support

| Device | Arch | Detected As | Training Defaults |
|--------|------|-------------|-------------------|
| RTX 4090 (24 GB) | x86_64 | Ada, sm_8.9 | 30K iters, SH 3 |
| RTX 3090 (24 GB) | x86_64 | Ampere, sm_8.6 | 30K iters, SH 3 |
| DGX Spark (128 GB) | aarch64 | Blackwell, sm_10.x | 30K iters, SH 4 |
| Low-VRAM (<20 GB) | any | auto | 20K iters, SH 2 |

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

## COLMAP Notes

- COLMAP 4.x renamed `SiftExtraction.use_gpu` → `FeatureExtraction.use_gpu` — OttoSplatto detects the version and uses the correct flags.
- Textureless scenes (e.g., empty interiors with smooth walls) will fail COLMAP reconstruction. Use COLMAP-free methods like [G3Splat](https://github.com/SZU-AdvTech-2023/174-G3Splat) or [VicaSplat](https://github.com/Vicassa/VicaSplat) for those scenes.
- For video input, **sequential matcher** is faster. For unordered photos, use **exhaustive**.
