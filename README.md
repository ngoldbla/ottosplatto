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
├── pipeline/
│   ├── device.py           # GPU/arch detection and adaptive defaults
│   ├── extract.py          # FFmpeg frame extraction
│   ├── reconstruct.py      # COLMAP sparse reconstruction
│   ├── train.py            # 3DGS training wrapper
│   └── viewer.py           # Web-based PLY point cloud viewer
└── tui/
    └── app.py              # Textual TUI application
```

## Device Support

| Device | Arch | Detected As | Training Defaults |
|--------|------|-------------|-------------------|
| RTX 4090 (24 GB) | x86_64 | Ada, sm_8.9 | 30K iters, SH 3 |
| RTX 3090 (24 GB) | x86_64 | Ampere, sm_8.6 | 30K iters, SH 3 |
| DGX Spark (128 GB) | aarch64 | Blackwell, sm_10.x | 30K iters, SH 4 |
| Low-VRAM (<20 GB) | any | auto | 20K iters, SH 2 |

## COLMAP Notes

- COLMAP 4.x renamed `SiftExtraction.use_gpu` → `FeatureExtraction.use_gpu` — OttoSplatto detects the version and uses the correct flags.
- Textureless scenes (e.g., empty interiors with smooth walls) will fail COLMAP reconstruction. Use COLMAP-free methods like [G3Splat](https://github.com/SZU-AdvTech-2023/174-G3Splat) or [VicaSplat](https://github.com/Vicassa/VicaSplat) for those scenes.
- For video input, **sequential matcher** is faster. For unordered photos, use **exhaustive**.
