# OttoSplatto

Gaussian Splatting pipeline for Linux + NVIDIA GPUs. Orchestrates: video → frames (FFmpeg) → sparse reconstruction (COLMAP) → 3DGS training → PLY viewer.

## Architecture

OttoSplatto is an **orchestrator** — it wraps existing CLI tools via subprocess, not reimplementing their internals.

```
main.py                  Entry point: dispatches CLI subcommands or launches TUI
pipeline/device.py       GPU/arch detection (Ada, Hopper, Blackwell, DGX Spark)
pipeline/extract.py      FFmpeg frame extraction from video
pipeline/reconstruct.py  COLMAP pipeline (feature extract → match → map → undistort)
pipeline/train.py        Wraps original 3DGS train.py via conda
pipeline/viewer.py       HTTP server + Three.js PLY point cloud viewer
tui/app.py               Textual TUI (tabbed: Project → Extract → Reconstruct → Train → View)
```

## Running

```bash
./run.sh                 # TUI (auto-installs Python deps)
python main.py device    # show GPU profile
python main.py run --name X --input video.mp4 --output ~/splats   # full pipeline
```

## System Dependencies

These must be installed on the host before the pipeline will work:

- **FFmpeg** — frame extraction (`ffmpeg`, `ffprobe`)
- **COLMAP** — sparse reconstruction (v3.x or v4.x; version-aware flag handling in reconstruct.py)
- **conda** — manages the `gs_original` environment for 3DGS training
- **NVIDIA driver + CUDA toolkit** — GPU acceleration for COLMAP and training
- **Original 3DGS repo** — auto-detected in common paths or cloned to `~/.ottosplatto/gaussian-splatting`

## Key Conventions

- All pipeline modules accept `on_output: Callable[[str], None]` for live log streaming and `check_cancel: Callable[[], bool]` for cancellation.
- Project state is stored in `project.json` inside each project directory.
- COLMAP 4.x renamed SIFT flags (`SiftExtraction.use_gpu` → `FeatureExtraction.use_gpu`). The `reconstruct.py` module detects the version automatically.
- The `create` command cleans stale data if a project directory already exists at the target path.
- Training uses `conda run -n gs_original` to invoke the original 3DGS `train.py`. The trainer path is auto-detected from common locations (see `SEARCH_PATHS` in `pipeline/train.py`).
- Device detection (`pipeline/device.py`) returns a `DeviceProfile` with hardware-aware defaults for training iterations, SH degree, and COLMAP settings.

## Known Limitations

- **Textureless scenes** (smooth walls, empty interiors) will fail COLMAP reconstruction. This is a fundamental COLMAP limitation, not an OttoSplatto bug. COLMAP-free methods (G3Splat, VicaSplat) are needed for those scenes.
- The PLY viewer renders point clouds via Three.js, not actual gaussian splats. For full splat rendering, open the PLY in [SuperSplat](https://playcanvas.com/supersplat/editor) or a dedicated viewer.
- DGX Spark (aarch64 + Blackwell) support is implemented in device detection but untested on real hardware. The 3DGS CUDA extensions may need recompilation with `TORCH_CUDA_ARCH_LIST="10.0"`.

## Testing

Run `/setup-and-test` to bootstrap dependencies and validate the full pipeline on this machine. The validation uses a synthetic 3D scene (textured room, 20 cameras on a circle) that reliably reconstructs in COLMAP.

## TUI Notes

- Built with [Textual](https://textual.textualize.io/) (v8.x). The `work` decorator is imported from `textual` (not `textual.worker`).
- `Select` widget uses `Select[str]([(display_text, value), ...])` format in Textual 8.x.
- Device profile is displayed in the log panel on mount.
