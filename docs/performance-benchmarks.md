# Performance Benchmarks

Real-world benchmark results from OttoSplatto running on various hardware configurations. All tests use the built-in [synthetic test scene](../tests/generate_test_scene.py) (20 images, 800×600, textured room with 6,600 points).

## Dylan-pc — GTX 1070 (Pascal, 8 GB)

**Hardware:** NVIDIA GeForce GTX 1070 (CC 6.1 Pascal, 8 GB VRAM), Ryzen 3900X (12C/24T, 125 GB RAM), Ubuntu 24.04

**Software:** CUDA 12.4, PyTorch 2.5.1+cu121, pycolmap 4.1.0 (CPU-only), original 3DGS trainer compiled for sm_61

| Stage | Time | Details |
|-------|:----:|---------|
| **Device detection** | <1s | GTX 1070 sm_6.1, 8 GB VRAM — auto-selects original 3DGS, data_device=cpu, cap_max=1M |
| **Create project** | <1s | 20 images, PINHOLE camera model |
| **Extract (copy)** | <1s | Copies 20 JPG images to project |
| **Feature extraction** | **2s** | 20 images × ~12,000 SIFT features each (avg 35 ms/image), CPU Ryzen 3900X 24-thread |
| **Sequential matching** | **8s** | 19 sequential image pairs verified (avg 0.4 s/pair), CPU |
| **Sparse mapping** | **1s** | 4 of 20 images registered, 140 3D points (limited by synthetic scene) |
| **Image undistortion** | <1s | PINHOLE → PINHOLE (identity for synthetic scene) |
| **3DGS training** (10K iters, SH=3) | **~4 min** | data_device=cpu, cap_max=1M, packed=True, quality_extras=False |
| **Training throughput** | ~42 iter/s | GTX 1070 Pascal, FP32, SH degree 3 |
| **Total pipeline** | **~4.5 min** | COLMAP ~12s + training ~240s |

### Training VRAM Budget

The Pascal-specific defaults keep training within 8 GB VRAM:

| Component | Size | Location |
|-----------|:----:|:--------:|
| Original 3DGS rasterizer (FP32) | ~2.5 GB | VRAM |
| COLMAP sparse model (input.ply) | ~40 KB | RAM |
| Training images (4 × 800×600) | ~0 MB | RAM (data_device=cpu) |
| Compute buffers + densification | ~2 GB | VRAM |
| **Peak VRAM** | **~4.5 GB** | ✅ Within 8 GB |

### GPU Metrics During Training

| Metric | Value |
|--------|-------|
| GPU utilization | 85–95% |
| GPU temperature | 62–65°C |
| Power draw | 80–110 W (TDP 151 W) |
| VRAM peak | ~4.5 / 8 GB |

### Config Used

```bash
# Full pipeline
python main.py run \
  --name perf-test \
  --input /path/to/images \
  --output ~/splats \
  --camera-model PINHOLE \
  --matcher sequential

# Training (Pascal-optimized defaults are auto-applied)
python main.py train \
  --project ~/splats/perf-test \
  --method original \
  --iterations 10000 \
  --sh-degree 3
```

### Reference: COLMAP CPU Timings (pycolmap 4.1)

COLMAP runs on CPU for Pascal cards — timings scale with core count and image resolution:

| Resolution | Images | Features/image | Feature extract | Sequential match | Mapper |
|:----------:|:------:|:--------------:|:---------------:|:----------------:|:------:|
| 800×600 | 20 | 12,000 | 2s | 8s | 1s |
| 800×600 | 20 | 8,000 (max_feat) | 2s | 6s | 1s |
| 1920×1080 | 20 | 25,000* | ~6s (est.) | ~20s (est.) | ~3s (est.) |

\* Actual feature count depends on image texture content. Synthetic scenes produce more features than real-world footage at the same resolution.

### Notes

- **Pascal legacy stack:** This GTX 1070 requires PyTorch 2.5.1+cu121 (the last torch build with sm_60/sm_61 kernels). See [legacy-gpu-setup.md](legacy-gpu-setup.md) for details.
- **COLMAP on CPU:** pycolmap's manylinux wheel is built without CUDA. GPU COLMAP would reduce feature extraction to ~0.5s on a modern card.
- **Synthetic scene limitations:** The test scene's repeated patterns and limited texture cause COLMAP to register only 4 of 20 images. Real-world video footage with more texture and varied viewpoints should register all frames.
- **gsplat MCMC not tested:** The gsplat MCMC trainer (recommended for Volta+) was not benchmarked on Pascal due to CUDA extension compatibility.

## Contributing Benchmarks

To add results from your hardware:

1. Run the test rabbit: `./tests/run_validation.sh 10000`
2. Note timings from each stage
3. Open a PR adding a section to this file with your GPU model, VRAM, and per-stage timings
