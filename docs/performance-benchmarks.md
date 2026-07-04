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
