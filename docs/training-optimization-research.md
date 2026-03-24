# Training Optimization Research (2025-2026 SOTA)

## Key Finding: gsplat > original 3DGS

gsplat (nerfstudio-project/gsplat) is the recommended training backend. It produces identical core quality but adds:
- **4x less VRAM**, 15% faster training
- **MCMC densification** — probabilistic, better novel view generalization
- **Appearance optimization** (`--app_opt`) — critical for varying lighting / auto-exposed phone footage
- **Bilateral grid post-processing** — per-view color correction
- **Pose refinement** (`--pose_opt`) — recovers from COLMAP pose errors
- **Depth loss** — when LiDAR depth available
- **Anti-aliasing** (`--antialiased`)
- **Scale/opacity regularization during training** (not just post-hoc cleanup)

## Recommended Training Commands

### gsplat with MCMC (Best Quality)
```bash
python simple_trainer.py mcmc \
  --data_dir /path/to/colmap_output \
  --data_factor 1 \
  --max_steps 40000 \
  --cap_max 2000000 \
  --noise_lr 5e5 \
  --app_opt --app_embed_dim 16 \
  --post_processing bilateral_grid \
  --sh_degree 3 --ssim_lambda 0.2 \
  --opacity_reg 0.01 --scale_reg 0.01 \
  --antialiased
```

### Original 3DGS (Maximized Quality)
```bash
python train.py -s /path/to/data \
  --iterations 50000 \
  --densify_until_iter 25000 \
  --densify_grad_threshold 0.00015 \
  --lambda_dssim 0.2 --sh_degree 3 -r 1
```

### 2DGS (With Regularization)
```bash
python train.py -s /path/to/data \
  --lambda_normal 0.05 \
  --lambda_distortion 0.01 \
  --depth_ratio 0 -r 1
```

## Key Parameters to Expose in TUI

| Parameter | Where | Impact |
|-----------|-------|--------|
| iterations | Already in TUI | 40-50K recommended for quality |
| densify_until_iter | New | Extend to 20-25K for complex scenes |
| densify_grad_threshold | New | Lower = more detail (0.0001-0.0002) |
| lambda_normal (2DGS) | New | 0.05 for geometry-accurate reconstruction |
| lambda_distortion (2DGS) | New | 0.01 reduces floaters |
| app_opt (gsplat) | New | Per-image appearance for varying lighting |
| pose_opt (gsplat) | New | Refine COLMAP poses during training |

## Repos to Integrate

| Priority | Repo | What |
|----------|------|------|
| **HIGH** | gsplat (nerfstudio-project/gsplat) | Replace training backend entirely |
| HIGH | 3DGS-MCMC (ubc-vision/3dgs-mcmc) | MCMC densification |
| MEDIUM | Mip-Splatting (autonomousvision/mip-splatting) | Anti-aliased rendering |
| MEDIUM | Self-Organizing-Gaussians (fraunhoferhhi) | 20-40x compression |
| LOW | Spec-Gaussian | Specular surface handling |
| LOW | Taming 3DGS | Budgeted Gaussian count |

## Implementation Plan

### Phase 1: Optimize existing 2DGS training
- Add `--lambda_normal 0.05` and `--lambda_distortion 0.01` to 2DGS command
- Expose densify_until_iter and densify_grad_threshold in TUI
- Default iterations to 40000 for quality mode

### Phase 2: Integrate gsplat as primary backend
- Install gsplat in a conda env (gs_gsplat)
- Create pipeline/train_gsplat.py wrapper
- Add MCMC + app_opt + bilateral_grid as defaults
- This would be the recommended method for emergency response

### Phase 3: Post-processing improvements
- Scale regularization during training (gsplat native)
- Self-Organizing Gaussians compression for web sharing
- Pose refinement for imperfect COLMAP/LiDAR poses
