# gsplat CUDA Build Environment Setup — Prompt for Next Session

## Copy this prompt to start the next session:

---

I need to set up a `gs_gsplat` conda environment on my Linux machine for the gsplat gaussian splatting training library. This is for the OttoSplatto project at ~/ottosplatto.

### What needs to happen:

1. Create a conda environment `gs_gsplat` with Python 3.10 (not 3.13 — better CUDA compat)
2. Install PyTorch with CUDA 12.6 (matching my system nvcc) — NOT CUDA 13.0
3. Build and install gsplat from source at `~/.ottosplatto/gsplat` (already cloned from https://github.com/nerfstudio-project/gsplat)
4. Install gsplat's training dependencies: imageio, tyro, viser, lpips, pyyaml
5. Verify the simple_trainer.py works: `conda run -n gs_gsplat python ~/.ottosplatto/gsplat/examples/simple_trainer.py --help`
6. Test training on a small dataset at `/home/dylan/splats/dino` (17 images with transforms.json poses)

### System info:
- Linux 6.17.0-14-generic (Ubuntu 24.04)
- NVIDIA GeForce RTX 4090 (24GB VRAM)
- Driver: 580.126.09
- System CUDA (nvcc): 12.6
- gcc: 13.3.0
- conda: 26.1.1
- Architecture: Ada Lovelace, sm_89

### The build error from last attempt:
```
RuntimeError: The detected CUDA version (12.6) mismatches the version that was used to compile PyTorch (13.0).
```
This happened because the base conda env has PyTorch built with CUDA 13.0 but nvcc is 12.6. The fix is to install PyTorch with CUDA 12.6 in the new env.

### Suggested commands (verify before running):
```bash
# Create env with Python 3.10 + PyTorch for CUDA 12.6
conda create -n gs_gsplat python=3.10 -y
conda activate gs_gsplat
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Build gsplat from source
cd ~/.ottosplatto/gsplat
pip install -e .

# Install training dependencies
pip install imageio tyro viser lpips pyyaml

# Verify
python examples/simple_trainer.py --help
```

### After setup, update OttoSplatto:
- The TUI already has gsplat as the default training method
- `pipeline/train.py` has `_train_gsplat()` that calls `simple_trainer.py mcmc`
- Just need to verify it works end-to-end with the dino test dataset

### Context:
OttoSplatto is an emergency response 3D reconstruction tool. gsplat MCMC is the SOTA training backend — 4x less VRAM, MCMC densification, appearance optimization, bilateral grid color correction, anti-aliasing. This is critical for production quality. See `~/ottosplatto/docs/training-optimization-research.md` for the full research.

---
