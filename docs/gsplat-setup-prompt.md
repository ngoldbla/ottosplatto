# gsplat CUDA Build Environment Setup — Prompt for Next Session

## Copy this prompt to start the next session:

---

I need to set up gsplat for the OttoSplatto project at ~/ottosplatto.

### The problem:
gsplat's CUDA extensions fail to build because system nvcc is CUDA 12.6 but PyTorch (base env) was built with CUDA 13.0. The fix is to upgrade the system CUDA toolkit to 13.0 so everything matches.

### Step 1: Upgrade system CUDA toolkit to 13.0
```bash
sudo apt install cuda-toolkit-13-0
```
Then update PATH to use nvcc 13:
```bash
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH
```
Verify: `nvcc --version` should show 13.0.

### Step 2: Create gs_gsplat conda env
```bash
conda create -n gs_gsplat python=3.10 -y
conda run -n gs_gsplat pip install torch torchvision  # should get CUDA 13.0 build
```

### Step 3: Build gsplat from source
The repo is already cloned at `~/.ottosplatto/gsplat`
```bash
cd ~/.ottosplatto/gsplat
conda run -n gs_gsplat pip install -e .
conda run -n gs_gsplat pip install imageio tyro viser lpips pyyaml
```

### Step 4: Verify
```bash
conda run -n gs_gsplat python examples/simple_trainer.py --help
```

### Step 5: Test training
```bash
conda run -n gs_gsplat python examples/simple_trainer.py mcmc \
  --data_dir /home/dylan/splats/dino \
  --data_factor 1 --max_steps 1000 --result_dir /tmp/gsplat_test \
  --sh_degree 3 --antialiased
```

### System info:
- Linux 6.17.0-14-generic (Ubuntu 24.04)
- NVIDIA GeForce RTX 4090 (24GB, Ada Lovelace sm_89)
- Driver: 580.126.09
- Current nvcc: 12.6 (needs upgrade to 13.0)
- PyTorch (base): 2.11.0+cu130
- gsplat pip: 1.5.3
- gcc: 13.3.0
- CUDA 13.0 toolkit is available: `sudo apt install cuda-toolkit-13-0`

### After setup, OttoSplatto integration is already done:
- TUI has gsplat as default training method (⭐ recommended)
- `pipeline/train.py` has `_train_gsplat()` that calls simple_trainer.py mcmc
- GPU monitoring reports every 30s during training
- Just needs the conda env to work end-to-end

### Context:
OttoSplatto is an emergency response 3D reconstruction tool for first responders. gsplat MCMC is the SOTA training backend — 4x less VRAM, MCMC densification, appearance optimization, bilateral grid color correction, anti-aliasing. See `~/ottosplatto/docs/training-optimization-research.md`.

---
