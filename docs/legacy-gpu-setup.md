# Legacy GPU Setup (GTX 10-series / Pascal and older)

OttoSplatto detects your GPU's architecture and adapts automatically, but
pre-Volta cards (compute capability < 7.0) need a specific software stack
because modern PyTorch wheels and CUDA toolkits have dropped their kernels.

## Affected hardware

| Architecture | Compute cap | Cards |
|---|---|---|
| Pascal | 6.0 / 6.1 | GTX 1050–1080 Ti, Titan X/Xp, Tesla P100/P40, Quadro P-series |
| Maxwell | 5.0 / 5.2 | GTX 750–980 Ti, GTX Titan X, Tesla M40/M60 |
| Kepler | 3.5 | GTX 780, GTX Titan, Tesla K40/K80 (effectively unsupported — CUDA 12 dropped it) |

Volta (7.0) and Turing (7.5, GTX 16xx / RTX 20xx) still work with current
PyTorch wheels and need no special setup.

## Why standard setup fails on Pascal

- **CUDA 13 removed Pascal support** entirely; CUDA 12.8 deprecated it. You
  need a CUDA **12.x (≤ 12.6 recommended)** or 11.8 toolkit installed for
  compiling the rasterizer extensions.
- **Recent PyTorch wheels dropped sm_60/61 kernels.** A torch build without
  them fails at runtime with
  `CUDA error: no kernel image is available for execution on the device`.
  OttoSplatto's preflight check catches this before training starts and
  prints what the installed wheel actually supports.
- **No tensor cores / weak FP16** on Pascal — train in FP32 (the default for
  original 3DGS; don't enable mixed precision).

## Recommended setup (GTX 1070/1080 class, 8 GB)

Use the **original 3DGS** trainer — its `diff-gaussian-rasterization` is
well proven on Pascal. OttoSplatto auto-selects it on legacy GPUs
(override with `--method gsplat` if you've built gsplat for your card).

```bash
conda create -n gs_original python=3.10 -y
conda activate gs_original

# Torch wheels built for CUDA 11.8 still ship sm_60 kernels (binary-compatible
# with sm_61 devices like the GTX 1070). cu121/cu124 wheels also work.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Compile the CUDA extensions for your card's architecture.
# (find yours: python main.py device → "sm_6.1" → use "6.1")
git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting.git ~/.ottosplatto/gaussian-splatting
cd ~/.ottosplatto/gaussian-splatting
pip install -r requirements.txt  # or install plyfile/tqdm manually
TORCH_CUDA_ARCH_LIST="6.1" pip install submodules/diff-gaussian-rasterization
TORCH_CUDA_ARCH_LIST="6.1" pip install submodules/simple-knn
```

Verify:

```bash
conda run -n gs_original python -c "import torch; print(torch.cuda.get_arch_list(), torch.cuda.is_available())"
```

You should see `sm_60` (or `sm_61`) in the list and `True`.

## What OttoSplatto adapts automatically on low-VRAM cards

From `python main.py device` you can see the defaults chosen for your GPU:

- **8 GB tier** (GTX 1070/1080, RTX 2070/3070): images stay in system RAM
  (`--data_device cpu`), gentler densification, gsplat capped at 1M
  gaussians with `--packed` rasterization, VRAM-hungry extras
  (appearance opt, bilateral grid, pose refinement) disabled.
- **6 GB tier** (GTX 1060 6GB, RTX 2060): additionally SH degree 2,
  20k iterations, training at half image resolution.
- **< 6 GB**: SH degree 2, 15k iterations, quarter resolution, 350k
  gaussian cap. Expect reduced quality; keep captures short (≤ 100 frames).
- **COLMAP**: sequential matching and smaller feature-extraction image
  sizes below 20 GB VRAM (below 6 GB: 1600 px, 4096 features).

## Practical tips for 8 GB cards

- Prefer 1080p input video over 4K; extraction at 2 fps keeps image counts
  manageable and COLMAP fast.
- Close browsers/compositors using VRAM before training (`nvidia-smi` shows
  per-process usage).
- If training still OOMs, lower SH degree to 2 (`--sh-degree 2`) — SH
  coefficients dominate per-gaussian memory — or add `--iterations 20000`.
