# Setup and Test OttoSplatto

Bootstrap OttoSplatto on this machine and run a full end-to-end validation.

## Steps

### 1. Check system dependencies

Verify the following are installed and report any missing ones:
- `ffmpeg` and `ffprobe`
- `colmap` (note the version — v3.x and v4.x have different flag names)
- `nvidia-smi` (CUDA GPU required)
- `conda` (for managing the training environment)
- Python 3.11+

For any missing dependency, explain how to install it on this system (detect the distro from `/etc/os-release`).

### 2. Install Python dependencies

```bash
pip install textual rich
```

### 3. Verify imports

Run each of these and report failures:
```bash
cd <repo_root>
python -c "from pipeline.device import detect; print(detect().summary())"
python -c "from pipeline.extract import extract_frames; print('extract OK')"
python -c "from pipeline.reconstruct import run_colmap; print('reconstruct OK')"
python -c "from pipeline.train import train, find_trainer; print(f'train OK — trainer: {find_trainer()}')"
python -c "from pipeline.viewer import launch_viewer; print('viewer OK')"
python -c "from tui.app import OttoSplattoApp; print('TUI OK')"
```

If the trainer is not found, report the searched paths and suggest either:
- Cloning the original 3DGS: `git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting.git ~/.ottosplatto/gaussian-splatting`
- Or setting up a conda env: `conda create -n gs_original python=3.11 pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia`

### 4. Check the gs_original conda environment

Verify the conda env exists and has the required packages:
```bash
conda run -n gs_original python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}')"
conda run -n gs_original python -c "import cv2; print(f'OpenCV {cv2.__version__}')"
```

If `gs_original` doesn't exist or is missing packages, guide the user through setup.

### 5. Generate synthetic test data

Create a synthetic 3D scene with real parallax that COLMAP can reconstruct:

```python
# Generate 20 images of a textured room from cameras on a circle
# Scene: checkerboard floor + patterned walls + scattered 3D objects
# Cameras: 20 positions on a circle of radius 4, looking at origin
# Resolution: 800x600
# Save to: /tmp/ottosplatto_validation/synth3d/images/
```

Write and run a Python script that renders this scene. Use numpy for projection math and ffmpeg to convert PPM→JPG (avoids PIL dependency). Key requirements:
- Multiple surfaces at different depths (floor y=0, walls at z=±3 and x=±3)
- Textured surfaces (checkerboard, sinusoidal color patterns) — NOT solid colors
- At least 5000 3D points for sufficient SIFT features
- Camera positions with real 3D parallax (circle, not line)

### 6. Run full pipeline validation

```bash
cd <repo_root>

# Create project from synthetic images
python main.py create --name validation --input /tmp/ottosplatto_validation/synth3d/images --output /tmp/ottosplatto_validation

# Copy images
python main.py extract --project /tmp/ottosplatto_validation/validation

# COLMAP reconstruction
python main.py reconstruct --project /tmp/ottosplatto_validation/validation --matcher exhaustive

# Train (short run — 500 iterations is enough for validation)
python main.py train --project /tmp/ottosplatto_validation/validation --iterations 500

# Verify PLY output exists
ls -la /tmp/ottosplatto_validation/validation/output/point_cloud/iteration_500/point_cloud.ply
```

### 7. Test the viewer

```python
from pipeline.viewer import launch_viewer
result = launch_viewer('<ply_path>', port=8765, open_browser=False)
# Verify HTTP 200 and HTML contains Three.js
import urllib.request
resp = urllib.request.urlopen(result['url'])
assert resp.status == 200
result['server'].shutdown()
```

### 8. Test the TUI

```python
from tui.app import OttoSplattoApp
import asyncio

async def test():
    app = OttoSplattoApp()
    async with app.run_test(size=(100, 30)) as pilot:
        assert app.query_one('#btn-create')
        assert app.query_one('#btn-extract')
        assert app.query_one('#btn-colmap')
        assert app.query_one('#btn-train')
        assert app.query_one('#btn-view')
        assert app.query_one('#log')

asyncio.run(test())
```

### 9. Report results

Print a summary table:

```
OttoSplatto Validation Results
──────────────────────────────
System deps      ✓/✗ (list any missing)
Python deps      ✓/✗
Device detection ✓/✗ (GPU name, arch, VRAM)
Trainer found    ✓/✗ (path)
Frame extraction ✓/✗ (N frames)
COLMAP recon     ✓/✗ (N images registered, N points)
Training         ✓/✗ (PLY size)
Viewer           ✓/✗
TUI              ✓/✗
```

If any step fails, diagnose the root cause and attempt to fix it before moving on.
