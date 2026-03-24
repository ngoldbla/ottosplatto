"""Gaussian Splatting training wrapper — finds and runs the original 3DGS trainer."""
import re
import subprocess
import os
import glob
import time
from typing import Optional, Callable

SEARCH_PATHS = [
    os.path.expanduser("~/gaussian-splat-pipeline/methods/01_original_3dgs/repo"),
    os.path.expanduser("~/gaussian-splatting"),
    os.path.expanduser("~/3d-gaussian-splatting"),
    os.path.expanduser("~/.ottosplatto/gaussian-splatting"),
]

SEARCH_PATHS_2DGS = [
    os.path.expanduser("~/gaussian-splat-pipeline/methods/06_2dgs/repo"),
    os.path.expanduser("~/2d-gaussian-splatting"),
    os.path.expanduser("~/.ottosplatto/2d-gaussian-splatting"),
]

SEARCH_PATHS_GSPLAT = [
    os.path.expanduser("~/.ottosplatto/gsplat"),
    os.path.expanduser("~/gsplat"),
]

DEFAULT_CONDA_ENV = "gs_original"


def find_trainer() -> Optional[str]:
    """Locate an existing 3DGS train.py on disk."""
    for path in SEARCH_PATHS:
        if os.path.isfile(os.path.join(path, "train.py")):
            return path
    return None


def find_trainer_2dgs() -> Optional[str]:
    """Locate an existing 2DGS train.py on disk."""
    for path in SEARCH_PATHS_2DGS:
        if os.path.isfile(os.path.join(path, "train.py")):
            return path
    return None


def install_trainer(on_output: Optional[Callable] = None) -> Optional[str]:
    """Clone the original 3DGS repo if not found."""
    dest = os.path.expanduser("~/.ottosplatto/gaussian-splatting")
    if on_output:
        on_output(f"Cloning graphdeco-inria/gaussian-splatting → {dest}")

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    result = subprocess.run(
        ["git", "clone", "--recursive",
         "https://github.com/graphdeco-inria/gaussian-splatting.git", dest],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        if on_output:
            on_output(f"Clone failed: {result.stderr.strip()}")
        return None

    if on_output:
        on_output("Cloned. Install deps in a conda env:")
        on_output(f"  conda activate {DEFAULT_CONDA_ENV}")
        on_output(f"  pip install -r {dest}/requirements.txt")
        on_output(f"  pip install {dest}/submodules/diff-gaussian-rasterization")
        on_output(f"  pip install {dest}/submodules/simple-knn")

    return dest


def _parse_training_line(line, total_iters, last_report_iter, report_every, t_start):
    """Parse 3DGS/2DGS training output into user-friendly progress messages."""
    cur_iter = None

    # Format 1: tqdm progress bar (2DGS)
    # "Training progress:  40%|████      | 20/50 [00:01<00:01, 18.01it/s, Loss=0.54587, Points=20764]"
    m = re.search(r"\|\s*(\d+)/(\d+)\s*\[", line)
    if m:
        cur_iter = int(m.group(1))

    # Format 2: Original 3DGS "[ITER 7000]" or "Iteration 7000"
    if cur_iter is None:
        m = re.search(r"(?:ITER\s+|\bIteration\s+|step\s+)(\d+)", line, re.IGNORECASE)
        if m:
            cur_iter = int(m.group(1))

    if cur_iter is not None and cur_iter > 0:
        if cur_iter - last_report_iter >= report_every or cur_iter == total_iters:
            pct = cur_iter / total_iters * 100
            elapsed = time.monotonic() - t_start
            eta = elapsed / cur_iter * (total_iters - cur_iter)
            eta_str = f"{eta/60:.1f}min left" if eta > 60 else f"{eta:.0f}s left"

            # Extract loss (matches "Loss=0.54587" or "loss: 0.023")
            loss_m = re.search(r"(?:loss|Loss)[=:\s]+([0-9.]+)", line)
            loss_str = f"  loss={loss_m.group(1)}" if loss_m else ""

            # Extract point count (matches "Points=20764" or "gaussians: 142000")
            pts_m = re.search(r"(?:points?|gaussians?|splats?)[=:\s]+([0-9,]+)", line, re.IGNORECASE)
            pts_str = f"  pts={pts_m.group(1)}" if pts_m else ""

            bar_len = 20
            filled = int(bar_len * cur_iter / total_iters)
            bar = "█" * filled + "░" * (bar_len - filled)

            msg = f"  [{bar}] {cur_iter}/{total_iters} ({pct:.0f}%){loss_str}{pts_str} — {eta_str}"
            return {"msg": msg, "iter": cur_iter, "loss": float(loss_m.group(1)) if loss_m else None}

    # Catch saving checkpoints
    if re.search(r"saving.*iteration|point_cloud.*saved", line, re.IGNORECASE):
        return {"msg": f"  💾 {line}"}

    # Catch densification events
    if re.search(r"densif|prun|split|clone|reset", line, re.IGNORECASE):
        pts_m = re.search(r"(\d[\d,]+)\s*(?:points?|gaussians?)", line, re.IGNORECASE)
        if pts_m:
            return {"msg": f"  🔬 Densification: {pts_m.group(1)} gaussians"}

    return None


def train(
    source_dir: str,
    output_dir: str,
    iterations: int = 30000,
    sh_degree: int = 3,
    save_iterations: Optional[list] = None,
    conda_env: str = DEFAULT_CONDA_ENV,
    method: str = "original",
    trainer_path: Optional[str] = None,
    env_override: Optional[dict] = None,
    on_output: Optional[Callable[[str], None]] = None,
    check_cancel: Optional[Callable[[], bool]] = None,
) -> dict:
    """Run Gaussian Splatting training via original 3DGS, 2DGS, or gsplat."""
    if method == "gsplat":
        if on_output:
            on_output("⭐ Using gsplat MCMC (best quality — appearance opt, anti-aliasing, 4x less VRAM)")
        if conda_env == DEFAULT_CONDA_ENV:
            conda_env = "gs_gsplat"
        # gsplat uses its own simple_trainer.py
        return _train_gsplat(
            source_dir, output_dir, iterations, sh_degree,
            conda_env, env_override, on_output, check_cancel,
        )

    if method == "2dgs":
        if on_output:
            on_output("Using 2D Gaussian Splatting (better surfaces, fewer artifacts)")
        if conda_env == DEFAULT_CONDA_ENV:
            conda_env = "gs_2dgs"
        if trainer_path is None:
            trainer_path = find_trainer_2dgs()
        if trainer_path is None:
            return {"status": "error", "message": "Could not find 2DGS trainer. Expected at: " + SEARCH_PATHS_2DGS[0]}
    else:
        if trainer_path is None:
            trainer_path = find_trainer()

    if trainer_path is None:
        if on_output:
            on_output("3DGS trainer not found — cloning...")
        trainer_path = install_trainer(on_output)
        if trainer_path is None:
            return {"status": "error", "message": "Could not find or install 3DGS"}

    train_script = os.path.join(trainer_path, "train.py")
    if on_output:
        on_output(f"Trainer: {trainer_path}")
        on_output(f"Source:  {source_dir}")
        on_output(f"Output:  {output_dir}")
        on_output(f"Iters:   {iterations}  |  SH: {sh_degree}")

    os.makedirs(output_dir, exist_ok=True)

    if save_iterations is None:
        save_iterations = [7000, iterations]

    save_str = " ".join(str(s) for s in save_iterations)

    cmd = [
        "conda", "run", "-n", conda_env,
        "python", "-u", train_script,
        "-s", source_dir,
        "--model_path", output_dir,
        "--iterations", str(iterations),
        "--sh_degree", str(sh_degree),
        "--save_iterations", *[str(s) for s in save_iterations],
        "--ip", "127.0.0.1",
        "--port", "0",  # disable GUI server to avoid port conflicts
    ]

    # Method-specific quality flags
    if method == "2dgs":
        cmd.extend([
            "--lambda_normal", "0.05",       # Surface normal consistency
            "--lambda_dist", "0.01",         # Depth distortion regularization (reduces floaters)
            "--depth_ratio", "0",            # Mean depth for unbounded scenes
        ])
        if iterations >= 20000:
            cmd.extend(["--densify_until_iter", str(min(25000, iterations * 3 // 4))])
    elif method == "original":
        if iterations >= 20000:
            # Extend densification window for complex scenes
            cmd.extend([
                "--densify_until_iter", str(min(25000, iterations * 3 // 4)),
                "--densify_grad_threshold", "0.00015",
            ])

    if on_output:
        on_output(f"$ {' '.join(cmd)}")
        on_output(f"Training {iterations} iterations — this will take a few minutes…")

    # Log initial GPU state
    from pipeline.gpu_monitor import log_gpu_stats
    log_gpu_stats(on_output)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")
    if env_override:
        env.update(env_override)

    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env,
    )

    t_start = time.monotonic()
    last_report_iter = 0
    last_gpu_report = t_start
    gpu_report_interval = 30  # Report GPU stats every 30 seconds
    report_every = max(500, iterations // 20)  # ~20 progress updates
    loss_history = []

    for line in iter(process.stdout.readline, ""):
        if check_cancel and check_cancel():
            process.terminate()
            process.wait()
            return {"status": "cancelled"}

        s = line.strip()
        if not s:
            continue

        # Parse 3DGS training output for key milestones
        # Format: "Training progress  7000/30000   Loss: 0.0234"
        # or: "[ITER 7000] ... loss = 0.0234"
        progress = _parse_training_line(s, iterations, last_report_iter, report_every, t_start)
        if progress:
            if on_output:
                on_output(progress["msg"])
            last_report_iter = progress.get("iter", last_report_iter)
            if progress.get("loss") is not None:
                loss_history.append((progress["iter"], progress["loss"]))
            # Periodic GPU stats alongside progress
            now = time.monotonic()
            if now - last_gpu_report >= gpu_report_interval:
                log_gpu_stats(on_output)
                last_gpu_report = now
        elif on_output and ("error" in s.lower() or "warning" in s.lower() or "saving" in s.lower()):
            on_output(f"  {s}")

    process.wait()

    if on_output:
        elapsed = time.monotonic() - t_start
        on_output(f"  Training finished in {elapsed/60:.1f} minutes")

    if process.returncode != 0:
        return {"status": "error", "returncode": process.returncode}

    # Locate output PLY
    ply_files = sorted(glob.glob(
        os.path.join(output_dir, "point_cloud", "iteration_*", "point_cloud.ply")
    ))
    final_ply = ply_files[-1] if ply_files else None

    if on_output:
        if final_ply:
            size_mb = os.path.getsize(final_ply) / (1024 * 1024)
            on_output(f"Training complete — {final_ply} ({size_mb:.1f} MB)")
        else:
            on_output("Training finished but no PLY produced")

    return {
        "status": "success",
        "ply_path": final_ply,
        "output_dir": output_dir,
        "loss_history": loss_history,
    }


def _train_gsplat(
    source_dir: str,
    output_dir: str,
    iterations: int = 40000,
    sh_degree: int = 3,
    conda_env: str = "gs_gsplat",
    env_override: Optional[dict] = None,
    on_output: Optional[Callable[[str], None]] = None,
    check_cancel: Optional[Callable[[], bool]] = None,
) -> dict:
    """Train using gsplat's simple_trainer with MCMC strategy."""
    # Find gsplat simple_trainer.py
    trainer_script = None
    for path in SEARCH_PATHS_GSPLAT:
        candidate = os.path.join(path, "examples", "simple_trainer.py")
        if os.path.isfile(candidate):
            trainer_script = candidate
            break

    if trainer_script is None:
        return {
            "status": "error",
            "message": "gsplat trainer not found. Run: git clone https://github.com/nerfstudio-project/gsplat.git ~/.ottosplatto/gsplat",
        }

    cmd = [
        "conda", "run", "-n", conda_env,
        "python", "-u", trainer_script,
        "mcmc",
        "--data-dir", source_dir,
        "--data-factor", "1",
        "--max-steps", str(iterations),
        "--result-dir", output_dir,
        "--strategy.cap-max", "2000000",
        "--sh-degree", str(sh_degree),
        "--ssim-lambda", "0.2",
        "--opacity-reg", "0.01",
        "--scale-reg", "0.01",
        "--antialiased",
        "--save-ply",
        "--disable-viewer",
        "--disable-video",
    ]

    if on_output:
        on_output(f"$ {' '.join(cmd)}")
        on_output(f"Training {iterations} steps with gsplat MCMC…")

    from pipeline.gpu_monitor import log_gpu_stats
    log_gpu_stats(on_output)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")
    if env_override:
        env.update(env_override)

    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env,
    )

    t_start = time.monotonic()
    last_report_iter = 0
    last_gpu_report = t_start
    gpu_report_interval = 30
    report_every = max(500, iterations // 20)
    loss_history = []

    for line in iter(process.stdout.readline, ""):
        if check_cancel and check_cancel():
            process.terminate()
            process.wait()
            return {"status": "cancelled"}

        s = line.strip()
        if not s:
            continue

        # GPU stats on a timer independent of progress parsing
        now = time.monotonic()
        if now - last_gpu_report >= gpu_report_interval:
            log_gpu_stats(on_output)
            last_gpu_report = now

        # gsplat progress format: "loss=0.535| sh degree=0| :  30%|███| 15/50 [...]"
        # or tqdm: "|5000/40000|"  or  "[ITER 7000]"
        progress = _parse_training_line(s, iterations, last_report_iter, report_every, t_start)
        if progress:
            if on_output:
                on_output(progress["msg"])
            last_report_iter = progress.get("iter", last_report_iter)
            if progress.get("loss") is not None:
                loss_history.append((progress["iter"], progress["loss"]))
        elif on_output and ("error" in s.lower() or "saving" in s.lower() or "psnr" in s.lower()):
            on_output(f"  {s}")

    process.wait()

    if on_output:
        elapsed = time.monotonic() - t_start
        on_output(f"  Training finished in {elapsed/60:.1f} minutes")
        log_gpu_stats(on_output)

    if process.returncode != 0:
        return {"status": "error", "returncode": process.returncode}

    # gsplat saves PLY in result_dir
    ply_files = sorted(glob.glob(os.path.join(output_dir, "**", "*.ply"), recursive=True))
    # Also check the standard 3DGS output path
    ply_files += sorted(glob.glob(
        os.path.join(output_dir, "point_cloud", "iteration_*", "point_cloud.ply")
    ))
    final_ply = ply_files[-1] if ply_files else None

    if on_output:
        if final_ply:
            size_mb = os.path.getsize(final_ply) / (1024 * 1024)
            on_output(f"Training complete — {final_ply} ({size_mb:.1f} MB)")
        else:
            on_output("Training finished but no PLY produced")

    return {
        "status": "success",
        "ply_path": final_ply,
        "output_dir": output_dir,
        "loss_history": loss_history,
    }
