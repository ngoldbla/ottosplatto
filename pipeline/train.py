"""Gaussian Splatting training wrapper — finds and runs the original 3DGS trainer."""
import subprocess
import os
import glob
from typing import Optional, Callable

SEARCH_PATHS = [
    os.path.expanduser("~/gaussian-splat-pipeline/methods/01_original_3dgs/repo"),
    os.path.expanduser("~/gaussian-splatting"),
    os.path.expanduser("~/3d-gaussian-splatting"),
    os.path.expanduser("~/.ottosplatto/gaussian-splatting"),
]

DEFAULT_CONDA_ENV = "gs_original"


def find_trainer() -> Optional[str]:
    """Locate an existing 3DGS train.py on disk."""
    for path in SEARCH_PATHS:
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


def train(
    source_dir: str,
    output_dir: str,
    iterations: int = 30000,
    sh_degree: int = 3,
    save_iterations: Optional[list] = None,
    conda_env: str = DEFAULT_CONDA_ENV,
    trainer_path: Optional[str] = None,
    env_override: Optional[dict] = None,
    on_output: Optional[Callable[[str], None]] = None,
    check_cancel: Optional[Callable[[], bool]] = None,
) -> dict:
    """Run Gaussian Splatting training via the original 3DGS train.py."""
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
    ]

    if on_output:
        on_output(f"$ {' '.join(cmd)}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")
    if env_override:
        env.update(env_override)

    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env,
    )

    for line in iter(process.stdout.readline, ""):
        if check_cancel and check_cancel():
            process.terminate()
            process.wait()
            return {"status": "cancelled"}
        if on_output and line.strip():
            on_output(line.strip())

    process.wait()

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
    }
