#!/usr/bin/env python3
"""OttoSplatto — Gaussian Splatting pipeline for any NVIDIA GPU on Linux.

Supports RTX consumer GPUs (4090, 3090, …), DGX Spark (Grace Blackwell),
and multi-GPU workstations.  Auto-detects hardware and adapts defaults.

Usage:
  python main.py                                    # Launch TUI
  python main.py tui                                # Launch TUI
  python main.py device                             # Print device profile
  python main.py create  --name X --input P --output D
  python main.py extract --project P [--fps N]
  python main.py reconstruct --project P [--camera-model M] [--matcher M]
  python main.py train   --project P [--iterations N] [--conda-env E] [--gpu 0]
  python main.py view    --ply P [--port N]
  python main.py run     --name X --input P --output D   # Full pipeline
"""
import argparse
import json
import os
import sys


def _log(msg):
    print(msg)


# ── commands ─────────────────────────────────────────────

def cmd_device(_args):
    from pipeline.device import detect
    profile = detect()
    _log(profile.summary())
    _log("")
    _log("Training defaults for this device:")
    for k, v in profile.training_defaults().items():
        _log(f"  {k}: {v}")


def cmd_create(args):
    from pipeline.device import detect
    profile = detect()

    project_dir = os.path.join(args.output, args.name)

    # Clean stale data from previous runs at this path
    import shutil
    for subdir in ("images", "output", "sparse", "undistorted"):
        p = os.path.join(project_dir, subdir)
        if os.path.isdir(p):
            shutil.rmtree(p)
    for f in ("database.db", "project.json"):
        p = os.path.join(project_dir, f)
        if os.path.isfile(p):
            os.remove(p)

    os.makedirs(os.path.join(project_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(project_dir, "output"), exist_ok=True)

    is_video = os.path.isfile(args.input)
    config = {
        "name": args.name,
        "input_path": os.path.abspath(args.input),
        "input_type": "video" if is_video else "images",
        "steps_completed": [],
        "device": {
            "arch": profile.arch,
            "gpu": profile.primary_gpu.name if profile.primary_gpu else "none",
            "is_dgx": profile.is_dgx,
        },
    }
    with open(os.path.join(project_dir, "project.json"), "w") as f:
        json.dump(config, f, indent=2)

    _log(f"Project created: {project_dir}")
    if profile.primary_gpu:
        _log(f"  Device: {profile.primary_gpu.name} ({profile.arch})")
    return project_dir


def cmd_extract(args):
    from pipeline.extract import extract_frames, copy_images

    config = _load_config(args.project)
    if config["input_type"] == "images":
        result = copy_images(config["input_path"], args.project, on_output=_log)
    else:
        result = extract_frames(
            config["input_path"], args.project,
            fps=args.fps, on_output=_log,
        )
    if result["status"] != "success":
        _log(f"ERROR: {result}")
        sys.exit(1)
    _update_steps(args.project, "extract")
    return result


def cmd_reconstruct(args):
    from pipeline.reconstruct import run_colmap

    result = run_colmap(
        args.project,
        camera_model=args.camera_model,
        use_gpu=not args.no_gpu,
        matcher=args.matcher,
        undistort=not args.no_undistort,
        on_output=_log,
    )
    if result["status"] != "success":
        _log(f"ERROR: {result}")
        sys.exit(1)

    config = _load_config(args.project)
    config["train_source"] = result["train_source"]
    _save_config(args.project, config)
    _update_steps(args.project, "colmap")
    return result


def cmd_train(args):
    from pipeline.train import train
    from pipeline.device import detect

    cfg_path = os.path.join(args.project, "project.json")
    config = _load_config(args.project) if os.path.isfile(cfg_path) else {}
    source = config.get("train_source", args.project)
    output = os.path.join(args.project, "output")

    # Device-aware GPU selection
    profile = detect()
    gpu_id = getattr(args, "gpu", None)
    if gpu_id is None and profile.num_gpus > 0:
        gpu_id = profile.cuda_visible_devices(1)
    env_override = {"CUDA_VISIBLE_DEVICES": str(gpu_id)} if gpu_id is not None else {}

    # Apply device defaults if user didn't override
    td = profile.training_defaults()
    iterations = args.iterations if args.iterations != 30000 else td.get("iterations", 30000)
    sh_degree = args.sh_degree if args.sh_degree != 3 else td.get("sh_degree", 3)

    result = train(
        source_dir=source, output_dir=output,
        iterations=iterations, sh_degree=sh_degree,
        conda_env=args.conda_env,
        env_override=env_override,
        on_output=_log,
    )
    if result["status"] != "success":
        _log(f"ERROR: {result}")
        sys.exit(1)

    if os.path.isfile(cfg_path):
        config["ply_path"] = result.get("ply_path")
        _save_config(args.project, config)
        _update_steps(args.project, "train")
    return result


def cmd_view(args):
    from pipeline.viewer import launch_viewer

    result = launch_viewer(args.ply, port=args.port, open_browser=not args.no_browser, on_output=_log)
    if result["status"] != "success":
        _log(f"ERROR: {result}")
        sys.exit(1)

    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        _log("Viewer stopped.")


def cmd_run(args):
    """Full pipeline: create → extract → reconstruct → train."""
    from pipeline.device import detect
    profile = detect()

    _log(f"OttoSplatto — {profile.primary_gpu.name if profile.primary_gpu else 'no GPU'} "
         f"({profile.arch})")
    if profile.is_dgx:
        _log(f"  DGX {profile.dgx_model} detected")
    _log("")

    project_dir = cmd_create(args)
    args.project = project_dir

    _log("\n━━━ EXTRACT ━━━")
    cmd_extract(args)

    _log("\n━━━ RECONSTRUCT ━━━")
    cmd_reconstruct(args)

    _log("\n━━━ TRAIN ━━━")
    cmd_train(args)

    config = _load_config(project_dir)
    ply = config.get("ply_path")
    if ply:
        _log(f"\n✓ Done! PLY: {ply}")
        _log(f"  View: python main.py view --ply {ply}")
    else:
        _log("\n✓ Pipeline complete (no PLY found — check training output)")


def cmd_tui(_args):
    from tui.app import OttoSplattoApp
    app = OttoSplattoApp()
    app.run()


# ── helpers ──────────────────────────────────────────────

def _load_config(project_dir):
    with open(os.path.join(project_dir, "project.json")) as f:
        return json.load(f)


def _save_config(project_dir, config):
    with open(os.path.join(project_dir, "project.json"), "w") as f:
        json.dump(config, f, indent=2)


def _update_steps(project_dir, step):
    config = _load_config(project_dir)
    if step not in config.get("steps_completed", []):
        config.setdefault("steps_completed", []).append(step)
    _save_config(project_dir, config)


# ── CLI parser ───────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="ottosplatto",
        description="OttoSplatto — Gaussian Splatting pipeline for any NVIDIA GPU on Linux",
    )
    sub = parser.add_subparsers(dest="command")

    # device
    sub.add_parser("device", help="Show detected device profile and recommended settings")

    # tui
    sub.add_parser("tui", help="Launch interactive TUI")

    # create
    p = sub.add_parser("create", help="Create a new project")
    p.add_argument("--name", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)

    # extract
    p = sub.add_parser("extract", help="Extract frames from video")
    p.add_argument("--project", required=True)
    p.add_argument("--fps", type=int, default=2)

    # reconstruct
    p = sub.add_parser("reconstruct", help="Run COLMAP reconstruction")
    p.add_argument("--project", required=True)
    p.add_argument("--camera-model", default="SIMPLE_RADIAL")
    p.add_argument("--matcher", default="exhaustive", choices=["exhaustive", "sequential"])
    p.add_argument("--no-gpu", action="store_true")
    p.add_argument("--no-undistort", action="store_true")

    # train
    p = sub.add_parser("train", help="Train Gaussian Splatting")
    p.add_argument("--project", required=True)
    p.add_argument("--iterations", type=int, default=30000)
    p.add_argument("--sh-degree", type=int, default=3)
    p.add_argument("--conda-env", default="gs_original")
    p.add_argument("--gpu", type=str, default=None, help="CUDA device index (e.g. 0, 0,1)")

    # view
    p = sub.add_parser("view", help="Launch PLY viewer")
    p.add_argument("--ply", required=True)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true", help="Don't auto-open browser")

    # run (full pipeline)
    p = sub.add_parser("run", help="Run full pipeline (create→extract→reconstruct→train)")
    p.add_argument("--name", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--fps", type=int, default=2)
    p.add_argument("--camera-model", default="SIMPLE_RADIAL")
    p.add_argument("--matcher", default="exhaustive")
    p.add_argument("--no-gpu", action="store_true")
    p.add_argument("--no-undistort", action="store_true")
    p.add_argument("--iterations", type=int, default=30000)
    p.add_argument("--sh-degree", type=int, default=3)
    p.add_argument("--conda-env", default="gs_original")
    p.add_argument("--gpu", type=str, default=None)

    args = parser.parse_args()

    if args.command is None or args.command == "tui":
        cmd_tui(args)
    elif args.command == "device":
        cmd_device(args)
    elif args.command == "create":
        cmd_create(args)
    elif args.command == "extract":
        cmd_extract(args)
    elif args.command == "reconstruct":
        cmd_reconstruct(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "view":
        cmd_view(args)
    elif args.command == "run":
        cmd_run(args)


if __name__ == "__main__":
    main()
