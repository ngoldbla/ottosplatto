"""OttoSplatto TUI — Textual-based terminal interface for Gaussian Splatting."""
import os
import json
import sys

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Button, Footer, Header, Input, Label, RichLog,
    Rule, Select, Static, Switch, TabbedContent, TabPane,
)
from textual import work

# Add parent dir to path so pipeline imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class OttoSplattoApp(App):
    TITLE = "OttoSplatto"
    SUB_TITLE = "Video → Splats on Linux + NVIDIA"

    CSS = """
    Screen {
        background: $surface;
    }
    #log-panel {
        height: 14;
        dock: bottom;
        border-top: solid $primary;
    }
    #log-panel RichLog {
        height: 100%;
        scrollbar-size: 1 1;
    }
    #log-label {
        dock: bottom;
        height: 1;
        background: $primary;
        color: $text;
        text-style: bold;
        padding: 0 1;
    }
    TabPane {
        padding: 1 2;
    }
    .form-label {
        margin-top: 1;
        color: $text-muted;
    }
    .form-input {
        margin-bottom: 1;
    }
    .form-row {
        height: 3;
        align: left middle;
    }
    .form-row Label {
        margin-left: 1;
        width: auto;
    }
    .form-row Switch {
        height: 3;
    }
    Button {
        margin-top: 1;
        min-width: 24;
    }
    .status-bar {
        height: 1;
        background: $boost;
        color: $text;
        padding: 0 1;
    }
    #project-status {
        dock: top;
        height: 1;
        background: $primary-darken-2;
        color: $text;
        padding: 0 1;
        text-style: bold;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("d", "toggle_dark", "Theme"),
        Binding("ctrl+c", "cancel_job", "Cancel"),
    ]

    def __init__(self, project_dir: str = None):
        super().__init__()
        self.project_dir: str | None = project_dir
        self._cancel = False
        self._config: dict = {}
        self._ply_path: str | None = None
        self._device = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static("No project loaded", id="project-status")

        with TabbedContent():
            # --- Tab 1: Project ---
            with TabPane("Project", id="tab-project"):
                yield Label("Project Name", classes="form-label")
                yield Input(placeholder="my_scene", id="project-name", classes="form-input")
                yield Label("Input (video file or image directory)", classes="form-label")
                yield Input(placeholder="/path/to/video.mp4", id="input-path", classes="form-input")
                yield Label("Output Directory", classes="form-label")
                yield Input(placeholder="/home/dylan/splats", id="output-dir", classes="form-input")
                with Horizontal():
                    yield Button("Create Project", variant="primary", id="btn-create")
                    yield Button("Load Existing", variant="default", id="btn-load")

            # --- Tab 2: Extract ---
            with TabPane("Extract", id="tab-extract"):
                yield Label("Frames Per Second", classes="form-label")
                yield Input(value="2", id="fps", classes="form-input")
                yield Label("Resolution (optional, e.g. 1920:1080)", classes="form-label")
                yield Input(placeholder="leave blank for original", id="resolution", classes="form-input")
                yield Button("Extract Frames", variant="primary", id="btn-extract")

            # --- Tab 3: Reconstruct ---
            with TabPane("Reconstruct", id="tab-reconstruct"):
                yield Label("Camera Model", classes="form-label")
                yield Select[str](
                    [("SIMPLE_RADIAL", "SIMPLE_RADIAL"),
                     ("PINHOLE", "PINHOLE"),
                     ("OPENCV", "OPENCV"),
                     ("OPENCV_FISHEYE", "OPENCV_FISHEYE")],
                    value="SIMPLE_RADIAL", id="camera-model",
                )
                yield Label("Matcher", classes="form-label")
                yield Select[str](
                    [("Exhaustive (best for <500 images)", "exhaustive"),
                     ("Sequential (faster for video)", "sequential")],
                    value="exhaustive", id="matcher",
                )
                with Horizontal(classes="form-row"):
                    yield Switch(value=True, id="use-gpu")
                    yield Label("Use GPU (CUDA)")
                with Horizontal(classes="form-row"):
                    yield Switch(value=True, id="undistort")
                    yield Label("Undistort Images")
                yield Button("Run COLMAP", variant="primary", id="btn-colmap")

            # --- Tab 4: Train ---
            with TabPane("Train", id="tab-train"):
                yield Label("Iterations", classes="form-label")
                yield Input(value="30000", id="iterations", classes="form-input")
                yield Label("SH Degree", classes="form-label")
                yield Input(value="3", id="sh-degree", classes="form-input")
                yield Label("Conda Environment", classes="form-label")
                yield Input(value="gs_original", id="conda-env", classes="form-input")
                yield Button("Start Training", variant="primary", id="btn-train")

            # --- Tab 5: View ---
            with TabPane("View", id="tab-view"):
                yield Label("PLY File Path", classes="form-label")
                yield Input(placeholder="auto-detected after training", id="ply-path", classes="form-input")
                yield Label("Viewer Port", classes="form-label")
                yield Input(value="8765", id="viewer-port", classes="form-input")
                yield Button("Launch Viewer", variant="primary", id="btn-view")

        with Vertical(id="log-panel"):
            yield RichLog(id="log", highlight=True, markup=True, wrap=True)

        yield Footer()

    def on_mount(self) -> None:
        from pipeline.device import detect
        self._device = detect()
        self._log("[bold cyan]OttoSplatto[/] — Gaussian Splatting for any NVIDIA GPU")
        self._log("")
        for line in self._device.summary().splitlines():
            self._log(f"  {line}")
        self._log("")

        # Apply device-aware defaults to form fields
        td = self._device.training_defaults()
        try:
            self.query_one("#iterations", Input).value = str(td["iterations"])
            self.query_one("#sh-degree", Input).value = str(td["sh_degree"])
        except Exception:
            pass

        self._log("Create or load a project to begin.")
        if self.project_dir:
            self._load_project(self.project_dir)

    # ── helpers ──────────────────────────────────────────────

    def _log(self, msg: str) -> None:
        try:
            self.query_one("#log", RichLog).write(msg)
        except Exception:
            pass

    def _set_status(self, text: str) -> None:
        try:
            self.query_one("#project-status", Static).update(text)
        except Exception:
            pass

    def _check_cancel(self) -> bool:
        return self._cancel

    def _load_config(self) -> dict:
        cfg_path = os.path.join(self.project_dir, "project.json")
        if os.path.isfile(cfg_path):
            with open(cfg_path) as f:
                self._config = json.load(f)
        return self._config

    def _save_config(self) -> None:
        cfg_path = os.path.join(self.project_dir, "project.json")
        with open(cfg_path, "w") as f:
            json.dump(self._config, f, indent=2)

    def _load_project(self, project_dir: str) -> None:
        self.project_dir = project_dir
        self._load_config()
        self._set_status(f"Project: {self.project_dir}")
        self._log(f"[green]Loaded project:[/] {self.project_dir}")

        # Pre-fill PLY path if training output exists
        import glob
        plys = sorted(glob.glob(
            os.path.join(project_dir, "output", "point_cloud", "iteration_*", "point_cloud.ply")
        ))
        if plys:
            self._ply_path = plys[-1]
            try:
                self.query_one("#ply-path", Input).value = self._ply_path
            except Exception:
                pass

    # ── actions ──────────────────────────────────────────────

    def action_cancel_job(self) -> None:
        self._cancel = True
        self._log("[yellow]Cancel requested…[/]")

    # ── button dispatch ──────────────────────────────────────

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self._cancel = False
        btn = event.button.id
        if btn == "btn-create":
            self._do_create_project()
        elif btn == "btn-load":
            self._do_load_project()
        elif btn == "btn-extract":
            self._do_extract()
        elif btn == "btn-colmap":
            self._do_colmap()
        elif btn == "btn-train":
            self._do_train()
        elif btn == "btn-view":
            self._do_view()

    # ── workers ──────────────────────────────────────────────

    @work(thread=True)
    def _do_create_project(self) -> None:
        name = self.query_one("#project-name", Input).value.strip()
        input_path = self.query_one("#input-path", Input).value.strip()
        output_dir = self.query_one("#output-dir", Input).value.strip()

        if not all([name, input_path, output_dir]):
            self._log("[red]Fill in all three fields[/]")
            return

        if not os.path.exists(input_path):
            self._log(f"[red]Input not found:[/] {input_path}")
            return

        self.project_dir = os.path.join(output_dir, name)
        os.makedirs(os.path.join(self.project_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.project_dir, "output"), exist_ok=True)

        is_video = os.path.isfile(input_path)
        self._config = {
            "name": name,
            "input_path": os.path.abspath(input_path),
            "input_type": "video" if is_video else "images",
            "steps_completed": [],
        }
        self._save_config()
        self._set_status(f"Project: {self.project_dir}")
        self._log(f"[green]Project created:[/] {self.project_dir}")
        self._log(f"  Input type: {'video' if is_video else 'image directory'}")

    @work(thread=True)
    def _do_load_project(self) -> None:
        output_dir = self.query_one("#output-dir", Input).value.strip()
        name = self.query_one("#project-name", Input).value.strip()
        if output_dir and name:
            path = os.path.join(output_dir, name)
        elif output_dir:
            path = output_dir
        else:
            self._log("[red]Enter an output directory (and optionally a project name)[/]")
            return

        if os.path.isfile(os.path.join(path, "project.json")):
            self._load_project(path)
        else:
            self._log(f"[red]No project.json in {path}[/]")

    @work(thread=True)
    def _do_extract(self) -> None:
        if not self.project_dir:
            self._log("[red]Create or load a project first[/]")
            return

        self._load_config()
        input_path = self._config.get("input_path", "")
        input_type = self._config.get("input_type", "video")

        if input_type == "images":
            from pipeline.extract import copy_images
            self._log("Copying images into project…")
            result = copy_images(input_path, self.project_dir, on_output=self._log)
        else:
            from pipeline.extract import extract_frames
            fps = int(self.query_one("#fps", Input).value or "2")
            res = self.query_one("#resolution", Input).value.strip() or None
            self._log(f"Extracting frames at {fps} fps…")
            result = extract_frames(
                input_path, self.project_dir,
                fps=fps, resolution=res,
                on_output=self._log, check_cancel=self._check_cancel,
            )

        if result["status"] == "success":
            self._config.setdefault("steps_completed", [])
            if "extract" not in self._config["steps_completed"]:
                self._config["steps_completed"].append("extract")
            self._config["num_frames"] = result["num_frames"]
            self._save_config()
            self._log(f"[green]✓ {result['num_frames']} frames ready[/]")
        else:
            self._log(f"[red]Extraction failed: {result.get('message', result['status'])}[/]")

    @work(thread=True)
    def _do_colmap(self) -> None:
        if not self.project_dir:
            self._log("[red]Create or load a project first[/]")
            return

        from pipeline.reconstruct import run_colmap

        camera_model = self.query_one("#camera-model", Select).value
        matcher = self.query_one("#matcher", Select).value
        use_gpu = self.query_one("#use-gpu", Switch).value
        undistort = self.query_one("#undistort", Switch).value

        self._log(f"Running COLMAP ({camera_model}, {matcher}, gpu={use_gpu})…")

        result = run_colmap(
            self.project_dir,
            camera_model=camera_model, use_gpu=use_gpu,
            matcher=matcher, undistort=undistort,
            on_output=self._log, check_cancel=self._check_cancel,
        )

        if result["status"] == "success":
            self._load_config()
            if "colmap" not in self._config.get("steps_completed", []):
                self._config.setdefault("steps_completed", []).append("colmap")
            self._config["train_source"] = result["train_source"]
            self._save_config()
            self._log(f"[green]✓ COLMAP complete — {result['num_images']} images reconstructed[/]")
        else:
            self._log(f"[red]COLMAP failed at {result.get('step', '?')}: {result.get('message', '')}[/]")

    @work(thread=True)
    def _do_train(self) -> None:
        if not self.project_dir:
            self._log("[red]Create or load a project first[/]")
            return

        from pipeline.train import train

        self._load_config()
        source = self._config.get("train_source", self.project_dir)
        output = os.path.join(self.project_dir, "output")
        iterations = int(self.query_one("#iterations", Input).value or "30000")
        sh_degree = int(self.query_one("#sh-degree", Input).value or "3")
        conda_env = self.query_one("#conda-env", Input).value.strip() or "gs_original"

        self._log(f"Training {iterations} iterations (SH {sh_degree}) in env '{conda_env}'…")

        result = train(
            source_dir=source, output_dir=output,
            iterations=iterations, sh_degree=sh_degree,
            conda_env=conda_env,
            on_output=self._log, check_cancel=self._check_cancel,
        )

        if result["status"] == "success":
            self._ply_path = result.get("ply_path")
            if self._ply_path:
                try:
                    self.query_one("#ply-path", Input).value = self._ply_path
                except Exception:
                    pass
            self._load_config()
            if "train" not in self._config.get("steps_completed", []):
                self._config.setdefault("steps_completed", []).append("train")
            self._config["ply_path"] = self._ply_path
            self._save_config()
            self._log(f"[green]✓ Training complete![/]")
        else:
            self._log(f"[red]Training failed: {result.get('message', result['status'])}[/]")

    @work(thread=True)
    def _do_view(self) -> None:
        from pipeline.viewer import launch_viewer

        ply = self.query_one("#ply-path", Input).value.strip()
        if not ply and self._ply_path:
            ply = self._ply_path
        if not ply:
            self._log("[red]No PLY file specified[/]")
            return

        port = int(self.query_one("#viewer-port", Input).value or "8765")
        self._log(f"Launching viewer for {ply}…")
        result = launch_viewer(ply, port=port, on_output=self._log)

        if result["status"] == "success":
            self._log(f"[green]✓ Viewer at {result['url']}[/]")
        else:
            self._log(f"[red]Viewer error: {result.get('message', '')}[/]")
