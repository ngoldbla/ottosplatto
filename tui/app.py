"""OttoSplatto TUI — Textual-based terminal interface for Gaussian Splatting."""
import io
import os
import json
import socket
import subprocess
import sys
import shutil
import time
import glob as glob_module

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Button, DirectoryTree, Footer, Header, Input, Label, RichLog,
    Rule, Select, Static, Switch, TabbedContent, TabPane,
)
from textual import work
from rich.text import Text


class QRCodeScreen(ModalScreen[None]):
    """Full-screen modal showing a QR code for phone upload."""

    CSS = """
    QRCodeScreen {
        align: center middle;
    }
    #qr-dialog {
        width: 80%;
        max-width: 60;
        height: auto;
        max-height: 90%;
        border: thick $success;
        background: $surface;
        padding: 1 2;
    }
    #qr-title {
        text-align: center;
        text-style: bold;
        color: $success;
        margin-bottom: 1;
    }
    #qr-url {
        text-align: center;
        color: $text;
        text-style: bold;
        margin-bottom: 1;
    }
    #qr-code {
        text-align: center;
        margin-bottom: 1;
    }
    #qr-hint {
        text-align: center;
        color: $text-muted;
        margin-bottom: 1;
    }
    #qr-dismiss {
        width: 100%;
    }
    """

    def __init__(self, url: str, qr_text: str):
        super().__init__()
        self._url = url
        self._qr_text = qr_text

    def compose(self) -> ComposeResult:
        with Vertical(id="qr-dialog"):
            yield Static("Upload from Phone", id="qr-title")
            yield Static(self._url, id="qr-url")
            yield Static(self._qr_text, id="qr-code")
            yield Static("Scan this QR code or open the URL on your phone", id="qr-hint")
            yield Button("Got it — start uploading", variant="success", id="qr-dismiss")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "qr-dismiss":
            self.dismiss(None)


class PathPickerScreen(ModalScreen[str]):
    """Modal directory/file picker."""

    CSS = """
    PathPickerScreen {
        align: center middle;
    }
    #picker-dialog {
        width: 80%;
        max-width: 70;
        height: 80%;
        border: thick $primary;
        background: $surface;
        padding: 1 2;
    }
    #picker-dialog DirectoryTree {
        height: 1fr;
        margin-bottom: 1;
    }
    #picker-buttons {
        height: 3;
        align: center middle;
    }
    #picker-path {
        height: 1;
        color: $text-muted;
        padding: 0 1;
        margin-bottom: 1;
    }
    """

    def __init__(self, start_path: str = "~", title: str = "Select Path"):
        super().__init__()
        self._start = os.path.expanduser(start_path)
        self._title = title
        self._selected = self._start

    def compose(self) -> ComposeResult:
        with Vertical(id="picker-dialog"):
            yield Label(self._title, classes="form-label")
            yield Static(self._start, id="picker-path")
            yield DirectoryTree(self._start)
            with Horizontal(id="picker-buttons"):
                yield Button("Select", variant="primary", id="picker-select")
                yield Button("Cancel", variant="default", id="picker-cancel")

    def on_directory_tree_directory_selected(self, event: DirectoryTree.DirectorySelected) -> None:
        self._selected = str(event.path)
        self.query_one("#picker-path", Static).update(self._selected)

    def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected) -> None:
        self._selected = str(event.path)
        self.query_one("#picker-path", Static).update(self._selected)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "picker-select":
            self.dismiss(self._selected)
        elif event.button.id == "picker-cancel":
            self.dismiss("")

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
        height: 1fr;
        min-height: 8;
        max-height: 50%;
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
        overflow-y: auto;
    }
    .form-label {
        margin-top: 1;
        color: $text-muted;
    }
    .form-input {
        margin-bottom: 1;
    }
    .form-row {
        height: auto;
        align: left middle;
    }
    .form-row Label {
        margin-left: 1;
        width: auto;
    }
    .form-row Switch {
        height: auto;
    }
    Button {
        margin-top: 1;
        min-width: 16;
    }
    .browse-btn {
        min-width: 8;
        margin-top: 0;
        margin-left: 1;
    }
    .path-row {
        height: auto;
    }
    .path-row Input {
        width: 1fr;
    }
    .hidden {
        display: none;
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
    .help-text {
        color: $text-muted;
        margin-bottom: 1;
        padding: 0 2;
        max-width: 100%;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("d", "toggle_dark", "Theme"),
        Binding("ctrl+c", "cancel_job", "Cancel"),
        Binding("bracketright", "grow_log", "Log+", show=True),
        Binding("bracketleft", "shrink_log", "Log-", show=True),
    ]

    def __init__(self, project_dir: str = None):
        super().__init__()
        self.project_dir: str | None = project_dir
        self._cancel = False
        self._config: dict = {}
        self._ply_path: str | None = None
        self._device = None
        self._upload_done = False
        self._copyparty_proc: subprocess.Popen | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static("No project loaded", id="project-status")

        with TabbedContent():
            # --- Tab 1: Project ---
            with TabPane("Project", id="tab-project"):
                yield Static(
                    "Capture Tips for Best Results:\n"
                    "  Lock exposure before recording (tap & hold for AE/AF Lock)\n"
                    "  Use main 1x lens — avoid ultra-wide or telephoto\n"
                    "  Move slowly in a spiral pattern at multiple heights\n"
                    "  70-80% overlap between frames\n"
                    "  Minimum: 50 photos (small), 100+ (room), 200+ (building)",
                    classes="help-text",
                )
                yield Label("Project Name", classes="form-label")
                yield Input(placeholder="my_scene", id="project-name", classes="form-input")
                yield Label("Input (video file or image directory)", classes="form-label")
                with Horizontal(classes="path-row"):
                    yield Input(placeholder="/path/to/video.mp4 or /path/to/images/", id="input-path", classes="form-input")
                    yield Button("Browse", variant="default", id="btn-browse-input", classes="browse-btn")
                yield Label("Output Directory", classes="form-label")
                with Horizontal(classes="path-row"):
                    yield Input(placeholder="/home/dylan/splats", id="output-dir", classes="form-input")
                    yield Button("Browse", variant="default", id="btn-browse-output", classes="browse-btn")
                with Horizontal():
                    yield Button("Create Project", variant="primary", id="btn-create")
                    yield Button("Load Existing", variant="default", id="btn-load")
                    yield Button("Upload from Phone", variant="success", id="btn-upload")
                    yield Button("Done Uploading", variant="warning", id="btn-upload-done", classes="hidden")

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
                yield Static(
                    "SIMPLE_RADIAL — Best for phone cameras (iPhone, Android). "
                    "Models one focal length + one radial distortion parameter. "
                    "Use this unless you have a reason not to.\n"
                    "PINHOLE — For calibrated cameras with no lens distortion. "
                    "Rarely needed for phone photos.\n"
                    "OPENCV — Full distortion model. Use for action cameras "
                    "(GoPro) or wide-angle lenses.\n"
                    "OPENCV_FISHEYE — For ultra-wide/fisheye lenses (>180° FOV).",
                    classes="help-text",
                )
                yield Label("Matcher", classes="form-label")
                yield Select[str](
                    [("Exhaustive (best for <500 images)", "exhaustive"),
                     ("Sequential (faster for video)", "sequential")],
                    value="exhaustive", id="matcher",
                )
                yield Static(
                    "Exhaustive — Compares every image pair. Slower but finds all matches. "
                    "Best for unordered photos (e.g. walking around an object).\n"
                    "Sequential — Only compares neighboring frames. Much faster for "
                    "video-extracted frames where order is known.",
                    classes="help-text",
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
                yield Label("Training Method", classes="form-label")
                yield Select[str](
                    [("Original 3DGS (fast, good quality)", "original"),
                     ("2D Gaussian Splatting (better surfaces, fewer artifacts)", "2dgs")],
                    value="2dgs", id="train-method",
                )
                yield Static(
                    "Original 3DGS — Fast training, good general quality. May produce floaters and needles.\n"
                    "2D Gaussian Splatting — Better surface reconstruction, fewer artifacts. "
                    "Recommended for emergency response scenes with structures and terrain.",
                    classes="help-text",
                )
                yield Label("Iterations", classes="form-label")
                yield Input(value="30000", id="iterations", classes="form-input")
                yield Static(
                    "How many optimization steps to run. More = better quality but slower.\n"
                    "7000 — Quick preview (~2 min on RTX 4090)\n"
                    "30000 — Good quality (~8 min on RTX 4090, recommended)\n"
                    "50000+ — Diminishing returns, use for final output only",
                    classes="help-text",
                )
                yield Label("SH Degree", classes="form-label")
                yield Input(value="3", id="sh-degree", classes="form-input")
                yield Static(
                    "Spherical Harmonics degree controls color fidelity.\n"
                    "0 — Flat color per splat (fastest, lowest quality)\n"
                    "3 — Full view-dependent color (recommended for most scenes)",
                    classes="help-text",
                )
                yield Label("Conda Environment", classes="form-label")
                yield Input(value="gs_original", id="conda-env", classes="form-input")
                yield Static(
                    "The conda environment with the original 3DGS training code installed.\n"
                    "Leave as 'gs_original' unless you have a custom setup.",
                    classes="help-text",
                )
                yield Button("Start Training", variant="primary", id="btn-train")

            # --- Tab 5: View ---
            with TabPane("View", id="tab-view"):
                yield Label("PLY File Path", classes="form-label")
                yield Input(placeholder="auto-detected after training", id="ply-path", classes="form-input")
                yield Label("Viewer Port", classes="form-label")
                yield Input(value="8765", id="viewer-port", classes="form-input")
                with Horizontal():
                    yield Button("Launch Viewer", variant="primary", id="btn-view")
                    yield Button("Clean Splat", variant="warning", id="btn-clean")

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

    def _switch_tab(self, tab_id: str) -> None:
        """Switch to a tab by ID. Safe to call from worker threads."""
        try:
            # Use run_worker with thread=False to schedule on the main thread
            self.call_later(self._apply_tab_switch, tab_id)
        except Exception:
            pass

    def _apply_tab_switch(self, tab_id: str) -> None:
        """Actually switch the tab — must run on the main thread."""
        try:
            tc = self.query_one(TabbedContent)
            tc.active = tab_id
        except Exception as e:
            self._log(f"[yellow]Tab switch issue: {e}[/]")

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
        plys = sorted(glob_module.glob(
            os.path.join(project_dir, "output", "point_cloud", "iteration_*", "point_cloud.ply")
        ))
        if plys:
            self._ply_path = plys[-1]
            try:
                self.query_one("#ply-path", Input).value = self._ply_path
            except Exception:
                pass

    def on_unmount(self) -> None:
        if self._copyparty_proc is not None:
            try:
                self._copyparty_proc.terminate()
                self._copyparty_proc.wait(timeout=5)
            except Exception:
                pass

    @staticmethod
    def _get_lan_ip() -> str:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    def _do_finish_upload(self) -> None:
        """Called on 'Done Uploading' button press — sets the flag for the worker."""
        self._upload_done = True

    # ── actions ──────────────────────────────────────────────

    def action_cancel_job(self) -> None:
        self._cancel = True
        self._log("[yellow]Cancel requested…[/]")

    def action_grow_log(self) -> None:
        """Increase log panel height."""
        panel = self.query_one("#log-panel")
        current = panel.styles.height
        if current is not None and hasattr(current, 'value'):
            new_val = min(current.value + 4, 80)
            panel.styles.height = new_val
        else:
            panel.styles.height = 20

    def action_shrink_log(self) -> None:
        """Decrease log panel height."""
        panel = self.query_one("#log-panel")
        current = panel.styles.height
        if current is not None and hasattr(current, 'value'):
            new_val = max(current.value - 4, 6)
            panel.styles.height = new_val
        else:
            panel.styles.height = 10

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
        elif btn == "btn-upload":
            self._do_upload_from_phone()
        elif btn == "btn-upload-done":
            self._do_finish_upload()
        elif btn == "btn-view":
            self._do_view()
        elif btn == "btn-clean":
            self._do_clean_splat()
        elif btn == "btn-browse-input":
            self._browse_path("input-path", "Select Input (video or image folder)")
        elif btn == "btn-browse-output":
            self._browse_path("output-dir", "Select Output Directory")

    def _browse_path(self, input_id: str, title: str) -> None:
        """Open a directory picker and fill the result into an Input widget."""
        current = self.query_one(f"#{input_id}", Input).value.strip()
        start = current if current and os.path.exists(current) else os.path.expanduser("~")

        def _on_result(path: str) -> None:
            if path:
                self.query_one(f"#{input_id}", Input).value = path

        self.push_screen(PathPickerScreen(start, title), _on_result)

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

        if is_video:
            # Video input — user needs to extract frames next
            self._log("[cyan]Video input detected — advancing to Extract tab[/]")
            self._switch_tab("tab-extract")
        else:
            # Image directory — copy images into project and skip extraction
            src_images = (
                glob_module.glob(os.path.join(input_path, "*.jpg"))
                + glob_module.glob(os.path.join(input_path, "*.JPG"))
                + glob_module.glob(os.path.join(input_path, "*.jpeg"))
                + glob_module.glob(os.path.join(input_path, "*.png"))
            )
            dest = os.path.join(self.project_dir, "images")
            for img in src_images:
                shutil.copy2(img, dest)

            num_images = len(src_images)
            self._config["num_frames"] = num_images
            if "extract" not in self._config["steps_completed"]:
                self._config["steps_completed"].append("extract")
            self._save_config()

            self._log(
                f"[green]{num_images} images detected[/] — skipping frame extraction"
            )

            # Check for transforms.json (app-provided camera poses — skip COLMAP)
            from pipeline.transforms_import import detect_transforms, import_transforms
            tf_path = detect_transforms(input_path)
            if tf_path:
                self._log("[bold green]✓ transforms.json found — camera poses provided by capture app![/]")
                self._log("[cyan]Importing poses and skipping COLMAP entirely…[/]")
                result = import_transforms(input_path, self.project_dir, on_output=self._log)
                if result["status"] == "success":
                    self._config["train_source"] = result["train_source"]
                    if "colmap" not in self._config["steps_completed"]:
                        self._config["steps_completed"].append("colmap")
                    self._config["has_transforms"] = True
                    self._save_config()
                    self._log(f"[bold green]✓ Ready for training — {result['num_images']} images with poses[/]")
                    self._switch_tab("tab-train")
                    return
                else:
                    self._log(f"[yellow]⚠ transforms.json import failed: {result.get('message', '')}[/]")
                    self._log("[cyan]Falling back to COLMAP reconstruction[/]")

            # Auto-detect camera from EXIF and recommend settings
            from pipeline.exif_detect import detect_camera
            detection = detect_camera(os.path.join(self.project_dir, "images"), on_output=self._log)
            if detection["status"] == "detected":
                cam_model = detection["camera_model"]
                matcher_val = detection["matcher"]
                self.app.call_later(
                    lambda v=cam_model: setattr(self.query_one("#camera-model", Select), "value", v)
                )
                self.app.call_later(
                    lambda v=matcher_val: setattr(self.query_one("#matcher", Select), "value", v)
                )
                self._config["single_camera"] = detection.get("single_camera", False)
                self._save_config()

            self._log("[cyan]Advancing to Reconstruct tab[/]")
            self._switch_tab("tab-reconstruct")

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
    def _do_upload_from_phone(self) -> None:
        import qrcode

        name = self.query_one("#project-name", Input).value.strip()
        output_dir = self.query_one("#output-dir", Input).value.strip()

        if not name or not output_dir:
            self._log("[red]Fill in Project Name and Output Directory[/]")
            return

        self.project_dir = os.path.join(output_dir, name)
        images_dir = os.path.join(self.project_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(os.path.join(self.project_dir, "output"), exist_ok=True)

        self._config = {
            "name": name,
            "input_path": images_dir,
            "input_type": "images",
            "steps_completed": [],
        }
        self._save_config()
        self._set_status(f"Project: {self.project_dir}")

        # Kill any existing process on port 3210 to avoid conflicts
        try:
            result = subprocess.run(
                ["lsof", "-ti", ":3210"], capture_output=True, text=True
            )
            if result.stdout.strip():
                for pid in result.stdout.strip().split():
                    try:
                        os.kill(int(pid), 9)
                    except (ProcessLookupError, ValueError):
                        pass
                time.sleep(1)
                self._log("[yellow]Killed existing process on port 3210[/]")
        except Exception:
            pass

        # Start copyparty serving the project root (not just images/)
        # This way transforms.json and pointcloud.ply land at the right level
        self._log(f"Starting upload server for {self.project_dir}")
        try:
            self._copyparty_proc = subprocess.Popen(
                ["copyparty", "-v", f"{self.project_dir}::rw", "--http-only", "-p", "3210", "-q"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(2)  # Give copyparty time to bind the port
            if self._copyparty_proc.poll() is not None:
                self._log("[red]copyparty failed to start (port 3210 may be in use)[/]")
                return
        except FileNotFoundError:
            self._log("[red]copyparty not found — install with: pip install copyparty[/]")
            return
        except OSError as e:
            self._log(f"[red]Failed to start copyparty: {e}[/]")
            return

        # Detect LAN IP and generate QR code
        lan_ip = self._get_lan_ip()
        url = f"http://{lan_ip}:3210/"

        if lan_ip == "127.0.0.1":
            self._log("[yellow]Warning: Could not detect LAN IP, using 127.0.0.1[/]")

        qr = qrcode.QRCode(box_size=1, border=1)
        qr.add_data(url)
        qr.make()
        buf = io.StringIO()
        qr.print_ascii(out=buf, invert=True)
        qr_text = buf.getvalue()

        self._log(f"[bold green]Upload URL:[/] {url}")
        self._log("Scan the QR code or open the URL on your phone")

        # Show QR as a full-screen modal so it's readable
        self.app.call_from_thread(
            lambda: self.push_screen(QRCodeScreen(url, qr_text))
        )

        # Swap buttons: hide Upload, show Done Uploading
        self._upload_done = False
        self.app.call_from_thread(
            lambda: (
                self.query_one("#btn-upload", Button).add_class("hidden"),
                self.query_one("#btn-upload-done", Button).remove_class("hidden"),
            )
        )

        # Poll loop — count images recursively (uploads may land in subdirs)
        IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif"}
        last_count = 0
        while not self._upload_done:
            time.sleep(2)
            try:
                count = 0
                for root, dirs, files in os.walk(images_dir):
                    # Skip copyparty metadata
                    dirs[:] = [d for d in dirs if not d.startswith(".")]
                    count += sum(
                        1 for f in files if os.path.splitext(f)[1].lower() in IMAGE_EXTS
                    )
                if count != last_count:
                    self._log(f"[cyan]📸 {count} images received…[/]")
                    last_count = count
            except Exception:
                pass

        # ── Cleanup ──
        # Terminate copyparty
        if self._copyparty_proc is not None:
            try:
                self._copyparty_proc.terminate()
                self._copyparty_proc.wait(timeout=5)
            except Exception:
                pass
            self._copyparty_proc = None

        # Sort uploaded files: images → images/, special files → project root
        # Phone uploads may flatten everything into one directory
        SPECIAL_FILES = {"transforms.json", "pointcloud.ply", "point_cloud.ply"}
        project_root = self.project_dir

        # Walk all files in the project dir and sort them
        for root, dirs, files in os.walk(project_root):
            # Skip .hist (copyparty metadata), output, sparse dirs
            dirs[:] = [d for d in dirs if d not in {".hist", "output", "sparse"}]
            for f in files:
                src = os.path.join(root, f)
                ext = os.path.splitext(f)[1].lower()
                fname_lower = f.lower()

                if fname_lower in SPECIAL_FILES:
                    # Special files go to project root
                    dst = os.path.join(project_root, f)
                    if src != dst:
                        shutil.move(src, dst)
                        self._log(f"[green]Found {f}[/]")
                elif ext in IMAGE_EXTS:
                    # Images go to images/
                    dst = os.path.join(images_dir, f)
                    if src != dst and not os.path.exists(dst):
                        shutil.move(src, dst)

        # Clean up empty subdirs (except images/, output/, sparse/)
        for d in os.listdir(project_root):
            dp = os.path.join(project_root, d)
            if os.path.isdir(dp) and d not in {"images", "output", "sparse", ".hist"}:
                try:
                    shutil.rmtree(dp)
                except Exception:
                    pass

        # Final image count
        try:
            files = os.listdir(images_dir)
            final_count = sum(
                1 for f in files if os.path.splitext(f)[1].lower() in IMAGE_EXTS
            )
        except Exception:
            final_count = 0

        if final_count == 0:
            self._log("[red]No images found after upload.[/]")
            self._log("[yellow]Tip: make sure you uploaded image files (JPG/PNG/HEIC), not a zip or other archive.[/]")
        else:
            self._log(f"[green]✓ Upload complete — {final_count} images in project[/]")

        # Convert HEIC to JPEG if needed
        from pipeline.convert import convert_heic_to_jpeg
        conv = convert_heic_to_jpeg(images_dir, on_output=self._log)
        if conv.get("converted", 0) > 0:
            self._log(f"[green]✓ Converted {conv['converted']} HEIC files to JPEG[/]")
            # Recount after conversion
            files = os.listdir(images_dir)
            final_count = sum(
                1 for f in files if os.path.splitext(f)[1].lower() in {".jpg", ".jpeg", ".png"}
            )

        # Run EXIF detection
        from pipeline.exif_detect import detect_camera
        detection = detect_camera(images_dir, on_output=self._log)
        if detection.get("single_camera"):
            self._config["single_camera"] = True
        # Auto-fill Reconstruct tab
        if detection["status"] == "detected":
            cam_model = detection["camera_model"]
            matcher_val = detection["matcher"]
            try:
                self.app.call_from_thread(
                    lambda: self.query_one("#camera-model", Select).__setattr__("value", cam_model)
                )
                self.app.call_from_thread(
                    lambda: self.query_one("#matcher", Select).__setattr__("value", matcher_val)
                )
            except Exception:
                pass

        # Save config
        self._config["num_frames"] = final_count
        if "extract" not in self._config.get("steps_completed", []):
            self._config.setdefault("steps_completed", []).append("extract")
        self._save_config()

        # Swap buttons back: show Upload, hide Done Uploading
        self.app.call_from_thread(
            lambda: (
                self.query_one("#btn-upload", Button).remove_class("hidden"),
                self.query_one("#btn-upload-done", Button).add_class("hidden"),
            )
        )

        if final_count > 0:
            # Check for transforms.json (app-provided poses — skip COLMAP)
            from pipeline.transforms_import import detect_transforms, import_transforms
            tf_path = detect_transforms(self.project_dir) or detect_transforms(images_dir)
            if tf_path:
                self._log("[bold green]✓ transforms.json found — camera poses from capture app![/]")
                self._log("[cyan]Importing poses — COLMAP will be skipped[/]")
                tf_dir = os.path.dirname(tf_path)
                result = import_transforms(tf_dir, self.project_dir, on_output=self._log)
                if result["status"] == "success":
                    self._config["train_source"] = result["train_source"]
                    if "colmap" not in self._config.get("steps_completed", []):
                        self._config.setdefault("steps_completed", []).append("colmap")
                    self._config["has_transforms"] = True
                    self._save_config()
                    self._log(f"[bold green]✓ Ready for training![/] {final_count} images with poses → advancing to Train")
                    self._switch_tab("tab-train")
                    return

            self._log(f"[bold green]Ready to reconstruct![/] {final_count} images → advancing to Reconstruct tab")
            self._switch_tab("tab-reconstruct")
        else:
            self._log("[yellow]Upload the images and try again.[/]")

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
            # Auto-advance to Reconstruct tab
            self._log("[cyan]Advancing to Reconstruct tab[/]")
            self._switch_tab("tab-reconstruct")
        else:
            self._log(f"[red]Extraction failed: {result.get('message', result['status'])}[/]")

    @work(thread=True)
    def _do_colmap(self) -> None:
        if not self.project_dir:
            self._log("[red]Create or load a project first[/]")
            return

        # Pre-check image quality before COLMAP
        from pipeline.precheck import precheck_images, cull_blurry_frames
        images_dir = os.path.join(self.project_dir, "images")
        check = precheck_images(images_dir, on_output=self._log)
        if check.get("score") == "POOR":
            self._log("[yellow]⚠ Image quality is POOR — reconstruction may fail. Consider adding more images or removing blurry ones.[/]")

        # Auto-cull blurry frames if any were detected
        if check.get("blur_scores") and any("very blurry" in i for i in check.get("issues", [])):
            self._log("[cyan]Auto-culling blurry frames…[/]")
            cull_result = cull_blurry_frames(images_dir, on_output=self._log)
            if cull_result["culled"] > 0:
                self._log(f"[green]✓ Moved {cull_result['culled']} blurry frames to culled/[/]")

        from pipeline.reconstruct import run_colmap

        camera_model = self.query_one("#camera-model", Select).value
        matcher = self.query_one("#matcher", Select).value
        use_gpu = self.query_one("#use-gpu", Switch).value
        undistort = self.query_one("#undistort", Switch).value

        # Check if EXIF detection found a single camera (shared intrinsics)
        self._load_config()
        single_camera = self._config.get("single_camera", False)

        self._log(f"Running COLMAP ({camera_model}, {matcher}, gpu={use_gpu}, single_camera={single_camera})…")

        # Auto-convert HEIC to JPEG before COLMAP
        from pipeline.convert import convert_heic_to_jpeg
        conv = convert_heic_to_jpeg(images_dir, on_output=self._log)
        if conv.get("converted", 0) > 0:
            self._log(f"[green]✓ Converted {conv['converted']} HEIC files to JPEG[/]")

        result = run_colmap(
            self.project_dir,
            camera_model=camera_model, use_gpu=use_gpu,
            matcher=matcher, undistort=undistort,
            single_camera=single_camera,
            on_output=self._log, check_cancel=self._check_cancel,
        )

        if result["status"] == "success":
            self._load_config()
            if "colmap" not in self._config.get("steps_completed", []):
                self._config.setdefault("steps_completed", []).append("colmap")
            self._config["train_source"] = result["train_source"]
            self._save_config()
            self._log(f"[green]✓ COLMAP complete — {result['num_images']} images reconstructed[/]")
            # Auto-advance to Train tab
            self._log("[cyan]Advancing to Train tab[/]")
            self._switch_tab("tab-train")
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
        method = self.query_one("#train-method", Select).value
        conda_env = "gs_2dgs" if method == "2dgs" else self.query_one("#conda-env", Input).value.strip() or "gs_original"

        self._log(f"Training {iterations} iterations (SH {sh_degree}) method='{method}' env='{conda_env}'…")

        result = train(
            source_dir=source, output_dir=output,
            iterations=iterations, sh_degree=sh_degree,
            conda_env=conda_env, method=method,
            on_output=self._log, check_cancel=self._check_cancel,
        )

        if result["status"] == "success":
            self._ply_path = result.get("ply_path")

            # Display loss curve
            loss_history = result.get("loss_history", [])
            if loss_history and len(loss_history) >= 2:
                losses = [l for _, l in loss_history]
                min_l, max_l = min(losses), max(losses)
                if max_l > min_l:
                    chars = "▁▂▃▄▅▆▇█"
                    sparkline = ""
                    for l in losses:
                        idx = int((l - min_l) / (max_l - min_l) * (len(chars) - 1))
                        sparkline += chars[idx]
                    self._log(f"  Loss curve: {sparkline}")
                    self._log(f"  Start: {losses[0]:.4f} → Final: {losses[-1]:.4f}")

            # ── Post-training: cleanup splat ──────────────────
            if self._ply_path and os.path.isfile(self._ply_path):
                self._log("")
                self._log("[cyan]Running post-training cleanup…[/]")
                from pipeline.cleanup import cleanup_splat
                cleanup_result = cleanup_splat(self._ply_path, on_output=self._log)
                if cleanup_result["status"] == "success":
                    removed = cleanup_result["removed"]
                    self._log(
                        f"[green]✓ Cleanup: removed {removed['total']} gaussians "
                        f"({cleanup_result['input_count']} → {cleanup_result['output_count']})[/]"
                    )
                    self._log(
                        f"  invisible={removed['invisible']} needles={removed['needles']} "
                        f"outliers={removed['outliers']} giants={removed['giants']}"
                    )
                    # Use the cleaned PLY going forward
                    self._ply_path = cleanup_result["output_path"]
                else:
                    self._log(f"[yellow]Cleanup skipped: {cleanup_result.get('message', 'unknown error')}[/]")

            # ── Post-training: generate manifest ──────────────
            if self._ply_path and self.project_dir:
                self._log("")
                from pipeline.manifest import generate_manifest
                manifest_result = generate_manifest(
                    self.project_dir, self._ply_path, on_output=self._log
                )
                if manifest_result["status"] == "success":
                    self._log(f"[green]✓ Manifest saved[/]")
                else:
                    self._log(f"[yellow]Manifest skipped: {manifest_result.get('message', '')}[/]")

            if self._ply_path:
                try:
                    ply_for_ui = self._ply_path
                    self.app.call_from_thread(
                        lambda: setattr(
                            self.query_one("#ply-path", Input), "value", ply_for_ui
                        )
                    )
                except Exception:
                    pass
            self._load_config()
            if "train" not in self._config.get("steps_completed", []):
                self._config.setdefault("steps_completed", []).append("train")
            self._config["ply_path"] = self._ply_path
            self._save_config()
            self._log(f"[green]✓ Training complete![/]")
            # Auto-advance to View tab with PLY path pre-filled
            self._log("[cyan]Advancing to View tab[/]")
            self._switch_tab("tab-view")
        else:
            self._log(f"[red]Training failed: {result.get('message', result['status'])}[/]")

    @work(thread=True)
    def _do_clean_splat(self) -> None:
        """Run cleanup on the current PLY file from the View tab."""
        ply = self.query_one("#ply-path", Input).value.strip()
        if not ply and self._ply_path:
            ply = self._ply_path
        if not ply:
            self._log("[red]No PLY file specified[/]")
            return
        if not os.path.isfile(ply):
            self._log(f"[red]PLY not found: {ply}[/]")
            return

        from pipeline.cleanup import cleanup_splat
        self._log(f"Cleaning {ply}…")
        result = cleanup_splat(ply, on_output=self._log)

        if result["status"] == "success":
            removed = result["removed"]
            self._log(
                f"[green]✓ Cleanup: removed {removed['total']} gaussians "
                f"({result['input_count']} → {result['output_count']})[/]"
            )
            self._ply_path = result["output_path"]
            try:
                cleaned_path = result["output_path"]
                self.app.call_from_thread(
                    lambda: setattr(
                        self.query_one("#ply-path", Input), "value", cleaned_path
                    )
                )
            except Exception:
                pass

            # Also generate manifest if we have a project dir
            if self.project_dir:
                from pipeline.manifest import generate_manifest
                manifest_result = generate_manifest(
                    self.project_dir, self._ply_path, on_output=self._log
                )
                if manifest_result["status"] == "success":
                    self._log(f"[green]✓ Manifest updated[/]")
        else:
            self._log(f"[red]Cleanup failed: {result.get('message', result['status'])}[/]")

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

        if os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland":
            self._log("[yellow]⚠ Wayland session detected — launching Chrome with X11 backend for WebGL[/]")

        result = launch_viewer(ply, port=port, on_output=self._log)

        if result["status"] == "success":
            self._log(f"[green]✓ Viewer at {result['url']}[/]")
        else:
            self._log(f"[red]Viewer error: {result.get('message', '')}[/]")
