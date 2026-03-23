"""Device detection and adaptation for multi-GPU/multi-architecture support.

Supports:
  - Consumer NVIDIA GPUs (RTX 4090, 3090, etc.) on x86_64 Linux
  - NVIDIA DGX Spark (Grace Blackwell, ARM aarch64)
  - Multi-GPU setups (DGX, workstations)
"""
import subprocess
import platform
import os
import re
import json
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class GPU:
    index: int
    name: str
    memory_mb: int
    compute_cap: str
    driver_version: str
    cuda_cores: int = 0  # estimated

    @property
    def memory_gb(self) -> float:
        return self.memory_mb / 1024

    @property
    def is_blackwell(self) -> bool:
        return self.compute_cap.startswith("10") or "blackwell" in self.name.lower() or "b200" in self.name.lower()

    @property
    def is_ada(self) -> bool:
        return self.compute_cap.startswith("8.9") or "4090" in self.name or "4080" in self.name

    @property
    def is_hopper(self) -> bool:
        return self.compute_cap.startswith("9.0") or "h100" in self.name.lower()

    @property
    def family(self) -> str:
        if self.is_blackwell:
            return "blackwell"
        if self.is_hopper:
            return "hopper"
        if self.is_ada:
            return "ada"
        if self.compute_cap.startswith("8."):
            return "ampere"
        return "unknown"


@dataclass
class DeviceProfile:
    hostname: str
    arch: str  # x86_64, aarch64
    os_name: str
    gpus: list[GPU] = field(default_factory=list)
    cuda_version: str = ""
    total_ram_gb: float = 0.0
    is_dgx: bool = False
    dgx_model: str = ""

    @property
    def num_gpus(self) -> int:
        return len(self.gpus)

    @property
    def primary_gpu(self) -> Optional[GPU]:
        return self.gpus[0] if self.gpus else None

    @property
    def total_vram_gb(self) -> float:
        return sum(g.memory_gb for g in self.gpus)

    @property
    def is_arm(self) -> bool:
        return self.arch in ("aarch64", "arm64")

    def training_defaults(self) -> dict:
        """Return recommended training defaults based on hardware."""
        gpu = self.primary_gpu
        if not gpu:
            return {"iterations": 30000, "sh_degree": 3, "densify_grad_threshold": 0.0002}

        defaults = {
            "iterations": 30000,
            "sh_degree": 3,
            "densify_grad_threshold": 0.0002,
        }

        # High-VRAM GPUs can afford more aggressive settings
        if gpu.memory_gb >= 40:  # DGX Spark, A100, H100
            defaults["sh_degree"] = 4
            defaults["densify_grad_threshold"] = 0.00015
        elif gpu.memory_gb >= 20:  # 4090, 3090
            defaults["sh_degree"] = 3
        else:  # lower-end
            defaults["sh_degree"] = 2
            defaults["iterations"] = 20000

        return defaults

    def colmap_defaults(self) -> dict:
        """Return recommended COLMAP settings based on hardware."""
        gpu = self.primary_gpu
        defaults = {
            "use_gpu": True,
            "camera_model": "SIMPLE_RADIAL",
            "matcher": "exhaustive",
        }

        if not gpu:
            defaults["use_gpu"] = False
            return defaults

        # DGX / multi-GPU: COLMAP only uses one GPU but has plenty of RAM
        if gpu.memory_gb >= 40:
            defaults["matcher"] = "exhaustive"
        elif gpu.memory_gb >= 20:
            defaults["matcher"] = "exhaustive"
        else:
            defaults["matcher"] = "sequential"

        return defaults

    def cuda_visible_devices(self, count: int = 1) -> str:
        """Return CUDA_VISIBLE_DEVICES string for the best N GPUs."""
        if not self.gpus:
            return "0"
        # Sort by VRAM descending, pick top N
        ranked = sorted(self.gpus, key=lambda g: g.memory_mb, reverse=True)
        return ",".join(str(g.index) for g in ranked[:count])

    def summary(self) -> str:
        lines = [f"Host: {self.hostname} ({self.arch})"]
        if self.is_dgx:
            lines[0] += f"  [DGX {self.dgx_model}]"
        lines.append(f"OS: {self.os_name}")
        lines.append(f"CUDA: {self.cuda_version}")
        lines.append(f"RAM: {self.total_ram_gb:.0f} GB")
        lines.append(f"GPUs: {self.num_gpus} ({self.total_vram_gb:.0f} GB total VRAM)")
        for g in self.gpus:
            lines.append(f"  [{g.index}] {g.name}  {g.memory_gb:.0f}GB  sm_{g.compute_cap}  ({g.family})")
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def detect() -> DeviceProfile:
    """Detect the current device profile."""
    profile = DeviceProfile(
        hostname=platform.node(),
        arch=platform.machine(),
        os_name=_get_os_name(),
    )

    # CUDA version
    profile.cuda_version = _get_cuda_version()

    # System RAM
    profile.total_ram_gb = _get_ram_gb()

    # DGX detection
    profile.is_dgx, profile.dgx_model = _detect_dgx()

    # GPUs
    profile.gpus = _detect_gpus()

    return profile


def _get_os_name() -> str:
    try:
        result = subprocess.run(
            ["lsb_release", "-ds"], capture_output=True, text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().strip('"')
    except FileNotFoundError:
        pass

    try:
        with open("/etc/os-release") as f:
            for line in f:
                if line.startswith("PRETTY_NAME="):
                    return line.split("=", 1)[1].strip().strip('"')
    except FileNotFoundError:
        pass

    return f"{platform.system()} {platform.release()}"


def _get_cuda_version() -> str:
    try:
        result = subprocess.run(
            ["nvcc", "--version"], capture_output=True, text=True,
        )
        match = re.search(r"release (\d+\.\d+)", result.stdout)
        if match:
            return match.group(1)
    except FileNotFoundError:
        pass
    return ""


def _get_ram_gb() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb / (1024 * 1024)
    except FileNotFoundError:
        pass
    return 0.0


def _detect_dgx() -> tuple[bool, str]:
    """Check if running on an NVIDIA DGX system."""
    # Check for DGX OS marker
    markers = [
        "/etc/dgx-release",
        "/etc/nv-dgx-release",
    ]
    for marker in markers:
        if os.path.isfile(marker):
            try:
                with open(marker) as f:
                    content = f.read()
                    for line in content.splitlines():
                        if "DGX_NAME" in line or "PRODUCT_NAME" in line:
                            model = line.split("=", 1)[1].strip().strip('"')
                            return True, model
            except Exception:
                pass
            return True, "unknown"

    # Check hostname or product name
    try:
        with open("/sys/devices/virtual/dmi/id/product_name") as f:
            product = f.read().strip()
            if "dgx" in product.lower():
                return True, product
    except FileNotFoundError:
        pass

    # Check for DGX Spark specifically (Grace Blackwell desktop)
    if platform.machine() == "aarch64":
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True,
            )
            names = result.stdout.strip().lower()
            if "blackwell" in names or "b200" in names or "gb" in names:
                return True, "DGX Spark"
        except FileNotFoundError:
            pass

    return False, ""


def _detect_gpus() -> list[GPU]:
    """Detect all NVIDIA GPUs via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.total,compute_cap,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            return []
    except FileNotFoundError:
        return []

    gpus = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 5:
            gpus.append(GPU(
                index=int(parts[0]),
                name=parts[1],
                memory_mb=int(float(parts[2])),
                compute_cap=parts[3],
                driver_version=parts[4],
            ))
    return gpus
