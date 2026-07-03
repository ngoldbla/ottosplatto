"""Device detection and adaptation for multi-GPU/multi-architecture support.

Supports:
  - Consumer NVIDIA GPUs from Kepler through Blackwell (GTX 900/10-series,
    RTX 20/30/40/50-series, Titan, Quadro/RTX A-series, Tesla/datacenter)
  - NVIDIA DGX Spark (Grace Blackwell, ARM aarch64)
  - Multi-GPU setups (DGX, workstations)

Hardware-aware defaults scale down for low-VRAM cards (6-8 GB) and flag
legacy architectures (pre-Turing, compute capability < 7.0) that need
older PyTorch wheels / CUDA toolkits — see docs/legacy-gpu-setup.md.
"""
import subprocess
import platform
import os
import re
import json
from dataclasses import dataclass, field, asdict
from typing import Optional


# Compute capability by architecture family (major version):
#   3.x Kepler   5.x Maxwell   6.x Pascal   7.0/7.2 Volta   7.5 Turing
#   8.0/8.6/8.7 Ampere   8.9 Ada   9.0 Hopper   10.x/12.x Blackwell
_FAMILY_BY_CC = [
    ((12, 0), "blackwell"),
    ((10, 0), "blackwell"),
    ((9, 0), "hopper"),
    ((8, 9), "ada"),
    ((8, 0), "ampere"),
    ((7, 5), "turing"),
    ((7, 0), "volta"),
    ((6, 0), "pascal"),
    ((5, 0), "maxwell"),
    ((3, 0), "kepler"),
]

# Fallback: infer compute capability from the GPU name when nvidia-smi does
# not report it (the compute_cap query field requires driver >= 510).
_NAME_CC_HINTS = [
    (r"\brtx\s*50[5-9]0", "12.0"),
    (r"\bb[12]00\b|\bgb10\b|\bgb200\b", "10.0"),
    (r"\bh100\b|\bh200\b", "9.0"),
    (r"\brtx\s*40[5-9]0|\bl4\b|\bl40", "8.9"),
    (r"\brtx\s*30[5-9]0|\ba10\b|\ba16\b|\ba40\b|rtx\s*a\d{4}", "8.6"),
    (r"\ba100\b|\ba30\b", "8.0"),
    (r"\brtx\s*20[678]0|titan\s*rtx|gtx\s*16[56]0|\bt4\b|\bt1000\b", "7.5"),
    (r"\bv100\b|titan\s*v", "7.0"),
    (r"gtx\s*10[5-8]0|titan\s*xp?\b|\bp100\b|\bp[456]000\b|\bp40\b", "6.1"),
    (r"gtx\s*9[578]0|gtx\s*titan\s*x|\bm[456]000\b|\bm40\b|\bm60\b", "5.2"),
    (r"gtx\s*[78][45]0", "5.0"),
    (r"gtx\s*7[78]0|gtx\s*titan\b|\bk80\b|\bk40\b", "3.5"),
]


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
    def cc(self) -> tuple:
        """Compute capability as (major, minor); (0, 0) if unknown."""
        m = re.match(r"(\d+)\.(\d+)", self.compute_cap.strip())
        if m:
            return (int(m.group(1)), int(m.group(2)))
        return (0, 0)

    @property
    def cc_major(self) -> int:
        return self.cc[0]

    @property
    def is_blackwell(self) -> bool:
        return self.family == "blackwell"

    @property
    def is_ada(self) -> bool:
        return self.family == "ada"

    @property
    def is_hopper(self) -> bool:
        return self.family == "hopper"

    @property
    def is_legacy(self) -> bool:
        """Pre-Volta architectures (Pascal, Maxwell, Kepler).

        These need older PyTorch wheels and CUDA <= 12.x toolkits; modern
        cu128+ wheels and CUDA 13 dropped their kernels.
        """
        return 0 < self.cc_major < 7

    @property
    def has_tensor_cores(self) -> bool:
        return self.cc >= (7, 0)

    @property
    def torch_cuda_arch(self) -> str:
        """Value for TORCH_CUDA_ARCH_LIST when compiling CUDA extensions."""
        major, minor = self.cc
        return f"{major}.{minor}" if major else ""

    @property
    def family(self) -> str:
        cc = self.cc
        if cc > (0, 0):
            for floor, name in _FAMILY_BY_CC:
                if cc >= floor:
                    return name
            return "unknown"
        # No compute capability reported — fall back to name matching
        lname = self.name.lower()
        if "blackwell" in lname or "b200" in lname:
            return "blackwell"
        for pattern, cap in _NAME_CC_HINTS:
            if re.search(pattern, lname):
                fake = GPU(self.index, self.name, self.memory_mb, cap, self.driver_version)
                return fake.family
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
        """Return recommended training defaults based on hardware.

        Keys beyond iterations/sh_degree are consumed by pipeline.train:
          data_device       "cuda" or "cpu" (original 3DGS --data_device)
          resolution_scale  image downscale divisor (1 = full res)
          cap_max           gsplat MCMC max gaussian count
          packed            gsplat --packed rasterization (less VRAM, slower)
          quality_extras    enable app-opt / bilateral grid / pose-opt (VRAM-hungry)
          recommended_method  training backend best suited to this GPU
        """
        gpu = self.primary_gpu
        defaults = {
            "iterations": 30000,
            "sh_degree": 3,
            "densify_grad_threshold": 0.0002,
            "data_device": "cuda",
            "resolution_scale": 1,
            "cap_max": 2_000_000,
            "packed": False,
            "quality_extras": True,
            "recommended_method": "gsplat",
        }
        if not gpu:
            return defaults

        vram = gpu.memory_gb
        if vram >= 40:  # DGX Spark, A100, H100
            defaults["sh_degree"] = 4
            defaults["densify_grad_threshold"] = 0.00015
            defaults["cap_max"] = 3_000_000
        elif vram >= 20:  # 4090, 3090
            defaults["densify_grad_threshold"] = 0.00015
        elif vram >= 12:  # 4070/3060 12GB class
            defaults["cap_max"] = 1_500_000
            defaults["densify_grad_threshold"] = 0.00015
        elif vram >= 8:  # GTX 1070/1080, RTX 2070/3070 class
            defaults["data_device"] = "cpu"
            defaults["cap_max"] = 1_000_000
            defaults["packed"] = True
            defaults["quality_extras"] = False
            defaults["densify_grad_threshold"] = 0.0003
        elif vram >= 6:  # GTX 1060 6GB, RTX 2060 class
            defaults["sh_degree"] = 2
            defaults["iterations"] = 20000
            defaults["data_device"] = "cpu"
            defaults["resolution_scale"] = 2
            defaults["cap_max"] = 600_000
            defaults["packed"] = True
            defaults["quality_extras"] = False
            defaults["densify_grad_threshold"] = 0.0004
        else:  # 4GB and below
            defaults["sh_degree"] = 2
            defaults["iterations"] = 15000
            defaults["data_device"] = "cpu"
            defaults["resolution_scale"] = 4
            defaults["cap_max"] = 350_000
            defaults["packed"] = True
            defaults["quality_extras"] = False
            defaults["densify_grad_threshold"] = 0.0004

        # Legacy architectures: the original 3DGS rasterizer is proven on
        # Pascal; gsplat needs a torch build that still ships these kernels.
        if gpu.is_legacy:
            defaults["recommended_method"] = "original"

        return defaults

    def colmap_defaults(self) -> dict:
        """Return recommended COLMAP settings based on hardware."""
        gpu = self.primary_gpu
        defaults = {
            "use_gpu": True,
            "camera_model": "SIMPLE_RADIAL",
            "matcher": "exhaustive",
            "max_image_size": 3200,
            "max_num_features": 8192,
        }

        if not gpu:
            defaults["use_gpu"] = False
            return defaults

        if gpu.memory_gb >= 20:
            pass  # exhaustive at full size
        elif gpu.memory_gb >= 6:
            defaults["matcher"] = "sequential"
            defaults["max_image_size"] = 2400
        else:
            defaults["matcher"] = "sequential"
            defaults["max_image_size"] = 1600
            defaults["max_num_features"] = 4096

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
            cap = f"sm_{g.compute_cap}" if g.compute_cap else "sm_?"
            lines.append(f"  [{g.index}] {g.name}  {g.memory_gb:.0f}GB  {cap}  ({g.family})")
            if g.is_legacy:
                lines.append(
                    f"      legacy arch — needs PyTorch <= cu126 wheels & CUDA <= 12.x "
                    f"(see docs/legacy-gpu-setup.md)"
                )
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


def check_torch_compat(gpu: GPU, conda_env: str, timeout: int = 120) -> list[str]:
    """Best-effort check that the conda env's PyTorch can run on this GPU.

    Returns a list of human-readable warning strings (empty = looks fine).
    Catches the classic legacy-GPU failure — a torch build whose kernels
    don't cover the card's compute capability — before training starts,
    instead of a cryptic 'no kernel image is available' mid-run.
    """
    snippet = (
        "import json, torch; "
        "print(json.dumps({'torch': torch.__version__, "
        "'cuda': torch.version.cuda, "
        "'archs': torch.cuda.get_arch_list(), "
        "'available': torch.cuda.is_available()}))"
    )
    try:
        result = subprocess.run(
            ["conda", "run", "-n", conda_env, "python", "-c", snippet],
            capture_output=True, text=True, timeout=timeout,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []  # no conda / too slow — skip silently, training will surface errors

    if result.returncode != 0:
        return []  # env or torch missing; the trainer gives a clearer error

    try:
        info = json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return []

    warnings = []
    major, minor = gpu.cc
    if major == 0:
        return []  # unknown compute cap — nothing to validate against

    # A cubin sm_XY runs on a device with the same major version and
    # minor >= Y; a compute_XY PTX target JIT-compiles on any cc >= X.Y.
    supported = False
    for arch in info.get("archs", []):
        m = re.match(r"(sm|compute)_(\d)(\d)", arch)
        if not m:
            continue
        kind, a_major, a_minor = m.group(1), int(m.group(2)), int(m.group(3))
        if kind == "sm" and a_major == major and a_minor <= minor:
            supported = True
        elif kind == "compute" and (a_major, a_minor) <= (major, minor):
            supported = True

    if not supported:
        warnings.append(
            f"PyTorch {info.get('torch')} (CUDA {info.get('cuda')}) in env "
            f"'{conda_env}' has no kernels for {gpu.name} (sm_{major}{minor}). "
            f"Supported archs: {', '.join(info.get('archs', [])) or 'none'}."
        )
        if gpu.is_legacy:
            warnings.append(
                f"Legacy GPU fix: install a torch wheel built for CUDA <= 12.6 "
                f"(e.g. --index-url https://download.pytorch.org/whl/cu118) and "
                f"rebuild CUDA extensions with TORCH_CUDA_ARCH_LIST=\"{gpu.torch_cuda_arch}\". "
                f"See docs/legacy-gpu-setup.md."
            )
    elif not info.get("available"):
        warnings.append(
            f"torch.cuda.is_available() is False in env '{conda_env}' — "
            f"driver/toolkit mismatch? (driver {gpu.driver_version}, "
            f"torch built for CUDA {info.get('cuda')})"
        )
    return warnings


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

    # No toolkit installed — report the driver's max supported CUDA version
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        match = re.search(r"CUDA Version:\s*(\d+\.\d+)", result.stdout)
        if match:
            return f"{match.group(1)} (driver)"
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


def _query_nvidia_smi(fields: str) -> Optional[list[list[str]]]:
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True,
        )
    except FileNotFoundError:
        return None
    if result.returncode != 0:
        return None
    return [
        [p.strip() for p in line.split(",")]
        for line in result.stdout.strip().splitlines()
        if line.strip()
    ]


def _cc_from_name(name: str) -> str:
    lname = name.lower()
    for pattern, cap in _NAME_CC_HINTS:
        if re.search(pattern, lname):
            return cap
    return ""


def _detect_gpus() -> list[GPU]:
    """Detect all NVIDIA GPUs via nvidia-smi."""
    rows = _query_nvidia_smi("index,name,memory.total,compute_cap,driver_version")
    has_compute_cap = True
    if rows is None:
        # Drivers < 510 don't support the compute_cap query field and fail
        # the whole query — retry without it and infer the cap from the name.
        rows = _query_nvidia_smi("index,name,memory.total,driver_version")
        has_compute_cap = False
    if rows is None:
        return []

    gpus = []
    for parts in rows:
        try:
            if has_compute_cap and len(parts) >= 5:
                index, name, mem, cap, driver = parts[0], parts[1], parts[2], parts[3], parts[4]
            elif not has_compute_cap and len(parts) >= 4:
                index, name, mem, driver = parts[0], parts[1], parts[2], parts[3]
                cap = ""
            else:
                continue
            if not re.match(r"\d+\.\d+", cap):  # "[N/A]" or empty
                cap = _cc_from_name(name)
            gpus.append(GPU(
                index=int(index),
                name=name,
                memory_mb=int(float(mem)),
                compute_cap=cap,
                driver_version=driver,
            ))
        except ValueError:
            continue
    return gpus
