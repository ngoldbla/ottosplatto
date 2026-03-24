"""GPU monitoring — real-time stats during training via nvidia-smi."""
import subprocess
from typing import Optional, Callable


def get_gpu_stats() -> Optional[dict]:
    """Query GPU stats via nvidia-smi. Returns None if unavailable."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total,memory.free,power.draw,power.limit,name",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return None

        parts = [p.strip() for p in result.stdout.strip().split(",")]
        if len(parts) < 8:
            return None

        temp_c = float(parts[0])
        temp_f = temp_c * 9 / 5 + 32

        return {
            "temp_c": temp_c,
            "temp_f": temp_f,
            "utilization": float(parts[1]),
            "vram_used_mb": float(parts[2]),
            "vram_total_mb": float(parts[3]),
            "vram_free_mb": float(parts[4]),
            "vram_used_gb": round(float(parts[2]) / 1024, 1),
            "vram_total_gb": round(float(parts[3]) / 1024, 1),
            "vram_pct": round(float(parts[2]) / float(parts[3]) * 100, 1),
            "power_w": float(parts[5]),
            "power_limit_w": float(parts[6]),
            "gpu_name": parts[7],
        }
    except Exception:
        return None


def format_gpu_stats(stats: dict) -> str:
    """Format GPU stats into a compact display string."""
    if not stats:
        return "GPU: unavailable"

    return (
        f"GPU {stats['temp_f']:.0f}°F ({stats['temp_c']:.0f}°C) | "
        f"Util {stats['utilization']:.0f}% | "
        f"VRAM {stats['vram_used_gb']}GB / {stats['vram_total_gb']}GB ({stats['vram_pct']}%) | "
        f"Power {stats['power_w']:.0f}W / {stats['power_limit_w']:.0f}W"
    )


def log_gpu_stats(on_output: Optional[Callable[[str], None]] = None) -> Optional[dict]:
    """Query and log GPU stats. Returns the stats dict."""
    stats = get_gpu_stats()
    if stats and on_output:
        on_output(f"  🖥️ {format_gpu_stats(stats)}")
    return stats
