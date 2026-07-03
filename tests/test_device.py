"""Unit tests for pipeline.device — pure logic only (no nvidia-smi needed).

Run:  python tests/test_device.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.device import GPU, DeviceProfile, _cc_from_name


def _gpu(name, mem_gb, cap):
    return GPU(index=0, name=name, memory_mb=int(mem_gb * 1024),
               compute_cap=cap, driver_version="test")


def _profile(gpu):
    return DeviceProfile(hostname="t", arch="x86_64", os_name="test", gpus=[gpu])


def test_family_mapping():
    cases = [
        ("NVIDIA GeForce GTX 780", "3.5", "kepler"),
        ("NVIDIA GeForce GTX 970", "5.2", "maxwell"),
        ("NVIDIA GeForce GTX 1070", "6.1", "pascal"),
        ("Tesla V100", "7.0", "volta"),
        ("NVIDIA GeForce RTX 2070", "7.5", "turing"),
        ("NVIDIA GeForce GTX 1660", "7.5", "turing"),
        ("NVIDIA GeForce RTX 3090", "8.6", "ampere"),
        ("NVIDIA GeForce RTX 4090", "8.9", "ada"),
        ("NVIDIA H100", "9.0", "hopper"),
        ("NVIDIA GB10", "10.0", "blackwell"),
        ("NVIDIA GeForce RTX 5090", "12.0", "blackwell"),
    ]
    for name, cap, family in cases:
        got = _gpu(name, 8, cap).family
        assert got == family, f"{name} sm_{cap}: expected {family}, got {got}"


def test_family_from_name_when_cap_missing():
    # Old drivers don't report compute_cap — family falls back to the name
    cases = [
        ("NVIDIA GeForce GTX 1070", "pascal"),
        ("NVIDIA GeForce GTX 980 Ti", "maxwell"),
        ("NVIDIA GeForce RTX 2080 Ti", "turing"),
        ("NVIDIA GeForce RTX 3060", "ampere"),
    ]
    for name, family in cases:
        got = _gpu(name, 8, "").family
        assert got == family, f"{name} (no cap): expected {family}, got {got}"


def test_cc_from_name():
    assert _cc_from_name("NVIDIA GeForce GTX 1070") == "6.1"
    assert _cc_from_name("NVIDIA GeForce GTX 1060 6GB") == "6.1"
    assert _cc_from_name("NVIDIA TITAN Xp") == "6.1"
    assert _cc_from_name("NVIDIA GeForce RTX 4090") == "8.9"
    assert _cc_from_name("Some Unknown GPU") == ""


def test_legacy_and_tensor_cores():
    gtx1070 = _gpu("NVIDIA GeForce GTX 1070", 8, "6.1")
    assert gtx1070.is_legacy
    assert not gtx1070.has_tensor_cores
    assert gtx1070.torch_cuda_arch == "6.1"

    rtx2070 = _gpu("NVIDIA GeForce RTX 2070", 8, "7.5")
    assert not rtx2070.is_legacy
    assert rtx2070.has_tensor_cores

    unknown = _gpu("Mystery GPU", 8, "")
    assert not unknown.is_legacy  # unknown cc must not be flagged legacy


def test_training_defaults_gtx1070():
    td = _profile(_gpu("NVIDIA GeForce GTX 1070", 8, "6.1")).training_defaults()
    assert td["data_device"] == "cpu"
    assert td["cap_max"] == 1_000_000
    assert td["packed"] is True
    assert td["quality_extras"] is False
    assert td["recommended_method"] == "original"
    assert td["sh_degree"] == 3  # 8GB can still afford SH 3
    assert td["resolution_scale"] == 1


def test_training_defaults_gtx1060():
    td = _profile(_gpu("NVIDIA GeForce GTX 1060 6GB", 6, "6.1")).training_defaults()
    assert td["sh_degree"] == 2
    assert td["iterations"] == 20000
    assert td["resolution_scale"] == 2
    assert td["cap_max"] == 600_000


def test_training_defaults_highend_unchanged():
    td = _profile(_gpu("NVIDIA GeForce RTX 4090", 24, "8.9")).training_defaults()
    assert td["iterations"] == 30000
    assert td["sh_degree"] == 3
    assert td["data_device"] == "cuda"
    assert td["cap_max"] == 2_000_000
    assert td["quality_extras"] is True
    assert td["recommended_method"] == "gsplat"

    td = _profile(_gpu("NVIDIA H100", 80, "9.0")).training_defaults()
    assert td["sh_degree"] == 4


def test_colmap_defaults_tiers():
    cd = _profile(_gpu("NVIDIA GeForce RTX 4090", 24, "8.9")).colmap_defaults()
    assert cd["matcher"] == "exhaustive"
    assert cd["max_image_size"] == 3200

    cd = _profile(_gpu("NVIDIA GeForce GTX 1070", 8, "6.1")).colmap_defaults()
    assert cd["matcher"] == "sequential"
    assert cd["max_image_size"] == 2400

    cd = _profile(_gpu("NVIDIA GeForce GTX 1050 Ti", 4, "6.1")).colmap_defaults()
    assert cd["max_image_size"] == 1600
    assert cd["max_num_features"] == 4096

    cd = DeviceProfile(hostname="t", arch="x86_64", os_name="test").colmap_defaults()
    assert cd["use_gpu"] is False


def test_summary_flags_legacy():
    s = _profile(_gpu("NVIDIA GeForce GTX 1070", 8, "6.1")).summary()
    assert "pascal" in s
    assert "legacy" in s


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS {name}")
            except AssertionError as e:
                failures += 1
                print(f"  FAIL {name}: {e}")
    print("All tests passed" if failures == 0 else f"{failures} test(s) failed")
    sys.exit(1 if failures else 0)
