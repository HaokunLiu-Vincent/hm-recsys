"""Tests for hmrec.platform_report: the machine, image and torch-side report.

The LightGBM-side report is tested in test_ranker_platform_report.py, which runs in
the LightGBM-side test process (tests/conftest.py).
"""

import os
import sys

import pytest

from hmrec.git_state import IMAGE_COMMIT_VARIABLE
from hmrec.memory import BYTES_PER_GB
from hmrec.platform_report import (
    MachineInfo,
    build_blas_line,
    image_lines,
    is_reference_platform,
    library_lines,
    machine_lines,
    openmp_of_import,
    openmp_of_import_line,
    package_version_lines,
    parse_cgroup_cpu_max,
    parse_cgroup_memory_max,
    read_machine_info,
    report_lines,
    thread_setting_lines,
)

IMAGE_COMMIT = "0123456789abcdef0123456789abcdef01234567"
LAPTOP = MachineInfo("Darwin", "arm64", "27.2.0", "3.12.16", 10, 10, None, 32 * BYTES_PER_GB, None)
CONTAINER = MachineInfo(
    "Linux", "aarch64", "6.12.0", "3.12.16", 6, 4, 4.0, 16 * BYTES_PER_GB, 12 * BYTES_PER_GB
)


def test_only_macos_on_apple_silicon_is_the_reference_platform() -> None:
    assert is_reference_platform("Darwin", "arm64")
    assert not is_reference_platform("Linux", "aarch64")  # the arm64 container on the laptop
    assert not is_reference_platform("Linux", "x86_64")  # CI and Cloud Run
    assert not is_reference_platform("Darwin", "x86_64")


def test_cgroup_files_are_parsed() -> None:
    assert parse_cgroup_cpu_max("600000 100000\n") == 6.0
    assert parse_cgroup_cpu_max("150000 100000") == 1.5
    assert parse_cgroup_cpu_max("max 100000\n") is None
    with pytest.raises(ValueError, match="'<quota> <period>'"):
        parse_cgroup_cpu_max("600000")
    assert parse_cgroup_memory_max("12884901888\n") == 12 * BYTES_PER_GB
    assert parse_cgroup_memory_max("max\n") is None


def test_the_laptop_is_reported_as_the_reference() -> None:
    lines = machine_lines(LAPTOP)
    assert lines == [
        "platform: Darwin arm64 (kernel 27.2.0)",
        "reference platform: yes: bit-identical checks against the recorded results are valid here",
        "python: 3.12.16",
        "cpus: 10 logical, 10 usable by this process, container quota none",
        "memory: 32.00 GiB total, container limit none",
    ]


def test_a_container_never_claims_bit_identity_with_the_laptop() -> None:
    lines = machine_lines(CONTAINER)
    assert lines[0] == "platform: Linux aarch64 (kernel 6.12.0)"
    assert lines[1].startswith("reference platform: NO: results here are NOT bit-identical")
    assert "tolerance of docs/mlops.md 8.3" in lines[1]
    assert lines[3] == "cpus: 6 logical, 4 usable by this process, container quota 4"
    assert lines[4] == "memory: 16.00 GiB total, container limit 12.00 GiB"


def test_the_image_commit_line() -> None:
    assert image_lines({IMAGE_COMMIT_VARIABLE: IMAGE_COMMIT}) == [f"image commit: {IMAGE_COMMIT}"]
    assert image_lines({}) == [
        "image commit: not in a container image (HMREC_IMAGE_GIT_COMMIT is not set)"
    ]
    with pytest.raises(ValueError, match="not a full 40-character commit hash"):
        image_lines({IMAGE_COMMIT_VARIABLE: "main"})


def test_thread_settings_show_what_is_set_and_what_is_not() -> None:
    lines = thread_setting_lines({"OMP_NUM_THREADS": "6", "UNRELATED": "1"})
    assert lines[0] == "env OMP_NUM_THREADS: 6"
    assert "env OPENBLAS_NUM_THREADS: not set" in lines
    assert "env POLARS_MAX_THREADS: not set" in lines
    assert not any("UNRELATED" in line for line in lines)


def test_library_lines_count_and_list(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "hmrec.platform_report.debian_package_version",
        lambda path: "libgomp1 14.2.0-19" if path.startswith("/usr/lib") else "unknown",
    )
    lines = library_lines(
        [
            "/usr/lib/aarch64-linux-gnu/libgomp.so.1.0.0",
            "/app/.venv/torch/lib/libgomp-a34b3233.so.1",
            "/app/.venv/numpy.libs/libscipy_openblas64_-6bb31eeb.so",
            "/usr/lib/aarch64-linux-gnu/libc.so.6",
        ]
    )
    assert lines == [
        "openmp runtimes loaded: 2",
        "openmp runtime: /app/.venv/torch/lib/libgomp-a34b3233.so.1 (system package: unknown)",
        "openmp runtime: /usr/lib/aarch64-linux-gnu/libgomp.so.1.0.0 "
        "(system package: libgomp1 14.2.0-19)",
        "blas libraries loaded: 1",
        "blas library: /app/.venv/numpy.libs/libscipy_openblas64_-6bb31eeb.so",
    ]
    assert library_lines([]) == ["openmp runtimes loaded: 0", "blas libraries loaded: 0"]


def test_the_build_blas_line() -> None:
    config = {"Build Dependencies": {"blas": {"name": "scipy-openblas", "version": "0.3.30"}}}
    assert build_blas_line("numpy", config) == "numpy blas: scipy-openblas 0.3.30"
    with pytest.raises(ValueError, match="'blas' is not a mapping"):
        build_blas_line("numpy", {"Build Dependencies": {"blas": "openblas"}})
    with pytest.raises(ValueError, match="'Build Dependencies' is not a mapping"):
        build_blas_line("scipy", {"Build Dependencies": []})
    with pytest.raises(ValueError, match="returned nothing"):
        build_blas_line("scipy", None)


def test_package_versions_are_the_installed_ones() -> None:
    import numpy as np

    assert package_version_lines(("numpy",)) == [f"package numpy: {np.__version__}"]


def test_the_machine_of_this_process() -> None:
    info = read_machine_info()
    assert info.python_version.startswith("3.12.")
    assert 1 <= info.usable_cpus <= info.logical_cpus
    assert info.memory_total_bytes > 0
    if sys.platform == "darwin":
        # The laptop: no cgroup files, and it is the reference platform.
        assert (info.system, info.cpu_quota, info.memory_limit_bytes) == ("Darwin", None, None)
        assert info.usable_cpus == info.logical_cpus
        assert is_reference_platform(info.system, info.machine) == (info.machine == "arm64")
    else:
        # A Linux container (arm64 on the laptop, x86-64 in CI): never the reference.
        assert info.system == "Linux" and info.machine in ("aarch64", "x86_64")
        assert not is_reference_platform(info.system, info.machine)
        assert info.memory_limit_bytes is None or info.memory_limit_bytes > 0


def test_the_torch_side_report_of_this_process() -> None:
    import torch

    lines = report_lines("torch", os.environ)
    assert lines[0] == "side: torch"
    assert f"package torch: {torch.__version__}" in lines
    assert any(line.startswith("package implicit: ") for line in lines)
    assert any(line.startswith("torch intra-op threads: ") for line in lines)
    assert any(line.startswith("numpy blas: ") for line in lines)
    openmp = [line for line in lines if line.startswith("openmp runtime: ")]
    reference = [line for line in lines if line.startswith("reference platform: ")]
    implicit_alone = [line for line in lines if line.startswith("openmp of `import implicit")]
    torch_alone = [line for line in lines if line.startswith("openmp of `import torch`")]
    assert len(implicit_alone) == 1 and len(torch_alone) == 1
    if sys.platform == "darwin":
        # The laptop: torch brings LLVM's libomp; the implicit wheel has no OpenMP at all,
        # so R5 (WALS) is fitted on one thread here. numpy and scipy use Accelerate.
        assert len(openmp) == 1 and "/torch/lib/libomp.dylib" in openmp[0]
        assert implicit_alone[0].endswith("none (this build uses no OpenMP: one thread)")
        assert "/torch/lib/libomp.dylib" in torch_alone[0]
        assert "numpy blas: accelerate unknown" in lines
        assert reference[0].startswith("reference platform: yes")
        assert "torch mps available: True" in lines
    else:
        # A Linux container: both use GNU libgomp, so the implicit build IS multi-threaded
        # here (R5 runs on num_threads threads, unlike on the laptop). No MPS, no Accelerate.
        assert len(openmp) >= 1
        assert all("libgomp" in line for line in openmp)
        assert "libgomp" in implicit_alone[0] and "libgomp" in torch_alone[0]
        assert reference[0].startswith("reference platform: NO")
        assert "torch mps available: False" in lines
        assert not any("accelerate" in line.lower() for line in lines)


def test_openmp_of_a_module_that_uses_none() -> None:
    # The standard library's json module loads no OpenMP runtime on any platform.
    assert openmp_of_import("json") == []
    assert openmp_of_import_line("json") == (
        "openmp of `import json` alone: none (this build uses no OpenMP: one thread)"
    )


def test_an_unknown_side_is_refused() -> None:
    with pytest.raises(ValueError, match="side must be one of"):
        report_lines("both", {})
