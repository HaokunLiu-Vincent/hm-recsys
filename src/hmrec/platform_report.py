"""Describe what this process runs on: machine, image, libraries, OpenMP, BLAS, threads.

Used by scripts/image_info.py (`make image-info`, docs/mlops.md §8). The report
is plain text, one "name: value" line per fact, so that the output of the
laptop, of the arm64 container and of the x86-64 image can be compared line by
line.

Three environments run this project (docs/mlops.md §8.2):

    laptop, native         macOS on Apple silicon. THE REFERENCE: every
                           bit-identical check of stages 1 and 2 was made here.
    container on laptop    Linux on arm64.
    Cloud Run, CI          Linux on x86-64.

The two Linux environments use another OpenMP runtime and another BLAS than the
laptop (hmrec.native_libraries), so their results are accepted by the tolerance
of docs/mlops.md §8.3 and are never called bit-identical to the laptop. The
report says which of the two cases this process is in.

torch and LightGBM never share a process (AGENTS.md §5), so there are two
reports, one per side. Each side's libraries are imported INSIDE the function
that reports them: importing them at the top of this module would load both
OpenMP runtimes into one process.
"""

import importlib.metadata
import os
import platform
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import psutil

from hmrec.git_state import IMAGE_COMMIT_VARIABLE, image_commit
from hmrec.memory import bytes_to_gb
from hmrec.native_libraries import (
    blas_libraries,
    debian_package_version,
    loaded_library_paths,
    openmp_libraries,
)

REFERENCE_SYSTEM = "Darwin"
REFERENCE_MACHINE = "arm64"
TORCH_SIDE = "torch"
LIGHTGBM_SIDE = "lightgbm"
SIDES = (TORCH_SIDE, LIGHTGBM_SIDE)
# cgroup v2 files of a container: its CPU quota and its memory limit.
CGROUP_CPU_MAX_FILE = Path("/sys/fs/cgroup/cpu.max")
CGROUP_MEMORY_MAX_FILE = Path("/sys/fs/cgroup/memory.max")
CGROUP_NO_LIMIT = "max"
# Thread settings that numeric libraries read from the environment.
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "POLARS_MAX_THREADS",
)
NOT_SET = "not set"
COMMON_PACKAGES = ("numpy", "scipy", "polars", "pyarrow")


def is_reference_platform(system: str, machine: str) -> bool:
    """True on the platform of the recorded results: macOS on Apple silicon."""
    return system == REFERENCE_SYSTEM and machine == REFERENCE_MACHINE


def parse_cgroup_cpu_max(text: str) -> float | None:
    """The CPU quota in a cgroup v2 `cpu.max` file, in CPUs.

    "600000 100000" -> 6.0 (600 ms of CPU time per 100 ms period); "max 100000" -> None.

    Raises:
        ValueError: If the text is not "<quota> <period>".
    """
    fields = text.split()
    if len(fields) != 2:
        raise ValueError(f"cpu.max must hold '<quota> <period>', got {text!r}")
    quota, period = fields
    if quota == CGROUP_NO_LIMIT:
        return None
    return int(quota) / int(period)


def parse_cgroup_memory_max(text: str) -> int | None:
    """The memory limit in a cgroup v2 `memory.max` file, in bytes; None for "max"."""
    value = text.strip()
    if value == CGROUP_NO_LIMIT:
        return None
    return int(value)


@dataclass(frozen=True)
class MachineInfo:
    """The machine as this process sees it.

    Attributes:
        system: "Darwin" (macOS) or "Linux".
        machine: CPU architecture: "arm64" (macOS), "aarch64" or "x86_64" (Linux).
        os_release: Kernel release.
        python_version: e.g. "3.12.16".
        logical_cpus: CPUs of the machine or virtual machine (os.cpu_count).
        usable_cpus: CPUs this process may run on (its affinity mask); equal to
            logical_cpus on macOS, which has no affinity mask.
        cpu_quota: CPU quota of the container in CPUs (`docker run --cpus`),
            None if there is none or this is not a container.
        memory_total_bytes: Physical memory of the machine or virtual machine.
        memory_limit_bytes: Memory limit of the container (`docker run --memory`),
            None if there is none or this is not a container.
    """

    system: str
    machine: str
    os_release: str
    python_version: str
    logical_cpus: int
    usable_cpus: int
    cpu_quota: float | None
    memory_total_bytes: int
    memory_limit_bytes: int | None


def read_usable_cpus(logical_cpus: int) -> int:
    """CPUs this process may run on: its affinity mask on Linux, all CPUs on macOS."""
    if sys.platform == "linux":
        return len(os.sched_getaffinity(0))
    return logical_cpus


def read_machine_info() -> MachineInfo:
    """Read MachineInfo for this process."""
    logical_cpus = os.cpu_count() or 0
    cpu_quota = None
    if CGROUP_CPU_MAX_FILE.is_file():
        cpu_quota = parse_cgroup_cpu_max(CGROUP_CPU_MAX_FILE.read_text())
    memory_limit = None
    if CGROUP_MEMORY_MAX_FILE.is_file():
        memory_limit = parse_cgroup_memory_max(CGROUP_MEMORY_MAX_FILE.read_text())
    return MachineInfo(
        system=platform.system(),
        machine=platform.machine(),
        os_release=platform.release(),
        python_version=platform.python_version(),
        logical_cpus=logical_cpus,
        usable_cpus=read_usable_cpus(logical_cpus),
        cpu_quota=cpu_quota,
        memory_total_bytes=psutil.virtual_memory().total,
        memory_limit_bytes=memory_limit,
    )


def machine_lines(info: MachineInfo) -> list[str]:
    """The report lines about the machine, including the reference-platform statement."""
    if is_reference_platform(info.system, info.machine):
        reference = "yes: bit-identical checks against the recorded results are valid here"
    else:
        reference = (
            "NO: results here are NOT bit-identical to the laptop; they are accepted by "
            "the tolerance of docs/mlops.md 8.3"
        )
    quota = "none" if info.cpu_quota is None else f"{info.cpu_quota:g}"
    limit = "none"
    if info.memory_limit_bytes is not None:
        limit = f"{bytes_to_gb(info.memory_limit_bytes):.2f} GiB"
    return [
        f"platform: {info.system} {info.machine} (kernel {info.os_release})",
        f"reference platform: {reference}",
        f"python: {info.python_version}",
        f"cpus: {info.logical_cpus} logical, {info.usable_cpus} usable by this process, "
        f"container quota {quota}",
        f"memory: {bytes_to_gb(info.memory_total_bytes):.2f} GiB total, container limit {limit}",
    ]


def image_lines(environment: Mapping[str, str]) -> list[str]:
    """The report line about the commit of the container image, if this is one."""
    commit = image_commit(environment)
    if commit is None:
        return [f"image commit: not in a container image ({IMAGE_COMMIT_VARIABLE} is not set)"]
    return [f"image commit: {commit}"]


def thread_setting_lines(environment: Mapping[str, str]) -> list[str]:
    """One line per thread-count environment variable: its value or "not set"."""
    return [f"env {name}: {environment.get(name, NOT_SET)}" for name in THREAD_VARIABLES]


def package_version_lines(names: Sequence[str]) -> list[str]:
    """One line per installed package: "package <name>: <version>"."""
    return [f"package {name}: {importlib.metadata.version(name)}" for name in names]


def library_lines(paths: Sequence[str]) -> list[str]:
    """The OpenMP runtimes and BLAS libraries among the loaded libraries `paths`.

    Each OpenMP runtime is listed with the version of the system package that
    owns it ("unknown" for a copy inside a wheel, whose version is fixed by the
    wheel's hash in uv.lock, and on macOS, where hmrec.ranker.runtime reads the
    Homebrew version from the path instead).
    """
    lines = []
    openmp = openmp_libraries(paths)
    lines.append(f"openmp runtimes loaded: {len(openmp)}")
    for path in openmp:
        lines.append(f"openmp runtime: {path} (system package: {debian_package_version(path)})")
    blas = blas_libraries(paths)
    lines.append(f"blas libraries loaded: {len(blas)}")
    lines.extend(f"blas library: {path}" for path in blas)
    return lines


def build_blas_line(package: str, build_config: Mapping[str, object] | None) -> str:
    """The BLAS a numpy or scipy wheel was built against, from its show_config dictionary.

    Args:
        package: "numpy" or "scipy".
        build_config: `<package>.show_config(mode="dicts")`.

    Raises:
        ValueError: If the dictionary is missing or does not have the expected form.
    """
    if build_config is None:
        raise ValueError(f"{package}.show_config(mode='dicts') returned nothing")
    dependencies = build_config["Build Dependencies"]
    if not isinstance(dependencies, Mapping):
        raise ValueError(f"{package}.show_config: 'Build Dependencies' is not a mapping")
    blas = dependencies["blas"]
    if not isinstance(blas, Mapping):
        raise ValueError(f"{package}.show_config: 'blas' is not a mapping")
    return f"{package} blas: {blas['name']} {blas['version']}"


def numpy_scipy_blas_lines() -> list[str]:
    """The BLAS of numpy and of scipy (Accelerate on the laptop, OpenBLAS in the Linux wheels)."""
    import numpy as np
    import scipy
    import scipy.linalg  # noqa: F401  (loads scipy's BLAS, so that library_lines can list it)

    return [
        build_blas_line("numpy", np.show_config(mode="dicts")),
        build_blas_line("scipy", scipy.show_config(mode="dicts")),
    ]


def torch_side_lines() -> list[str]:
    """Report the torch side: torch, implicit (R5), their threads and native libraries.

    Imports torch and implicit, so it must not run in a LightGBM process.
    """
    import implicit.cpu.als  # noqa: F401  (loads the compiled ALS code and its OpenMP runtime)
    import torch

    lines = package_version_lines(("torch", "implicit", *COMMON_PACKAGES))
    lines.append(f"torch intra-op threads: {torch.get_num_threads()}")
    lines.append(f"torch inter-op threads: {torch.get_num_interop_threads()}")
    # torch.__config__ lists how the wheel was built: which BLAS / LAPACK (MKL is linked
    # statically into the x86-64 wheel, so it is not a library file of its own) and OpenMP.
    wanted = ("BLAS_INFO", "LAPACK_INFO", "USE_OPENMP", "USE_MKL")
    for setting in torch.__config__.show().replace("\n", " ").split(","):
        name = setting.strip().partition("=")[0]
        if name in wanted:
            lines.append(f"torch build: {setting.strip()}")
    lines.append(f"torch mps available: {torch.backends.mps.is_available()}")
    lines.extend(numpy_scipy_blas_lines())
    lines.extend(library_lines(loaded_library_paths()))
    return lines


def lightgbm_side_lines() -> list[str]:
    """Report the LightGBM side: LightGBM and the one OpenMP runtime it loaded.

    Imports lightgbm, so it must not run in a torch process.
    """
    from hmrec.ranker.runtime import read_lightgbm_runtime

    runtime = read_lightgbm_runtime()
    lines = package_version_lines(("lightgbm", *COMMON_PACKAGES))
    lines.append(f"lightgbm openmp runtime: {runtime.libomp_path}")
    lines.append(f"lightgbm openmp version: {runtime.libomp_version}")
    lines.extend(numpy_scipy_blas_lines())
    lines.extend(library_lines(loaded_library_paths()))
    return lines


def report_lines(side: str, environment: Mapping[str, str]) -> list[str]:
    """The whole report of one side.

    Args:
        side: "torch" or "lightgbm".
        environment: The environment variables of the process (os.environ).

    Raises:
        ValueError: If `side` is not one of the two.
    """
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, got {side!r}")
    lines = [f"side: {side}"]
    lines.extend(image_lines(environment))
    lines.extend(machine_lines(read_machine_info()))
    lines.extend(thread_setting_lines(environment))
    side_lines = torch_side_lines() if side == TORCH_SIDE else lightgbm_side_lines()
    lines.extend(side_lines)
    return lines
