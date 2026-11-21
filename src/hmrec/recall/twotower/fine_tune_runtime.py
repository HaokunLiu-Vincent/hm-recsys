"""Runtime settings that can surround an R6 fine-tune (M10 part 3; decisions.md 2026-11-22).

Why: stage 1 trains R6 on the CPU because training there is bit-reproducible on
the laptop. In the arm64 Linux container it is not: two fine-tunes of the same
week from the same model give different weights (results/m10_calibration/
repeatability.csv). This module holds the settings that were tried to make a
fine-tune repeatable on Linux. Each one is a `with` block; when the block ends,
what it changed is set back.

    none                         nothing is changed (the settings of stage 1)
    torch_blas_1                 the BLAS library inside the torch wheel runs on
                                 one thread; torch's OpenMP threads stay as they are
    deterministic                torch.use_deterministic_algorithms(True)
    deterministic+torch_blas_1   both of the above
    one_thread                   torch runs on one intra-op thread

    implemented                  what a real fine-tune runs under on this platform
                                 (r6_fine_tune_runtime, below)

The settings change HOW torch computes (threads, choice of algorithm), never
what is computed: no hyperparameter, seed or data is touched.

What a real fine-tune uses (r6_fine_tune_runtime; decision of 2026-11-22):

    Linux    torch.use_deterministic_algorithms(True) for the duration of the
             fine-tune. It was the first candidate that made two fine-tunes in
             fresh processes bit-identical in the arm64 container
             (results/m10_calibration/r6_repeatability.csv).
    macOS    NOTHING. The laptop is the reference of stage 1 and its behaviour
             must not change, so the block runs exactly as it always did.

The scope is the fine-tune only (hmrec.recall.twotower.chain.fine_tune_on_week).
Prediction, base training and every other use of torch are not inside the block.

This module is on the torch side.
"""

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import torch
from threadpoolctl import ThreadpoolController, threadpool_info

SETTING_NONE = "none"
SETTING_TORCH_BLAS = "torch_blas_1"
SETTING_DETERMINISTIC = "deterministic"
SETTING_DETERMINISTIC_AND_BLAS = "deterministic+torch_blas_1"
SETTING_ONE_THREAD = "one_thread"
SETTING_IMPLEMENTED = "implemented"
SETTINGS = (
    SETTING_NONE,
    SETTING_TORCH_BLAS,
    SETTING_DETERMINISTIC,
    SETTING_DETERMINISTIC_AND_BLAS,
    SETTING_ONE_THREAD,
    SETTING_IMPLEMENTED,
)
LINUX = "linux"
BLAS_API = "blas"
TORCH_PACKAGE_FOLDER = "torch"


class SettingNotApplicable(RuntimeError):
    """A setting cannot be applied in this process (for example: torch has no BLAS file here)."""


def torch_blas_paths(library_paths: list[str]) -> list[str]:
    """The BLAS libraries that lie inside the torch package, among `library_paths`.

    Args:
        library_paths: File paths of BLAS libraries loaded in a process.

    Returns:
        The paths with a folder named "torch" among their parents
        (".../site-packages/torch/lib/libopenblas.so.0"), in the given order.
    """
    return [path for path in library_paths if TORCH_PACKAGE_FOLDER in Path(path).parent.parts]


@contextmanager
def torch_blas_one_thread() -> Iterator[list[str]]:
    """Run a block with torch's own BLAS on one thread and torch's OpenMP threads unchanged.

    On Linux arm64 the BLAS inside the torch wheel is an OpenBLAS built on
    OpenMP: setting its thread count also lowers the OpenMP thread count of the
    process (measured 2026-11-22: torch.get_num_threads() went from 6 to 1). So
    the OpenMP thread count is set back to its value right after the limit.

    Yields:
        The file paths of the limited libraries.

    Raises:
        SettingNotApplicable: If no BLAS file of torch is loaded (macOS uses
            Accelerate; the x86-64 wheel has MKL linked into torch itself).
    """
    controller = ThreadpoolController()
    blas_paths = [
        library.filepath for library in controller.lib_controllers if library.user_api == BLAS_API
    ]
    limited = torch_blas_paths(blas_paths)
    if not limited:
        raise SettingNotApplicable(
            f"no BLAS library of torch is loaded (BLAS libraries found: {blas_paths})"
        )
    openmp_threads = torch.get_num_threads()
    try:
        with controller.select(filepath=limited).limit(limits=1):
            torch.set_num_threads(openmp_threads)
            yield limited
    finally:
        torch.set_num_threads(openmp_threads)


@contextmanager
def deterministic_algorithms() -> Iterator[None]:
    """Run a block with torch.use_deterministic_algorithms(True); the earlier value is set back.

    Inside the block torch uses a deterministic algorithm where it has one and
    raises a RuntimeError at an operation that has none.
    """
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous)


@contextmanager
def one_intra_op_thread() -> Iterator[None]:
    """Run a block with one intra-op thread of torch; the earlier thread count is set back."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@contextmanager
def r6_fine_tune_runtime(platform: str) -> Iterator[str]:
    """The runtime a real R6 fine-tune runs under: deterministic algorithms on Linux only.

    Args:
        platform: sys.platform of this process ("linux", or "darwin" = macOS).

    Yields:
        The name of the setting in force inside the block: "deterministic" on
        Linux, "none" everywhere else (nothing is changed there).
    """
    if platform != LINUX:
        # The laptop: nothing is touched (module docstring).
        yield SETTING_NONE
        return
    with deterministic_algorithms():
        yield SETTING_DETERMINISTIC


@contextmanager
def fine_tune_setting(name: str) -> Iterator[None]:
    """Run a block under one of the SETTINGS (see the module docstring).

    Raises:
        ValueError: On an unknown name.
        SettingNotApplicable: If the setting cannot be applied in this process.
    """
    if name not in SETTINGS:
        raise ValueError(f"setting must be one of {SETTINGS}, got {name!r}")
    if name == SETTING_NONE:
        yield
    elif name == SETTING_TORCH_BLAS:
        with torch_blas_one_thread():
            yield
    elif name == SETTING_DETERMINISTIC:
        with deterministic_algorithms():
            yield
    elif name == SETTING_DETERMINISTIC_AND_BLAS:
        with deterministic_algorithms(), torch_blas_one_thread():
            yield
    elif name == SETTING_IMPLEMENTED:
        with r6_fine_tune_runtime(sys.platform):
            yield
    else:
        with one_intra_op_thread():
            yield


def thread_report() -> str:
    """The thread counts in force right now, as one line for a record.

    Example: "torch intra-op 6; libopenblas (blas) 1; libgomp (openmp) 6".
    """
    parts = [f"torch intra-op {torch.get_num_threads()}"]
    for pool in threadpool_info():
        name = Path(pool["filepath"]).name.split(".")[0]
        parts.append(f"{name} ({pool['user_api']}) {pool['num_threads']}")
    deterministic = torch.are_deterministic_algorithms_enabled()
    parts.append(f"deterministic algorithms {deterministic}")
    return "; ".join(parts)
