"""Limit the BLAS that `implicit` calls to one thread during an R5 fit, on Linux only.

Why (decision of 2026-11-22, docs/decisions.md): on Linux `implicit` (R5, WALS)
runs its own OpenMP threads (num_threads of configs/base.yaml = 6), and inside
each of them it calls BLAS / LAPACK through scipy. The OpenBLAS inside the
Linux wheel of scipy has a thread pool of its own. Six OpenMP threads that each
start a six-thread BLAS call fight for the same six cores; `implicit` itself
warns about this and asks for a single-threaded BLAS.

What is limited: ONLY the BLAS library of scipy (the one `implicit` links to,
through scipy.linalg.cython_blas), and only inside the `with` block. Not
limited on purpose:

    numpy's BLAS   another copy of OpenBLAS (numpy.libs); the R5 code does not
                   call it during the fit.
    torch's BLAS   on Linux arm64 it is an OpenBLAS built on OpenMP. Limiting it
                   calls omp_set_num_threads(1) and so changes torch's thread
                   count (measured in the arm64 image on 2026-11-22: 6 -> 1).
                   torch's threads must not change, so it is left alone.

On macOS (the laptop, the reference) NOTHING is done: the block runs as it
always did. The macOS wheel of `implicit` has no OpenMP and scipy uses Apple's
Accelerate there, so there is nothing to limit, and the laptop's results must
stay exactly as they were.

`implicit` still prints its warning once ("OpenBLAS is configured to use 6
threads ..."): it looks at EVERY BLAS of the process, and numpy's and torch's
are not limited, for the reasons above. The warning does not mean the limit is
missing; the log line "BLAS limited to 1 thread during the fit" names the
library that is limited.

threadpoolctl is used for the limit. It is a direct dependency of the project
(pyproject.toml, since 2026-11-22), at the version `implicit` already pinned.

This module is at the bottom layer. It imports no torch, lightgbm or polars.
"""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

from threadpoolctl import ThreadpoolController

LINUX = "linux"
BLAS_API = "blas"
# A Linux wheel keeps its bundled libraries in "<package>.libs" next to the package.
SCIPY_LIBRARY_FOLDER = "scipy.libs"


def scipy_blas_paths(library_paths: Sequence[str]) -> list[str]:
    """The BLAS libraries that belong to the scipy wheel, among `library_paths`.

    Args:
        library_paths: File paths of BLAS libraries loaded in a process.

    Returns:
        The paths whose folder is "scipy.libs", in the given order.
    """
    return [path for path in library_paths if Path(path).parent.name == SCIPY_LIBRARY_FOLDER]


@contextmanager
def single_threaded_scipy_blas(platform: str) -> Iterator[list[str]]:
    """Run a block with scipy's BLAS limited to one thread (Linux), or unchanged (macOS).

    scipy must already be imported, so that its BLAS is loaded. When the block
    ends, the thread count the library had before is set again.

    Args:
        platform: sys.platform of this process ("linux", or "darwin" = macOS).

    Yields:
        The file paths of the libraries that are limited inside the block
        (empty on macOS).

    Raises:
        RuntimeError: On Linux, if no BLAS of scipy is loaded. Then the limit
            would silently do nothing, so the run stops instead.
    """
    if platform != LINUX:
        # The laptop: nothing is changed (module docstring).
        yield []
        return
    controller = ThreadpoolController()
    blas_paths = [
        library.filepath for library in controller.lib_controllers if library.user_api == BLAS_API
    ]
    limited = scipy_blas_paths(blas_paths)
    if not limited:
        raise RuntimeError(
            f"no BLAS library of scipy is loaded (BLAS libraries found: {blas_paths}); "
            "the one-thread limit for the R5 fit cannot be set"
        )
    # select() keeps the libraries whose file path is in the list; limit() sets their
    # thread count to 1 and restores the earlier value when the block ends.
    with controller.select(filepath=limited).limit(limits=1):
        yield limited
