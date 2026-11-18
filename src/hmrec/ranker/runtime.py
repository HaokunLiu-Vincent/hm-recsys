"""Which LightGBM and which OpenMP runtime (libomp) this process really uses.

Why this is recorded with every M7 run (docs/decisions.md 2026-11-18): LightGBM
does not bring its own OpenMP runtime. On this machine it loads Homebrew's libomp
(`brew install libomp`), a SYSTEM library that is not in uv.lock, so `uv sync`
neither installs nor pins it. Its version can change with a `brew upgrade`
without any change in the repository. Each M7 result row therefore names the
LightGBM version and the path and version of the libomp that was loaded; after a
libomp upgrade the "two trainings give the same model" check must be repeated.

How the loaded library is found: the operating system is asked for the files
this process has loaded (macOS: the dyld image list; Linux: /proc/self/maps),
so the answer is the library in use, not a guess from an install location.

The version is read from the path: Homebrew installs into
.../Cellar/libomp/<version>/lib/libomp.dylib. Any other location gives
"unknown" (the path is still recorded).

A process with TWO OpenMP runtimes (for example torch's own libomp next to
Homebrew's) aborts, hangs or crashes, so read_lightgbm_runtime refuses it.
"""

import ctypes
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb

# File names of OpenMP runtimes: LLVM (libomp), GNU (libgomp), Intel (libiomp5).
OPENMP_LIBRARY_PREFIXES = ("libomp", "libgomp", "libiomp")
HOMEBREW_CELLAR = "Cellar"
HOMEBREW_FORMULA = "libomp"
UNKNOWN = "unknown"
NOT_LOADED = "not loaded"


def loaded_library_paths() -> list[str]:
    """Paths of the shared libraries loaded into this process, as the OS reports them."""
    if sys.platform == "darwin":
        process = ctypes.CDLL(None)
        process._dyld_image_count.restype = ctypes.c_uint32
        process._dyld_get_image_name.restype = ctypes.c_char_p
        process._dyld_get_image_name.argtypes = [ctypes.c_uint32]
        names = [
            process._dyld_get_image_name(index) for index in range(process._dyld_image_count())
        ]
        return [name.decode() for name in names if name]
    maps_file = Path("/proc/self/maps")
    if not maps_file.is_file():
        return []
    # A line looks like "7f.. r-xp 0000 08:01 123  /usr/lib/libgomp.so.1": keep the path.
    paths = [line.split()[-1] for line in maps_file.read_text().splitlines() if "/" in line]
    return sorted(set(paths))


def openmp_libraries(paths: Sequence[str]) -> list[str]:
    """The OpenMP runtimes among `paths`, with symbolic links resolved; distinct and sorted."""
    found = {
        os.path.realpath(path)
        for path in paths
        if Path(path).name.startswith(OPENMP_LIBRARY_PREFIXES)
    }
    return sorted(found)


def libomp_version_from_path(path: str) -> str:
    """The Homebrew version in a libomp path, or "unknown".

    ".../Cellar/libomp/22.1.4/lib/libomp.dylib" -> "22.1.4".
    """
    parts = Path(path).parts
    for index, part in enumerate(parts[:-2]):
        if part == HOMEBREW_CELLAR and parts[index + 1] == HOMEBREW_FORMULA:
            return parts[index + 2]
    return UNKNOWN


@dataclass(frozen=True)
class LightGBMRuntime:
    """The libraries a Ranker process runs on.

    Attributes:
        lightgbm_version: lightgbm.__version__, e.g. "4.7.0".
        libomp_path: Resolved path of the loaded OpenMP runtime ("not loaded" if
            the OS lists none).
        libomp_version: Its version, read from the path ("unknown" if the path
            does not tell).
    """

    lightgbm_version: str
    libomp_path: str
    libomp_version: str

    def columns(self) -> dict[str, str]:
        """The three values as result-table columns."""
        return {
            "lightgbm_version": self.lightgbm_version,
            "libomp_path": self.libomp_path,
            "libomp_version": self.libomp_version,
        }


RUNTIME_COLUMNS = ("lightgbm_version", "libomp_path", "libomp_version")


def runtime_from_libraries(lightgbm_version: str, openmp_paths: Sequence[str]) -> LightGBMRuntime:
    """Build the record from the OpenMP runtimes found in a process.

    Args:
        lightgbm_version: The LightGBM version.
        openmp_paths: Output of openmp_libraries for this process.

    Raises:
        RuntimeError: If more than one OpenMP runtime is loaded.
    """
    if len(openmp_paths) > 1:
        raise RuntimeError(
            f"{len(openmp_paths)} OpenMP runtimes are loaded in this process: "
            f"{list(openmp_paths)}. LightGBM must be the only library that brings one "
            "(torch must not be imported in a Ranker process, see AGENTS.md §5)."
        )
    if not openmp_paths:
        return LightGBMRuntime(lightgbm_version, NOT_LOADED, UNKNOWN)
    path = openmp_paths[0]
    return LightGBMRuntime(lightgbm_version, path, libomp_version_from_path(path))


def read_lightgbm_runtime() -> LightGBMRuntime:
    """The LightGBM version and the OpenMP runtime loaded in THIS process.

    Importing lightgbm (at the top of this module) loads its libomp, so the
    library is already in the process when this function looks for it.

    Raises:
        RuntimeError: If more than one OpenMP runtime is loaded.
    """
    return runtime_from_libraries(lgb.__version__, openmp_libraries(loaded_library_paths()))
