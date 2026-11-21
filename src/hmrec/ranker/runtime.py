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

On macOS (the laptop, the reference of every bit-identical check) the version
is read from the path: Homebrew installs into
.../Cellar/libomp/<version>/lib/libomp.dylib. Any other location gives
"unknown" (the path is still recorded).

On Linux (a container, docs/mlops.md §8) LightGBM loads the GNU OpenMP runtime
(libgomp) of the image, not LLVM's libomp. Its path does not tell a version, so
the version is asked from the package database of the system the process runs
on, and recorded with the package name: "libgomp1 14.2.0-19". The two columns
keep their names (libomp_path, libomp_version) because the result tables have a
fixed header; on Linux they describe libgomp. A Linux value can never equal a
Homebrew version such as "22.1.4" (it starts with a package name), so the check
of the Ranker lock (hmrec.ranker.locked.check_runtime_matches_lock) never takes
a Linux process for the runtime the V phase ran on: a Linux result is not
bit-identical to the laptop and is accepted by the tolerance of docs/mlops.md
§8.3 instead.

A process with TWO OpenMP runtimes (for example torch's own libomp next to
Homebrew's) aborts, hangs or crashes, so read_lightgbm_runtime refuses it.

The functions that list loaded libraries live in hmrec.native_libraries (bottom
layer, no lightgbm), because the torch side reports its libraries too; they are
re-exported here under the names they always had.
"""

import sys
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import lightgbm as lgb

from hmrec.native_libraries import (  # noqa: F401  (re-exported, see the module docstring)
    OPENMP_LIBRARY_PREFIXES,
    UNKNOWN,
    debian_package_version,
    loaded_library_paths,
    openmp_libraries,
)

HOMEBREW_CELLAR = "Cellar"
HOMEBREW_FORMULA = "libomp"
NOT_LOADED = "not loaded"
LINUX = "linux"


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
        libomp_version: Its version: read from the Homebrew path on macOS, from
            the package database on Linux, where it carries the package name
            ("libgomp1 14.2.0-19"); "unknown" if neither tells.
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


def with_platform_version(runtime: LightGBMRuntime, platform: str) -> LightGBMRuntime:
    """Fill in the OpenMP version on Linux, where the path does not tell it.

    On macOS the record is returned as it is: the version comes from the
    Homebrew path, exactly as in every recorded run of stages 1 and 2.

    Args:
        runtime: The record built from the loaded libraries (runtime_from_libraries).
        platform: sys.platform of this process ("darwin" = macOS, "linux").

    Returns:
        On Linux with a loaded runtime: the same record with libomp_version set
        to "<package> <version>" from the package database, e.g.
        "libgomp1 14.2.0-19" ("unknown" if no package owns the file). Otherwise
        `runtime` unchanged.
    """
    if platform != LINUX or runtime.libomp_path == NOT_LOADED:
        return runtime
    return replace(runtime, libomp_version=debian_package_version(runtime.libomp_path))


def read_lightgbm_runtime() -> LightGBMRuntime:
    """The LightGBM version and the OpenMP runtime loaded in THIS process.

    Importing lightgbm (at the top of this module) loads its OpenMP runtime, so
    the library is already in the process when this function looks for it.

    Raises:
        RuntimeError: If more than one OpenMP runtime is loaded.
    """
    runtime = runtime_from_libraries(lgb.__version__, openmp_libraries(loaded_library_paths()))
    return with_platform_version(runtime, sys.platform)
