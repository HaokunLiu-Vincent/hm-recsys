"""Which native numeric libraries (OpenMP runtimes, BLAS) a process has really loaded.

Why this exists (docs/mlops.md §8): results depend on libraries that are NOT in
uv.lock or that differ between platforms.

    laptop (macOS, the reference)   LightGBM loads Homebrew's libomp, a system
                                    library; numpy and scipy use Apple's Accelerate.
    container (Linux)               LightGBM loads the GNU OpenMP runtime (libgomp)
                                    of the Debian package libgomp1; numpy and scipy
                                    use the OpenBLAS inside their wheels; torch and
                                    implicit bring their own copies of libgomp.

Another OpenMP runtime or BLAS can change the order of floating-point sums, so a
Linux result is never bit-identical to the laptop by construction. It is accepted
by the tolerance of docs/mlops.md §8.3, and every run records what it ran on.

How the loaded libraries are found: the operating system is asked for the files
this process has loaded (macOS: the dyld image list; Linux: /proc/self/maps), so
the answer is the library in use, not a guess from an install location.

This module is at the bottom layer and imports only the standard library. It must
stay free of torch and lightgbm, because both sides of the project use it
(AGENTS.md §5: torch and LightGBM never share a process).
"""

import ctypes
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

# File names of OpenMP runtimes: LLVM (libomp), GNU (libgomp), Intel (libiomp5).
OPENMP_LIBRARY_PREFIXES = ("libomp", "libgomp", "libiomp")
# File names of BLAS libraries: OpenBLAS (also the copies inside the numpy and scipy
# wheels, "libscipy_openblas..."), Intel MKL, the reference BLAS, BLIS, and the two
# libraries of Apple's Accelerate framework (vecLib).
BLAS_LIBRARY_PREFIXES = (
    "libopenblas",
    "libscipy_openblas",
    "libmkl",
    "libblas",
    "libblis",
    "libBLAS",
    "libLAPACK",
)
UNKNOWN = "unknown"
DPKG_QUERY = "dpkg-query"
HOMEBREW_CELLAR = "Cellar"
HOMEBREW_FORMULA = "libomp"


def libomp_version_from_path(path: str) -> str:
    """The Homebrew version in a libomp path, or "unknown".

    ".../Cellar/libomp/22.1.4/lib/libomp.dylib" -> "22.1.4".
    """
    parts = Path(path).parts
    for index, part in enumerate(parts[:-2]):
        if part == HOMEBREW_CELLAR and parts[index + 1] == HOMEBREW_FORMULA:
            return parts[index + 2]
    return UNKNOWN


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


def libraries_with_prefix(paths: Sequence[str], prefixes: tuple[str, ...]) -> list[str]:
    """The files among `paths` whose name starts with a prefix; links resolved, distinct, sorted."""
    found = {os.path.realpath(path) for path in paths if Path(path).name.startswith(prefixes)}
    return sorted(found)


def openmp_libraries(paths: Sequence[str]) -> list[str]:
    """The OpenMP runtimes among `paths`, with symbolic links resolved; distinct and sorted."""
    return libraries_with_prefix(paths, OPENMP_LIBRARY_PREFIXES)


def blas_libraries(paths: Sequence[str]) -> list[str]:
    """The BLAS libraries among `paths`, with symbolic links resolved; distinct and sorted.

    A BLAS that is linked statically into another library (for example MKL inside
    torch's x86-64 wheel) is not a file of its own and is not listed here;
    torch reports it itself (hmrec.platform_report).
    """
    return libraries_with_prefix(paths, BLAS_LIBRARY_PREFIXES)


def parse_dpkg_search(output: str) -> str:
    """The package name in one line of `dpkg-query --search`.

    "libgomp1:arm64: /usr/lib/aarch64-linux-gnu/libgomp.so.1.0.0" -> "libgomp1:arm64".

    Raises:
        ValueError: If the line does not have the form "<package>: <path>".
    """
    package, separator, _ = output.strip().partition(": ")
    if not separator or not package:
        raise ValueError(f"cannot read a package name from dpkg-query output {output!r}")
    return package


def debian_package_version(path: str) -> str:
    """Name and version of the Debian package that installed the file at `path`.

    Used on Linux for the OpenMP runtime LightGBM loads: the path of libgomp
    (".../libgomp.so.1.0.0") does not tell its version, but the package database
    of the image does. The answer comes from the system this process runs on, so
    it is the installed version, not the one a Dockerfile asked for.

    Args:
        path: Resolved path of a file, e.g. "/usr/lib/aarch64-linux-gnu/libgomp.so.1.0.0".

    Returns:
        "<package> <version>" with the architecture removed from the package
        name, e.g. "libgomp1 14.2.0-19". "unknown" if there is no dpkg on this
        system (not a Debian system) or no package owns the file (for example
        the copy of libgomp inside a wheel).
    """
    if shutil.which(DPKG_QUERY) is None:
        return UNKNOWN
    # check=False on purpose: dpkg-query exits with 1 when no package owns the file,
    # which is an expected answer here ("unknown"), not an error.
    search = subprocess.run(
        [DPKG_QUERY, "--search", path], capture_output=True, text=True, check=False
    )
    if search.returncode != 0:
        return UNKNOWN
    package = parse_dpkg_search(search.stdout.splitlines()[0])
    version = subprocess.run(
        [DPKG_QUERY, "--show", "--showformat=${Version}", package],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return f"{package.partition(':')[0]} {version}"
