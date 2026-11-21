"""The platform fields every tolerance-accepted result carries (docs/mlops.md §8.3).

A result computed in a container is not bit-identical to the laptop: Linux uses
another OpenMP runtime and another BLAS. Such a result is accepted by a
tolerance, and it must say what it ran on, so that two results are only ever
compared as equals when their platforms are the same. These fields are written
into every row of the tables of stage 3 (decision of 2026-11-22):

    os, architecture        "linux" / "aarch64", "linux" / "x86_64", "darwin" / "arm64"
    openmp_library          path(s) of the OpenMP runtime(s) loaded in the process
    openmp_version          their versions: the Debian package on Linux
                            ("libgomp1 14.2.0-19"), the Homebrew version on macOS
    blas_libraries          path(s) of the BLAS libraries loaded in the process
    image_commit            the commit the container image was built from
    code_tree_sha256        the verified hash of the image's code tree

The fields are read from the process itself, AFTER the work is done, so they
name the libraries that were really loaded (hmrec.native_libraries).

This module is at the bottom layer. It imports only the standard library and
bottom-layer modules, and must stay free of torch and lightgbm: both sides use it.
"""

import os
import platform
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from hmrec.git_state import NOT_AN_IMAGE, code_tree_of_run, image_commit
from hmrec.native_libraries import (
    UNKNOWN,
    blas_libraries,
    debian_package_version,
    libomp_version_from_path,
    loaded_library_paths,
    openmp_libraries,
)

LINUX = "linux"
NOT_LOADED = "not loaded"
# Several libraries in one field are joined with this (a path never contains it here).
SEPARATOR = "; "
# Names used in folder and file names. Docker and Cloud Run say "arm64" and "amd64";
# the Linux kernel says "aarch64" and "x86_64".
ARCHITECTURE_NAMES = {"aarch64": "arm64", "arm64": "arm64", "x86_64": "amd64", "amd64": "amd64"}

PLATFORM_COLUMNS: tuple[str, ...] = (
    "os",
    "architecture",
    "openmp_library",
    "openmp_version",
    "blas_libraries",
    "image_commit",
    "code_tree_sha256",
)


@dataclass(frozen=True)
class PlatformFields:
    """What a process ran on (see the module docstring).

    Attributes:
        os: sys.platform: "linux" or "darwin".
        architecture: The machine type the kernel reports, e.g. "aarch64".
        openmp_library: Resolved path(s) of the loaded OpenMP runtime(s), joined
            by "; "; "not loaded" if there is none.
        openmp_version: One version per library, in the same order.
        blas_libraries: Resolved path(s) of the loaded BLAS libraries, joined by
            "; "; "not loaded" if there is none.
        image_commit: The commit of the container image; "not an image" on the laptop.
        code_tree_sha256: The verified code-tree hash of the image; "not an image"
            on the laptop.
    """

    os: str
    architecture: str
    openmp_library: str
    openmp_version: str
    blas_libraries: str
    image_commit: str
    code_tree_sha256: str

    @property
    def name(self) -> str:
        """A short name for folders and tables, e.g. "linux-arm64" or "darwin-arm64"."""
        return f"{self.os}-{ARCHITECTURE_NAMES.get(self.architecture, self.architecture)}"

    def columns(self) -> dict[str, str]:
        """The fields as result-table columns, in the order of PLATFORM_COLUMNS."""
        return {name: getattr(self, name) for name in PLATFORM_COLUMNS}


def openmp_version(path: str, system: str) -> str:
    """The version of one OpenMP runtime file.

    Args:
        path: Resolved path of the library.
        system: sys.platform of the process.

    Returns:
        Linux: "<package> <version>" of the Debian package that owns the file,
        or "unknown" (a copy inside a Python wheel belongs to no package).
        macOS: the Homebrew version in the path, or "unknown".
    """
    if system == LINUX:
        return debian_package_version(path)
    return libomp_version_from_path(path)


def joined(values: Sequence[str]) -> str:
    """Several values in one field, or "not loaded" for none."""
    return SEPARATOR.join(values) if values else NOT_LOADED


def platform_fields_from(
    system: str,
    machine: str,
    library_paths: Sequence[str],
    commit_of_image: str | None,
    code_tree: str,
) -> PlatformFields:
    """Build the fields from what was read about a process (no IO except the package query).

    Args:
        system: sys.platform.
        machine: platform.machine().
        library_paths: Paths of every shared library loaded in the process.
        commit_of_image: The image's commit, or None outside an image.
        code_tree: The code-tree hash of the run (hmrec.git_state.code_tree_of_run).
    """
    openmp_paths = openmp_libraries(library_paths)
    versions = [openmp_version(path, system) for path in openmp_paths]
    return PlatformFields(
        os=system,
        architecture=machine,
        openmp_library=joined(openmp_paths),
        openmp_version=joined(versions) if versions else UNKNOWN,
        blas_libraries=joined(blas_libraries(library_paths)),
        image_commit=commit_of_image if commit_of_image is not None else NOT_AN_IMAGE,
        code_tree_sha256=code_tree,
    )


def read_platform_fields(repo_root: Path) -> PlatformFields:
    """The platform fields of THIS process, as it is now.

    Call it after the work is done: a library that was not loaded yet is not
    listed. Inside an image the code tree is verified again here.

    Args:
        repo_root: Root of the code (hmrec.paths.PROJECT_ROOT).
    """
    return platform_fields_from(
        sys.platform,
        platform.machine(),
        loaded_library_paths(),
        image_commit(os.environ),
        code_tree_of_run(repo_root),
    )
