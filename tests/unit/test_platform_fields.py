"""Tests for hmrec.platform_fields: what a tolerance-accepted result says it ran on."""

from pathlib import Path

import pytest

import hmrec.platform_fields as platform_fields
from hmrec.git_state import NOT_AN_IMAGE
from hmrec.platform_fields import (
    PLATFORM_COLUMNS,
    openmp_version,
    platform_fields_from,
    read_platform_fields,
)

REPO = Path(__file__).resolve().parents[2]
COMMIT = "0123456789abcdef0123456789abcdef01234567"
TREE = "a" * 64
GOMP = "/usr/lib/aarch64-linux-gnu/libgomp.so.1.0.0"
NUMPY_BLAS = "/venv/lib/python3.12/site-packages/numpy.libs/libscipy_openblas64_-f552eb69.so"
HOMEBREW_OMP = "/opt/homebrew/Cellar/libomp/22.1.4/lib/libomp.dylib"


def test_fields_of_a_linux_container(monkeypatch: pytest.MonkeyPatch) -> None:
    # The package database is asked on Linux; here it answers as the arm64 image does.
    monkeypatch.setattr(
        platform_fields, "debian_package_version", lambda path: "libgomp1 14.2.0-19"
    )
    fields = platform_fields_from(
        "linux", "aarch64", ["/usr/lib/libc.so.6", GOMP, NUMPY_BLAS], COMMIT, TREE
    )
    assert fields.columns() == {
        "os": "linux",
        "architecture": "aarch64",
        "openmp_library": GOMP,
        "openmp_version": "libgomp1 14.2.0-19",
        "blas_libraries": NUMPY_BLAS,
        "image_commit": COMMIT,
        "code_tree_sha256": TREE,
    }
    assert tuple(fields.columns()) == PLATFORM_COLUMNS
    assert fields.name == "linux-arm64"


def test_fields_of_the_laptop() -> None:
    fields = platform_fields_from("darwin", "arm64", [HOMEBREW_OMP], None, NOT_AN_IMAGE)
    assert (fields.os, fields.architecture, fields.name) == ("darwin", "arm64", "darwin-arm64")
    # macOS: the version is the Homebrew version in the path; the package database is not asked.
    assert (fields.openmp_library, fields.openmp_version) == (HOMEBREW_OMP, "22.1.4")
    assert fields.blas_libraries == "not loaded"
    assert (fields.image_commit, fields.code_tree_sha256) == (NOT_AN_IMAGE, NOT_AN_IMAGE)


def test_no_openmp_runtime_and_two_runtimes(monkeypatch: pytest.MonkeyPatch) -> None:
    none = platform_fields_from("linux", "x86_64", ["/usr/lib/libc.so.6"], COMMIT, TREE)
    assert (none.openmp_library, none.openmp_version) == ("not loaded", "unknown")
    assert none.name == "linux-amd64"
    # Two runtimes are both named, each with its own version, in the same order.
    versions = {GOMP: "libgomp1 14.2.0-19", "/venv/torch/lib/libgomp.so.1": "unknown"}
    monkeypatch.setattr(platform_fields, "debian_package_version", lambda path: versions[path])
    two = platform_fields_from(
        "linux", "aarch64", [GOMP, "/venv/torch/lib/libgomp.so.1"], COMMIT, TREE
    )
    assert two.openmp_library == f"{GOMP}; /venv/torch/lib/libgomp.so.1"
    assert two.openmp_version == "libgomp1 14.2.0-19; unknown"


def test_openmp_version_asks_the_package_database_only_on_linux(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_dpkg(path: str) -> str:
        raise AssertionError("the package database must not be asked on macOS")

    monkeypatch.setattr(platform_fields, "debian_package_version", no_dpkg)
    assert openmp_version(HOMEBREW_OMP, "darwin") == "22.1.4"
    assert openmp_version("/somewhere/libomp.dylib", "darwin") == "unknown"


def test_read_platform_fields_of_this_process() -> None:
    import sys

    fields = read_platform_fields(REPO)
    assert fields.os == sys.platform
    if sys.platform == "darwin":
        # The laptop: git describes the code; nothing was stored by a build.
        assert (fields.image_commit, fields.code_tree_sha256) == (NOT_AN_IMAGE, NOT_AN_IMAGE)
    else:
        # A container image: the commit of the image and its verified code tree.
        assert len(fields.image_commit) == 40
        assert len(fields.code_tree_sha256) == 64
