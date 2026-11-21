"""Tests for hmrec.native_libraries: loaded OpenMP / BLAS libraries and Debian package versions.

The Linux functions are tested with a stand-in for dpkg-query, so those tests run
on the laptop too. The last tests look at the real system: each has an explicit
branch for macOS (the laptop) and one for Linux (a container image).
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import hmrec.native_libraries as native_libraries
from hmrec.native_libraries import (
    UNKNOWN,
    blas_libraries,
    debian_package_version,
    libraries_with_prefix,
    loaded_library_paths,
    openmp_libraries,
    parse_dpkg_search,
)

REPO = Path(__file__).resolve().parents[2]
LIBGOMP = "/usr/lib/aarch64-linux-gnu/libgomp.so.1.0.0"


def test_importing_the_module_loads_neither_torch_nor_lightgbm() -> None:
    code = (
        "import sys, hmrec.native_libraries; "
        "print('torch' in sys.modules, 'lightgbm' in sys.modules, 'numpy' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=REPO
    )
    assert result.stdout.split() == ["False", "False", "False"]


def test_libraries_are_picked_by_the_start_of_the_file_name() -> None:
    paths = [
        "/nowhere/lib/libgomp.so.1",
        "/nowhere/lib/libgomp.so.1",  # listed twice: kept once
        "/nowhere/torch/lib/libgomp-a34b3233.so.1",  # the copy inside a wheel
        "/nowhere/numpy.libs/libscipy_openblas64_-6bb31eeb.so",
        "/nowhere/Accelerate.framework/Versions/A/libBLAS.dylib",
        "/nowhere/lib/libcompression.dylib",  # "omp" inside the name: not an OpenMP runtime
        "/nowhere/lib/libblasted.txt.d/readme",  # only the FILE name counts, not a folder
    ]
    assert openmp_libraries(paths) == [
        "/nowhere/lib/libgomp.so.1",
        "/nowhere/torch/lib/libgomp-a34b3233.so.1",
    ]
    assert blas_libraries(paths) == [
        "/nowhere/Accelerate.framework/Versions/A/libBLAS.dylib",
        "/nowhere/numpy.libs/libscipy_openblas64_-6bb31eeb.so",
    ]
    assert libraries_with_prefix(paths, ("libcompression",)) == [
        "/nowhere/lib/libcompression.dylib"
    ]
    assert libraries_with_prefix([], ("libgomp",)) == []


def test_symbolic_links_are_resolved(tmp_path: Path) -> None:
    real = tmp_path / "libgomp.so.1.0.0"
    real.write_bytes(b"")
    link = tmp_path / "libgomp.so.1"
    link.symlink_to(real)
    # The link and its target are one library, reported under the real file's path.
    assert openmp_libraries([str(link), str(real)]) == [str(real.resolve())]


def test_the_package_name_is_read_from_a_dpkg_search_line() -> None:
    assert parse_dpkg_search(f"libgomp1:arm64: {LIBGOMP}\n") == "libgomp1:arm64"
    assert parse_dpkg_search("dpkg: /usr/bin/dpkg-query") == "dpkg"
    with pytest.raises(ValueError, match="cannot read a package name"):
        parse_dpkg_search("no separator here")
    with pytest.raises(ValueError, match="cannot read a package name"):
        parse_dpkg_search("")


def _fake_dpkg(monkeypatch: pytest.MonkeyPatch, owner: str | None, version: str) -> list[list[str]]:
    """Replace dpkg-query: `owner` owns every file (None = no package does)."""
    commands: list[list[str]] = []

    def run(command: list[str], **options: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if command[1] == "--search":
            if owner is None:
                return subprocess.CompletedProcess(command, 1, "", "no path found matching")
            return subprocess.CompletedProcess(command, 0, f"{owner}: {command[2]}\n", "")
        return subprocess.CompletedProcess(command, 0, version, "")

    monkeypatch.setattr(native_libraries.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(native_libraries.subprocess, "run", run)
    return commands


def test_the_version_is_the_owning_package_and_its_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands = _fake_dpkg(monkeypatch, "libgomp1:arm64", "14.2.0-19")
    # The architecture is dropped: arm64 and amd64 images give the same text.
    assert debian_package_version(LIBGOMP) == "libgomp1 14.2.0-19"
    assert commands == [
        ["dpkg-query", "--search", LIBGOMP],
        ["dpkg-query", "--show", "--showformat=${Version}", "libgomp1:arm64"],
    ]


def test_a_file_no_package_owns_is_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    # For example the copy of libgomp inside the torch wheel.
    commands = _fake_dpkg(monkeypatch, None, "")
    assert debian_package_version("/x/torch/lib/libgomp-a34b3233.so.1") == UNKNOWN
    assert len(commands) == 1  # the version is not asked for


def test_without_dpkg_the_version_is_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_run(*arguments: object, **options: object) -> None:
        raise AssertionError("nothing must be run when dpkg-query does not exist")

    monkeypatch.setattr(native_libraries.shutil, "which", lambda name: None)
    monkeypatch.setattr(native_libraries.subprocess, "run", no_run)
    assert debian_package_version(LIBGOMP) == UNKNOWN


# ---------------------------------------------------------------- the real system


def test_the_operating_system_lists_the_libraries_of_this_process() -> None:
    paths = loaded_library_paths()
    assert paths, "the OS reported no loaded library"
    names = [Path(path).name for path in paths]
    if sys.platform == "darwin":
        assert any(name.startswith("libSystem") for name in names)
    else:
        # Linux: /proc/self/maps lists the C library among the mapped files.
        assert any(name.startswith("libc.so") for name in names)


def test_package_versions_on_the_real_system() -> None:
    if sys.platform == "darwin":
        # The laptop has no Debian package database: every file is "unknown".
        assert shutil.which("dpkg-query") is None
        assert debian_package_version("/usr/lib/libSystem.B.dylib") == UNKNOWN
    else:
        # The container image is Debian: dpkg-query itself belongs to the package dpkg.
        dpkg_query = shutil.which("dpkg-query")
        assert dpkg_query is not None
        version = debian_package_version(dpkg_query)
        assert version.startswith("dpkg ") and version != "dpkg "
        # A file that no package installed (this test file).
        assert debian_package_version(str(Path(__file__).resolve())) == UNKNOWN
