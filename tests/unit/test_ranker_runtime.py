"""Tests for hmrec.ranker.runtime: the LightGBM version and the loaded OpenMP runtime."""

import subprocess
import sys
from pathlib import Path

import lightgbm as lgb
import pytest

import hmrec.ranker.runtime as runtime_module
from hmrec.ranker.runtime import (
    NOT_LOADED,
    RUNTIME_COLUMNS,
    UNKNOWN,
    debian_package_version,
    libomp_version_from_path,
    loaded_library_paths,
    openmp_libraries,
    read_lightgbm_runtime,
    runtime_from_libraries,
    with_platform_version,
)

REPO = Path(__file__).resolve().parents[2]
HOMEBREW = "/opt/homebrew/Cellar/libomp/22.1.4/lib/libomp.dylib"
LIBGOMP = "/usr/lib/aarch64-linux-gnu/libgomp.so.1.0.0"


def test_version_is_read_from_a_homebrew_path() -> None:
    assert libomp_version_from_path(HOMEBREW) == "22.1.4"
    assert libomp_version_from_path("/usr/local/Cellar/libomp/19.1.0_1/lib/libomp.dylib") == (
        "19.1.0_1"
    )
    # Not a Homebrew location (torch's own copy, a system path): the version is not guessed.
    assert libomp_version_from_path("/x/site-packages/torch/lib/libomp.dylib") == UNKNOWN
    assert libomp_version_from_path("/usr/lib/x86_64-linux-gnu/libgomp.so.1") == UNKNOWN
    assert libomp_version_from_path("/opt/homebrew/Cellar/llvm/22.1.4/lib/libomp.dylib") == UNKNOWN


def test_openmp_libraries_are_picked_by_file_name() -> None:
    paths = [
        "/usr/lib/libcompression.dylib",  # "omp" inside the name, but not an OpenMP runtime
        "/usr/lib/system/libcompiler_rt.dylib",
        "/nowhere/lib/libomp.dylib",
        "/nowhere/lib/libomp.dylib",  # listed twice: kept once
        "/nowhere/lib/libgomp.so.1",
    ]
    assert openmp_libraries(paths) == ["/nowhere/lib/libgomp.so.1", "/nowhere/lib/libomp.dylib"]
    assert openmp_libraries(paths[:2]) == []


def test_runtime_record_from_the_found_libraries() -> None:
    runtime = runtime_from_libraries("4.7.0", [HOMEBREW])
    assert runtime.columns() == {
        "lightgbm_version": "4.7.0",
        "libomp_path": HOMEBREW,
        "libomp_version": "22.1.4",
    }
    assert tuple(runtime.columns()) == RUNTIME_COLUMNS
    none = runtime_from_libraries("4.7.0", [])
    assert (none.libomp_path, none.libomp_version) == (NOT_LOADED, UNKNOWN)
    # Two OpenMP runtimes in one process abort, hang or crash: refuse to go on.
    with pytest.raises(RuntimeError, match="2 OpenMP runtimes are loaded"):
        runtime_from_libraries("4.7.0", [HOMEBREW, "/x/torch/lib/libomp.dylib"])


def test_this_process_uses_one_openmp_runtime() -> None:
    runtime = read_lightgbm_runtime()
    assert runtime.lightgbm_version == lgb.__version__
    # LightGBM is imported, so its OpenMP runtime is loaded; it is a real file.
    assert Path(runtime.libomp_path).is_file()
    assert Path(runtime.libomp_path).name.startswith(("libomp", "libgomp", "libiomp"))
    assert runtime.libomp_path in openmp_libraries(loaded_library_paths())
    if sys.platform == "darwin":
        # The laptop: the version is the one in the Homebrew path.
        assert runtime.libomp_version == libomp_version_from_path(runtime.libomp_path)
    else:
        # A Linux container: GNU libgomp of the Debian package libgomp1, with the version
        # the package database of the image gives. It is known, and it names the package,
        # so it cannot be taken for a Homebrew libomp version.
        assert Path(runtime.libomp_path).name.startswith("libgomp")
        assert runtime.libomp_version == debian_package_version(runtime.libomp_path)
        assert runtime.libomp_version.startswith("libgomp1 ")
        assert runtime.libomp_version != "libgomp1 "


def test_on_macos_the_record_is_returned_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_lookup(path: str) -> str:
        raise AssertionError("the package database must not be asked on macOS")

    monkeypatch.setattr(runtime_module, "debian_package_version", no_lookup)
    homebrew = runtime_from_libraries("4.7.0", [HOMEBREW])
    assert with_platform_version(homebrew, "darwin") is homebrew
    # A libomp outside Homebrew stays "unknown" on macOS, as in the recorded runs.
    other = runtime_from_libraries("4.7.0", ["/x/lib/libomp.dylib"])
    assert with_platform_version(other, "darwin").libomp_version == UNKNOWN


def test_on_linux_the_version_comes_from_the_package_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked: list[str] = []

    def lookup(path: str) -> str:
        asked.append(path)
        return "libgomp1 14.2.0-19"

    monkeypatch.setattr(runtime_module, "debian_package_version", lookup)
    from_path = runtime_from_libraries("4.7.0", [LIBGOMP])
    assert from_path.libomp_version == UNKNOWN  # the path of libgomp tells no version
    on_linux = with_platform_version(from_path, "linux")
    assert on_linux.columns() == {
        "lightgbm_version": "4.7.0",
        "libomp_path": LIBGOMP,
        "libomp_version": "libgomp1 14.2.0-19",
    }
    assert asked == [LIBGOMP]
    # No OpenMP runtime loaded: there is no file to ask about.
    none = runtime_from_libraries("4.7.0", [])
    assert with_platform_version(none, "linux") is none
    assert asked == [LIBGOMP]


def test_a_process_that_also_imports_torch_is_refused() -> None:
    # Importing both does not crash yet (the crash comes when both runtimes work), so the
    # check can report the problem before any training starts.
    code = (
        "import torch\n"
        "from hmrec.ranker.runtime import read_lightgbm_runtime\n"
        "try:\n"
        "    read_lightgbm_runtime()\n"
        "except RuntimeError as error:\n"
        "    print('refused:', str(error)[:40])\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=REPO
    )
    assert result.stdout.startswith("refused: 2 OpenMP runtimes are loaded")
