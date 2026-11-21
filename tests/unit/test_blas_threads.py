"""Tests for hmrec.blas_threads: scipy's BLAS on one thread during an R5 fit, Linux only.

The two platform branches are tested on every machine with a stand-in for
threadpoolctl's controller (the function takes the platform as an argument).
One test then runs the real libraries of the machine it runs on: in a Linux
container it checks the thread counts of every pool, on the laptop it checks
that nothing changes.
"""

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

import pytest
import torch
from scipy import linalg
from threadpoolctl import threadpool_info

import hmrec.blas_threads as blas_threads
from hmrec.blas_threads import scipy_blas_paths, single_threaded_scipy_blas

SCIPY_BLAS = "/venv/lib/python3.12/site-packages/scipy.libs/libscipy_openblas-b7e2810b.so"
NUMPY_BLAS = "/venv/lib/python3.12/site-packages/numpy.libs/libscipy_openblas64_-f552eb69.so"
TORCH_BLAS = "/venv/lib/python3.12/site-packages/torch/lib/libopenblas.so.0"
GOMP = "/usr/lib/aarch64-linux-gnu/libgomp.so.1.0.0"


def test_scipy_blas_paths_keeps_only_the_library_of_the_scipy_wheel() -> None:
    assert scipy_blas_paths([NUMPY_BLAS, SCIPY_BLAS, TORCH_BLAS]) == [SCIPY_BLAS]
    assert scipy_blas_paths([NUMPY_BLAS, TORCH_BLAS]) == []
    assert scipy_blas_paths([]) == []


@dataclass
class FakeLibrary:
    """One thread pool as threadpoolctl describes it."""

    filepath: str
    user_api: str


@dataclass
class FakeController:
    """Stands in for threadpoolctl.ThreadpoolController; records what was limited."""

    lib_controllers: list[FakeLibrary]
    events: list[str] = field(default_factory=list)
    selected: list[str] = field(default_factory=list)

    def select(self, filepath: list[str]) -> "FakeController":
        self.selected = list(filepath)
        return self

    @contextmanager
    def limit(self, limits: int) -> Iterator[None]:
        self.events.append(f"limit {limits}")
        yield
        self.events.append("restored")


def _use_fake_controller(
    monkeypatch: pytest.MonkeyPatch, libraries: list[FakeLibrary]
) -> FakeController:
    controller = FakeController(libraries)
    monkeypatch.setattr(blas_threads, "ThreadpoolController", lambda: controller)
    return controller


def test_on_linux_only_the_blas_of_scipy_is_limited_and_then_restored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The pools of the arm64 image (measured 2026-11-22): two copies of OpenBLAS from the
    # numpy and scipy wheels, torch's OpenBLAS, and the OpenMP runtime.
    controller = _use_fake_controller(
        monkeypatch,
        [
            FakeLibrary(NUMPY_BLAS, "blas"),
            FakeLibrary(SCIPY_BLAS, "blas"),
            FakeLibrary(TORCH_BLAS, "blas"),
            FakeLibrary(GOMP, "openmp"),
        ],
    )
    with single_threaded_scipy_blas("linux") as limited:
        assert limited == [SCIPY_BLAS]
        assert controller.events == ["limit 1"]
    # numpy's BLAS, torch's BLAS and the OpenMP runtime were never selected.
    assert controller.selected == [SCIPY_BLAS]
    assert controller.events == ["limit 1", "restored"]


def test_on_linux_the_limit_is_restored_when_the_block_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _use_fake_controller(monkeypatch, [FakeLibrary(SCIPY_BLAS, "blas")])
    with pytest.raises(ValueError, match="fit failed"), single_threaded_scipy_blas("linux"):
        raise ValueError("fit failed")
    assert controller.selected == [SCIPY_BLAS]


def test_on_linux_a_process_without_the_blas_of_scipy_is_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Without this error the "limit" would silently limit nothing.
    _use_fake_controller(
        monkeypatch, [FakeLibrary(NUMPY_BLAS, "blas"), FakeLibrary(GOMP, "openmp")]
    )
    with (
        pytest.raises(RuntimeError, match="no BLAS library of scipy is loaded"),
        single_threaded_scipy_blas("linux"),
    ):
        pass


def test_on_macos_nothing_is_touched(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_controller() -> None:
        raise AssertionError("threadpoolctl must not be used on the laptop")

    monkeypatch.setattr(blas_threads, "ThreadpoolController", no_controller)
    with single_threaded_scipy_blas("darwin") as limited:
        assert limited == []


def _threads_by_file() -> dict[str, int]:
    """File path of every thread pool of this process -> its current thread count."""
    return {pool["filepath"]: pool["num_threads"] for pool in threadpool_info()}


def test_with_the_real_libraries_of_this_machine() -> None:
    # scipy's BLAS is loaded by this import; torch is imported at the top of the file.
    assert linalg.norm([3.0, 4.0]) == 5.0
    before = _threads_by_file()
    torch_threads = torch.get_num_threads()
    with single_threaded_scipy_blas(sys.platform) as limited:
        inside = _threads_by_file()
        torch_inside = torch.get_num_threads()
    after = _threads_by_file()
    # On every platform: torch's thread count never changes, and everything is as before
    # once the block has ended.
    assert torch_inside == torch_threads
    assert torch.get_num_threads() == torch_threads
    assert after == before
    if sys.platform == "linux":
        # A container: exactly the BLAS of scipy runs on one thread inside the block;
        # every other pool (numpy's BLAS, torch's BLAS, the OpenMP runtime) is unchanged.
        assert len(limited) == 1 and "scipy.libs" in limited[0]
        assert inside[limited[0]] == 1
        others = {path: count for path, count in inside.items() if path != limited[0]}
        assert others == {path: count for path, count in before.items() if path != limited[0]}
    else:
        # The laptop: nothing is limited and no pool changes.
        assert limited == []
        assert inside == before
