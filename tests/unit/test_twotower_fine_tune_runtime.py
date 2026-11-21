"""Tests for hmrec.recall.twotower.fine_tune_runtime: runtime settings around an R6 fine-tune.

Every setting must put back what it changed. The setting that limits torch's
own BLAS exists only where torch brings a BLAS file (Linux arm64); the test has
one branch for a process with such a file and one for a process without.
"""

from pathlib import Path

import pytest
import torch
from threadpoolctl import threadpool_info

from hmrec.recall.twotower.fine_tune_runtime import (
    SETTINGS,
    SettingNotApplicable,
    deterministic_algorithms,
    fine_tune_setting,
    one_intra_op_thread,
    thread_report,
    torch_blas_one_thread,
    torch_blas_paths,
)

TORCH_BLAS = "/venv/lib/python3.12/site-packages/torch/lib/libopenblas.so.0"
SCIPY_BLAS = "/venv/lib/python3.12/site-packages/scipy.libs/libscipy_openblas-b7e2810b.so"


def _pools() -> dict[str, int]:
    """File path of every thread pool of this process -> its thread count."""
    return {pool["filepath"]: pool["num_threads"] for pool in threadpool_info()}


def _torch_blas_of_this_process() -> list[str]:
    blas = [pool["filepath"] for pool in threadpool_info() if pool["user_api"] == "blas"]
    return torch_blas_paths(blas)


def test_torch_blas_paths_keeps_only_files_inside_the_torch_package() -> None:
    assert torch_blas_paths([SCIPY_BLAS, TORCH_BLAS]) == [TORCH_BLAS]
    assert torch_blas_paths([SCIPY_BLAS]) == []
    # A folder that only contains the word is not the torch package.
    assert torch_blas_paths(["/venv/site-packages/torchvision.libs/libblas.so"]) == []


def test_the_settings_that_were_tried() -> None:
    assert SETTINGS == (
        "none",
        "torch_blas_1",
        "deterministic",
        "deterministic+torch_blas_1",
        "one_thread",
    )
    with pytest.raises(ValueError, match="setting must be one of"), fine_tune_setting("fast"):
        pass


def test_none_changes_nothing() -> None:
    before = (thread_report(), torch.are_deterministic_algorithms_enabled())
    with fine_tune_setting("none"):
        assert (thread_report(), torch.are_deterministic_algorithms_enabled()) == before
    assert (thread_report(), torch.are_deterministic_algorithms_enabled()) == before


def test_deterministic_algorithms_are_on_inside_and_set_back_afterwards() -> None:
    assert not torch.are_deterministic_algorithms_enabled()
    threads = torch.get_num_threads()
    with fine_tune_setting("deterministic"):
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.get_num_threads() == threads  # the thread count is not touched
        assert "deterministic algorithms True" in thread_report()
    assert not torch.are_deterministic_algorithms_enabled()
    # Also when the block fails.
    with pytest.raises(ValueError, match="stop"), deterministic_algorithms():
        raise ValueError("stop")
    assert not torch.are_deterministic_algorithms_enabled()


def test_one_thread_inside_and_the_earlier_count_afterwards() -> None:
    torch.set_num_threads(3)
    try:
        with one_intra_op_thread():
            assert torch.get_num_threads() == 1
        assert torch.get_num_threads() == 3
        with pytest.raises(ValueError, match="stop"), fine_tune_setting("one_thread"):
            raise ValueError("stop")
        assert torch.get_num_threads() == 3
    finally:
        torch.set_num_threads(torch.get_num_threads())


def test_the_blas_of_torch_on_this_platform() -> None:
    # A tiny matrix product loads torch's BLAS where it has one.
    assert float((torch.ones(2, 2) @ torch.ones(2, 2)).sum()) == 8.0
    torch_blas = _torch_blas_of_this_process()
    threads = torch.get_num_threads()
    before = _pools()
    if torch_blas:
        # Linux arm64: torch brings an OpenBLAS file. Inside the block it runs on one
        # thread, torch's OpenMP thread count is unchanged, and no other BLAS is touched.
        with torch_blas_one_thread() as limited:
            inside = _pools()
            assert limited == torch_blas
            assert all(inside[path] == 1 for path in torch_blas)
            assert torch.get_num_threads() == threads
            others = [path for path in inside if path not in torch_blas and "gomp" not in path]
            assert all(inside[path] == before[path] for path in others)
        assert _pools() == before
        assert torch.get_num_threads() == threads
    else:
        # macOS (Accelerate) and Linux x86-64 (MKL linked into torch itself): there is no
        # BLAS file of torch to limit, and the setting says so instead of doing nothing.
        assert not any(
            Path(path).parent.name == "lib" and "torch" in path
            for path in before
            if "blas" in Path(path).name
        )
        with (
            pytest.raises(SettingNotApplicable, match="no BLAS library of torch"),
            torch_blas_one_thread(),
        ):
            pass
        with pytest.raises(SettingNotApplicable), fine_tune_setting("deterministic+torch_blas_1"):
            pass
        # A setting that could not be applied leaves nothing behind.
        assert not torch.are_deterministic_algorithms_enabled()
        assert _pools() == before
