"""Tests for hmrec.threads (Polars thread setup before polars is imported)."""

import subprocess
import sys
from pathlib import Path

import pytest

from hmrec.threads import (
    POLARS_THREADS_ENV,
    check_polars_thread_pool,
    configure_polars_threads,
    read_num_threads,
)

REPO_BASE_YAML = Path(__file__).resolve().parents[2] / "configs" / "base.yaml"


def test_importing_threads_does_not_import_polars() -> None:
    # Run in a fresh interpreter: in this test process polars is already imported.
    code = "import sys, hmrec.threads; sys.exit(1 if 'polars' in sys.modules else 0)"
    result = subprocess.run([sys.executable, "-c", code], check=False)
    assert result.returncode == 0


def test_read_num_threads_from_repo_config() -> None:
    assert read_num_threads(REPO_BASE_YAML) == 6


def test_read_num_threads_rejects_bad_value(tmp_path: Path) -> None:
    config = tmp_path / "base.yaml"
    config.write_text("num_threads: 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="positive integer"):
        read_num_threads(config)


def test_configure_sets_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(POLARS_THREADS_ENV, raising=False)
    assert configure_polars_threads(REPO_BASE_YAML) == 6
    # Imported inside the test so the env var is read back after configure_polars_threads.
    import os

    assert os.environ[POLARS_THREADS_ENV] == "6"


def test_configure_rejects_conflicting_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(POLARS_THREADS_ENV, "3")
    with pytest.raises(RuntimeError, match="POLARS_MAX_THREADS=3"):
        configure_polars_threads(REPO_BASE_YAML)


def test_check_polars_thread_pool_detects_mismatch() -> None:
    import polars as pl

    actual = pl.thread_pool_size()
    assert check_polars_thread_pool(actual) == actual
    with pytest.raises(RuntimeError, match="expected num_threads"):
        check_polars_thread_pool(actual + 1)
