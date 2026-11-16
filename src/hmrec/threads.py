"""Set the Polars thread count from configs/base.yaml BEFORE polars is imported.

Polars reads the environment variable POLARS_MAX_THREADS only once, when it is
first imported. Most hmrec modules import polars at the top, so a script must
call `configure_polars_threads` before importing them:

    from hmrec.threads import configure_polars_threads
    configure_polars_threads()
    import polars as pl  # noqa: E402
    from hmrec.data import ...  # noqa: E402

This module must therefore import only the standard library and yaml, never
polars or another hmrec module that imports polars. tests/unit/test_threads.py
checks that importing it does not import polars.
"""

import os
from pathlib import Path

import yaml

POLARS_THREADS_ENV = "POLARS_MAX_THREADS"

# src/hmrec/threads.py -> parents[2] = repo root. Built by hand instead of with
# hmrec.paths, because hmrec.paths imports hmrec.config (fine) but this module
# must stay importable without any risk of pulling in polars.
DEFAULT_BASE_CONFIG_FILE = Path(__file__).resolve().parents[2] / "configs" / "base.yaml"


def read_num_threads(base_config_file: Path) -> int:
    """Read `num_threads` from base.yaml.

    Args:
        base_config_file: Path to configs/base.yaml.

    Returns:
        The configured number of CPU threads (a positive int).

    Raises:
        ValueError: If num_threads is missing or not a positive integer.
    """
    raw = yaml.safe_load(base_config_file.read_text(encoding="utf-8"))
    num_threads = raw.get("num_threads") if isinstance(raw, dict) else None
    if isinstance(num_threads, bool) or not isinstance(num_threads, int) or num_threads <= 0:
        raise ValueError(
            f"{base_config_file}: num_threads must be a positive integer, got {num_threads!r}"
        )
    return num_threads


def configure_polars_threads(base_config_file: Path = DEFAULT_BASE_CONFIG_FILE) -> int:
    """Set POLARS_MAX_THREADS to num_threads from base.yaml.

    Must be called before polars is imported anywhere in the process.

    Args:
        base_config_file: Path to configs/base.yaml.

    Returns:
        The number of threads that was set.

    Raises:
        RuntimeError: If the environment already sets a DIFFERENT value. We do
            not silently override it, because the user may have set it on purpose.
    """
    num_threads = read_num_threads(base_config_file)
    preset_value = os.environ.get(POLARS_THREADS_ENV)
    if preset_value is not None and preset_value != str(num_threads):
        raise RuntimeError(
            f"{POLARS_THREADS_ENV}={preset_value} is set in the environment, but base.yaml "
            f"num_threads={num_threads}; unset it or make them equal"
        )
    os.environ[POLARS_THREADS_ENV] = str(num_threads)
    return num_threads


def check_polars_thread_pool(expected_threads: int) -> int:
    """Fail if polars did not pick up the configured thread count.

    Call this after polars has been imported, e.g. at the start of main().

    Args:
        expected_threads: num_threads from base.yaml.

    Returns:
        The actual thread pool size (equal to expected_threads).

    Raises:
        RuntimeError: If polars uses a different number of threads, which means
            polars was imported before configure_polars_threads was called.
    """
    # Imported here, not at the top: importing this module must not import polars
    # (see the module docstring). By the time this function runs, the thread
    # count is already fixed, so importing polars now is safe.
    import polars as pl

    pool_size = pl.thread_pool_size()
    if pool_size != expected_threads:
        raise RuntimeError(
            f"polars uses {pool_size} threads, expected num_threads={expected_threads}; "
            "was polars imported before configure_polars_threads()?"
        )
    return pool_size
