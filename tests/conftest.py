"""Split the test suite into two processes: torch-side and LightGBM-side.

WHY: torch and LightGBM each load their own copy of the OpenMP runtime (libomp).
Two copies in one process abort ("OMP: Error #15"), hang or crash. So a test
process may use torch OR LightGBM, never both.

HOW:
    uv run pytest tests/unit                    the torch-side tests (everything up to M6)
    uv run pytest tests/unit --lightgbm-side    the LightGBM-side tests (the Ranker, M7)
`make test-fast` and `make test` run both, one after the other.

A test file belongs to the LightGBM side if its name starts with one of
LIGHTGBM_SIDE_PREFIXES. The other side's files are not even imported (importing
them would load the other library). At the end of a run this file fails the run
if both libraries were imported anyway, so a new test that mixes them is caught.
"""

import sys
from pathlib import Path

import pytest

LIGHTGBM_SIDE_PREFIXES = (
    "test_ranker_",
    "test_v_",
    "test_speed_test",
    "test_m7_",
    "test_final_",  # M8, the final evaluation: it trains LightGBM models too
    # The diagnostics after M8 train nothing, but they read the Ranker lock and the metric
    # tables through modules that import lightgbm (hmrec.ranker.locked, hmrec.hpo.v_records).
    "test_diag_",
)
LIGHTGBM_OPTION = "--lightgbm-side"
_skipped_files: list[str] = []


def is_lightgbm_side(path: Path) -> bool:
    """True if a test file belongs to the LightGBM-side process."""
    return path.name.startswith(LIGHTGBM_SIDE_PREFIXES)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add --lightgbm-side: run the LightGBM-side test files instead of the torch-side ones."""
    parser.addoption(
        LIGHTGBM_OPTION,
        action="store_true",
        default=False,
        help="run the LightGBM-side tests (Ranker, M7) instead of the torch-side tests",
    )


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool | None:
    """Leave out (do not import) the test files of the other side."""
    is_test_file = collection_path.is_file() and collection_path.name.startswith("test_")
    if not is_test_file or collection_path.suffix != ".py":
        return None
    wanted = bool(config.getoption(LIGHTGBM_OPTION))
    if is_lightgbm_side(collection_path) != wanted:
        _skipped_files.append(collection_path.name)
        return True
    return None


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter) -> None:
    """Say which side ran, so a plain `pytest` run is not mistaken for the whole suite."""
    lightgbm_side = bool(terminalreporter.config.getoption(LIGHTGBM_OPTION))
    ran = "LightGBM-side" if lightgbm_side else "torch-side"
    other = "without" if lightgbm_side else "with"
    terminalreporter.write_line(
        f"{ran} tests only: {len(_skipped_files)} test files of the other side were not "
        f"collected (run again {other} {LIGHTGBM_OPTION}; `make test-fast` runs both)."
    )


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Fail the run if torch and LightGBM were both imported in this process."""
    if "torch" in sys.modules and "lightgbm" in sys.modules:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
        sys.stderr.write(
            "\nERROR: torch and lightgbm were both imported in this test process. Their two "
            "OpenMP runtimes must not share a process (see tests/conftest.py).\n"
        )
