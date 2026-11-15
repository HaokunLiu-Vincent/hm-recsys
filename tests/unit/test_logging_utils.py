import io
import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from hmrec.logging_utils import ROOT_LOGGER_NAME, get_logger, setup_logging


@pytest.fixture(autouse=True)
def reset_hmrec_logger() -> Iterator[None]:
    """Undo setup_logging() after each test so other tests see a default logger."""
    yield
    root_logger = logging.getLogger(ROOT_LOGGER_NAME)
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        handler.close()
    root_logger.propagate = True
    root_logger.setLevel(logging.NOTSET)


def test_get_logger_adds_namespace_prefix() -> None:
    assert get_logger("__main__").name == "hmrec.__main__"
    assert get_logger("hmrec.recall.r1_popularity").name == "hmrec.recall.r1_popularity"
    assert get_logger("hmrec").name == "hmrec"
    # "hmrecX" is not inside the namespace, so it gets the prefix.
    assert get_logger("hmrecX").name == "hmrec.hmrecX"


def test_setup_logging_formats_and_filters_by_level() -> None:
    stream = io.StringIO()
    setup_logging(level=logging.INFO, stream=stream)
    logger = get_logger("unit")

    logger.debug("hidden")
    logger.info("hello %d", 42)

    output = stream.getvalue()
    assert "hidden" not in output
    assert "| INFO    | hmrec.unit | hello 42" in output


def test_setup_logging_twice_does_not_duplicate_lines() -> None:
    stream = io.StringIO()
    setup_logging(stream=stream)
    setup_logging(stream=stream)

    get_logger("unit").info("once")

    assert stream.getvalue().count("once") == 1


def test_setup_logging_writes_log_file(tmp_path: Path) -> None:
    log_file = tmp_path / "runs" / "r1" / "run.log"
    setup_logging(log_file=log_file, stream=io.StringIO())

    get_logger("unit").warning("to file")
    for handler in logging.getLogger(ROOT_LOGGER_NAME).handlers:
        handler.flush()

    assert "to file" in log_file.read_text()
