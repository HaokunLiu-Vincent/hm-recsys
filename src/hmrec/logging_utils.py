"""One shared logger setup for the whole project (AGENTS.md §6.4).

All project loggers live under the "hmrec" namespace:
    - modules in src/: `logger = get_logger(__name__)` gives "hmrec.recall.r1_popularity" etc.
    - scripts: `get_logger("m0_preprocess")` gives "hmrec.m0_preprocess".

Scripts call `setup_logging()` once at start-up. Modules never configure
handlers themselves. Use logging instead of print (AGENTS.md §3.4).
"""

import logging
import sys
from pathlib import Path
from typing import TextIO

ROOT_LOGGER_NAME = "hmrec"
LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """Return a logger inside the "hmrec" namespace.

    Args:
        name: Usually `__name__`. Names that do not already start with "hmrec"
            (for example "__main__" in a script) get the prefix "hmrec.".

    Returns:
        The logger. It has no handlers of its own; messages go up to the
        "hmrec" logger configured by setup_logging.
    """
    if name == ROOT_LOGGER_NAME or name.startswith(ROOT_LOGGER_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")


def setup_logging(
    level: int = logging.INFO,
    log_file: Path | None = None,
    stream: TextIO | None = None,
) -> logging.Logger:
    """Configure the "hmrec" logger. Safe to call more than once.

    Each call removes the handlers added by a previous call, so log lines are
    never printed twice.

    Args:
        level: Minimum level to log, e.g. logging.INFO.
        log_file: If given, also write logs to this file (parent folders are created).
            Typically artifacts/runs/{run_id}/run.log.
        stream: Where console logs go. Defaults to sys.stderr.

    Returns:
        The configured "hmrec" logger.
    """
    root_logger = logging.getLogger(ROOT_LOGGER_NAME)
    _remove_handlers(root_logger)
    root_logger.setLevel(level)
    # Do not pass records on to Python's root logger: other libraries may have
    # configured it, which would print every line twice.
    root_logger.propagate = False

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    console_handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return root_logger


def _remove_handlers(logger: logging.Logger) -> None:
    """Remove and close all handlers of `logger`."""
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
