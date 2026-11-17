"""Run the steps of a fold one after another, each in its OWN Python process (M6).

Why one process per step: a step's true memory peak is the process max RSS
(hmrec.memory), which never goes down. If several steps shared a process, only
the largest peak could be measured. Separate processes also give every step
the whole memory budget, because nothing of the previous step is still loaded.

Steps of one fold, in order: the six channels R1 ... R6 (R6 includes its weekly
fine-tune), then the monitor. A step that fails (non-zero exit) stops
everything: the child process has already logged why (schema or consistency
check, sha256 mismatch, loss check, memory above the limit).
"""

import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

from hmrec.eval.monitor import NOT_COMPUTED, UNION, flagged_channels
from hmrec.locks import ALL_CHANNELS
from hmrec.logging_utils import get_logger

LOGGER = get_logger("pipeline")

MONITOR_STEP = "monitor"


def run_script(script: Path, arguments: Sequence[str]) -> float:
    """Run a Python script in a new process with this interpreter and wait for it.

    Args:
        script: Path of the script, e.g. scripts/m6_generate_candidates.py.
        arguments: Its command-line arguments.

    Returns:
        Wall time in seconds.

    Raises:
        subprocess.CalledProcessError: If the script exits with a non-zero code.
    """
    command = [sys.executable, str(script), *arguments]
    LOGGER.info("starting: %s", " ".join(command[1:]))
    started = time.perf_counter()
    subprocess.run(command, check=True)
    return time.perf_counter() - started


def fold_steps(
    fold: str, generate_script: Path, monitor_script: Path
) -> list[tuple[str, Path, list[str]]]:
    """The steps of one fold as (step name, script, arguments), in the order they must run."""
    steps = [
        (channel, generate_script, ["--fold", fold, "--channel", channel])
        for channel in ALL_CHANNELS
    ]
    steps.append((MONITOR_STEP, monitor_script, ["--fold", fold]))
    return steps


def _percent(text: str) -> str:
    """A recall cell as a percentage; "n/a" for an empty cell (T folds have no recall)."""
    return "n/a" if text == NOT_COMPUTED else f"{float(text) * 100:.2f}%"


def fold_summary(
    fold: str, run_rows: Sequence[Mapping[str, str]], monitor_rows: Sequence[Mapping[str, str]]
) -> str:
    """One log line for a finished fold, from its rows of the two results files.

    Args:
        fold: Fold name.
        run_rows: Rows of results/m6_runs.csv (any fold; filtered here).
        monitor_rows: Rows of results/m6_candidates.csv (any fold; filtered here).

    Returns:
        E.g. "F1 | union 306.9 cand/query, recall 24.31% | r1_popularity 7.01% ... |
        max process RSS 4.20 GB (r4_itemcf) | 251 s | low recall: none".
    """
    runs = [row for row in run_rows if row["fold"] == fold]
    monitors = [row for row in monitor_rows if row["fold"] == fold]
    if not runs or not monitors:
        raise ValueError(f"fold {fold} has no rows yet in the results files")
    union = next(row for row in monitors if row["scope"] == UNION)
    channels = [row for row in monitors if row["scope"] != UNION]
    recalls = " ".join(f"{row['scope']} {_percent(row['recall'])}" for row in channels)
    heaviest = max(runs, key=lambda row: float(row["process_max_rss_gb"]))
    seconds = sum(float(row["total_seconds"]) for row in runs)
    flags = flagged_channels(monitors)
    return (
        f"{fold} | union {float(union['candidates_per_query_mean']):.1f} cand/query, "
        f"recall {_percent(union['recall'])} | {recalls} | "
        f"max process RSS {float(heaviest['process_max_rss_gb']):.2f} GB ({heaviest['step']}) | "
        f"{seconds:.0f} s | low recall: {', '.join(flags) if flags else 'none'}"
    )
