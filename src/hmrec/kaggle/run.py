"""The driver of a Kaggle run: every step in its own Python process, in order.

Why one process per step: a step's true memory peak is its process max RSS
(hmrec.memory), and torch (recall) and LightGBM (ranker) must never share a
process (their two OpenMP runtimes abort or hang).

Steps of a run, in order (each is safe to start again: finished files are
skipped, never replaced):

    prepare                     free disk, customers, query list
    recall x 6                  one process per channel: fit once, all shards
    features --prepare          the tables over all customers, once
    features --shard k          one process per shard
    ranker sample               only folds M8 did not sample (the real run: T4)
    ranker dataset, train       the model, trained from scratch
    ranker score                the Top-K of every shard
    submission                  the file, its format checks, the candidate counts
    ranker check                the rehearsal only: the comparison with M8

A step that fails (non-zero exit) stops the run; the child process has logged why.

This module imports neither torch nor LightGBM.
"""

import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from hmrec.kaggle.plan import RUN_REHEARSAL, RUN_SUBMISSION, check_run
from hmrec.locks import ALL_CHANNELS
from hmrec.logging_utils import get_logger

LOGGER = get_logger("kaggle.run")

PREPARE_SCRIPT = "kaggle_prepare.py"
RECALL_SCRIPT = "kaggle_recall.py"
FEATURES_SCRIPT = "kaggle_features.py"
RANKER_SCRIPT = "kaggle_ranker.py"
SUBMISSION_SCRIPT = "kaggle_submission.py"

# A step: (script file name, its arguments after --run).
Step = tuple[str, list[str]]


def steps_before_shards(run: str) -> list[Step]:
    """The steps that do not need the number of shards: prepare, then the six channels."""
    check_run(run)
    steps: list[Step] = [(PREPARE_SCRIPT, [])]
    steps += [(RECALL_SCRIPT, ["--channel", channel]) for channel in ALL_CHANNELS]
    steps.append((FEATURES_SCRIPT, ["--prepare"]))
    return steps


def steps_after_recall(run: str, n_shards: int) -> list[Step]:
    """The remaining steps, once the query list (and so the number of shards) exists."""
    check_run(run)
    if n_shards <= 0:
        raise ValueError(f"a run has at least one shard, got {n_shards}")
    steps: list[Step] = [(FEATURES_SCRIPT, ["--shard", str(shard)]) for shard in range(n_shards)]
    if run == RUN_SUBMISSION:
        steps.append((RANKER_SCRIPT, ["--step", "sample"]))
    steps += [(RANKER_SCRIPT, ["--step", step]) for step in ("dataset", "train", "score")]
    steps.append((SUBMISSION_SCRIPT, []))
    if run == RUN_REHEARSAL:
        steps.append((RANKER_SCRIPT, ["--step", "check"]))
    return steps


def run_script(script: Path, arguments: Sequence[str]) -> float:
    """Run a Python script in a new process with this interpreter and wait for it.

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


def run_steps(scripts_dir: Path, run: str, steps: Sequence[Step]) -> float:
    """Run the given steps one after another; the first failure stops everything.

    Returns:
        Total wall time in seconds.
    """
    total = 0.0
    for script, arguments in steps:
        seconds = run_script(scripts_dir / script, ["--run", run, *arguments])
        LOGGER.info("finished in %.1f s: %s %s", seconds, script, " ".join(arguments))
        total += seconds
    return total
