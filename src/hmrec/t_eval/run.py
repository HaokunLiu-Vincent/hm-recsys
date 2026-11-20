"""Run the trainings of the M9 evaluation on T1-T4 in order (proposal §16.12).

Every training is ONE process (scripts/m9_t_trial.py), so its process max RSS is
its own true peak.

    rehearsal   the chosen configuration on the rehearsal fold, every seed, then
                the comparison with the V records (hmrec.t_eval.rehearsal). A
                training already recorded AT HEAD is not started again.
    formal      the gate first (hmrec.t_eval.gate.begin_formal_run): the rehearsal
                must have passed at HEAD, and the read of T labels is recorded
                before any training starts. Then every seed on every T fold. A
                training that is recorded is not started again, so an interrupted
                run is continued by the same command.

This module starts processes and reads result tables; it imports neither torch
nor LightGBM.
"""

import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from hmrec.logging_utils import get_logger
from hmrec.t_eval.gate import begin_formal_run, require_clean_tree
from hmrec.t_eval.plan import (
    STAGE_FORMAL,
    STAGE_REHEARSAL,
    TEvalContext,
    TrialKey,
    area_of,
    read_csv_table,
)
from hmrec.t_eval.rehearsal import Check, check_rehearsal, trial_rows_of, write_check_row

LOGGER = get_logger("t_eval.run")

Launcher = Callable[[TrialKey], None]


def key_arguments(key: TrialKey) -> list[str]:
    """The command-line arguments of scripts/m9_t_trial.py for one training."""
    return ["--stage", key.stage, "--seed", str(key.seed), "--target", key.target_fold]


def subprocess_launcher(trial_script: Path) -> Launcher:
    """A launcher that runs one training with the trial script in a new process.

    A non-zero exit (a failed check, a NaN loss, a memory peak above the limit)
    raises subprocess.CalledProcessError and so stops the run.
    """

    def launch(key: TrialKey) -> None:
        command = [sys.executable, str(trial_script), *key_arguments(key)]
        subprocess.run(command, check=True)

    return launch


def stage_keys(context: TEvalContext, stage: str) -> list[TrialKey]:
    """Every training of a stage, in the order it is run (seed by seed, fold by fold).

    Raises:
        ValueError: If the stage is unknown.
    """
    config = context.config
    if stage == STAGE_REHEARSAL:
        folds: tuple[str, ...] = (config.rehearsal_target_fold,)
    elif stage == STAGE_FORMAL:
        folds = config.target_folds
    else:
        raise ValueError(f"unknown stage {stage!r}")
    return [TrialKey(stage, seed, fold) for seed in config.seeds for fold in folds]


def missing_keys(context: TEvalContext, keys: Sequence[TrialKey]) -> list[TrialKey]:
    """The trainings that have no trial row yet.

    The area of a rehearsal holds the commit, so a rehearsal training counts
    only if it was recorded at HEAD. The formal run has one area: a formal
    training counts whenever it is recorded (the gate allows one commit only).

    Raises:
        ValueError: If a training is recorded more than once.
    """
    rows = read_csv_table(context.paths.m9_t_trials_file)
    config_id, commit = context.config.config_id, context.stamp.git_commit
    missing = []
    for key in keys:
        area = area_of(key.stage, commit)
        found = trial_rows_of(rows, area, config_id, key.seed, key.target_fold)
        if len(found) > 1:
            raise ValueError(f"the training {key} is recorded {len(found)} times")
        if not found:
            missing.append(key)
    return missing


def run_trainings(context: TEvalContext, keys: Sequence[TrialKey], launcher: Launcher) -> None:
    """Start the given trainings one after another, each in its own process.

    Raises:
        RuntimeError: If a training returns without its trial row.
    """
    for number, key in enumerate(keys, start=1):
        LOGGER.info(
            "training %d of %d: %s | seed %d %s",
            number,
            len(keys),
            key.stage,
            key.seed,
            key.target_fold,
        )
        launcher(key)
        if missing_keys(context, [key]):
            raise RuntimeError(f"the training {key} finished without its trial row")


def run_rehearsal(
    context: TEvalContext, launcher: Launcher, metrics: Sequence[str], now: Callable[[], str]
) -> tuple[str, list[Check]]:
    """Run the rehearsal trainings that are missing at HEAD, then compare with the V records.

    Args:
        context: The context of this process.
        launcher: Starts one training and returns when it is finished and recorded.
        metrics: The metrics of the result tables, the main one first.
        now: Gives the current time as ISO text (called when the check row is written).

    Returns:
        ("passed" or "failed", the checks). The row in
        results/m9_t_rehearsal_check.csv exists when this function returns.

    Raises:
        TEvalRefused: If the working tree is not clean.
    """
    require_clean_tree(context, "the rehearsal")
    keys = stage_keys(context, STAGE_REHEARSAL)
    run_trainings(context, missing_keys(context, keys), launcher)
    checks = check_rehearsal(context, metrics)
    return write_check_row(context, checks, now()), checks


def run_formal(
    context: TEvalContext, launcher: Launcher, now: Callable[[], str], script: str
) -> int:
    """Run the formal trainings that are missing, after the gate recorded the read.

    If every training is recorded already, nothing is read and no row is
    appended to the access table.

    Args:
        context: The context of this process.
        launcher: Starts one training and returns when it is finished and recorded.
        now: Gives the current time as ISO text (called when the access row is written).
        script: The script that starts the run (written to the access table).

    Returns:
        How many trainings were started.

    Raises:
        TEvalRefused: If the gate refuses (no passed rehearsal at HEAD, a dirty
            tree, or a read recorded at another commit). Nothing was read then.
    """
    missing = missing_keys(context, stage_keys(context, STAGE_FORMAL))
    if not missing:
        LOGGER.info("every training of the formal run is recorded; nothing to do")
        return 0
    # The gate: checks the rehearsal and appends the access row BEFORE any training starts.
    permit = begin_formal_run(context, now(), script)
    LOGGER.info(
        "T labels: read recorded at %s (commit %s); %d trainings to run",
        permit.recorded_at,
        permit.git_commit[:12],
        len(missing),
    )
    run_trainings(context, missing, launcher)
    return len(missing)
