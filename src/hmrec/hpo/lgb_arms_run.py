"""Run the trainings of the two LightGBM arms of M9 in order (proposal §16.8).

Every training is ONE process (scripts/m9_lgb_trial.py), so its process max RSS
is its own true peak; a training already recorded in results/m9_lgb_trials.csv
is not started again, so an interrupted run continues where it stopped.

Order:
    1. lambdarank: every seed after the first, on every fold, the locked number
       of trees. (The first seed is the M7 record and is not trained.)
    2. binary, first seed: every fold to `binary_num_trees` trees, evaluated at
       every checkpoint.
    3. k* of the binary arm by the number-of-trees rule on that seed
       (hmrec.hpo.lgb_arm_records.choose_binary_num_trees).
    4. binary, the other seeds: every fold to k* trees, evaluated at k*.

This module starts processes and reads result tables; it imports neither
LightGBM nor torch.
"""

import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from hmrec.hpo.lgb_arm_records import (
    ARM_BINARY,
    ARM_LAMBDARANK,
    choose_binary_num_trees,
    is_recorded,
    read_metric_rows,
    read_trial_rows,
)
from hmrec.hpo.selection import NumTreesChoice
from hmrec.logging_utils import get_logger
from hmrec.paths import ProjectPaths

LOGGER = get_logger("hpo.lgb_arms_run")


@dataclass(frozen=True)
class ArmRequest:
    """One training the comparison needs.

    Attributes:
        arm: "lambdarank" or "binary".
        seed: The LightGBM seed.
        target_fold: The V fold.
        num_trees: Trees to train.
        evaluate_all: True = evaluate every checkpoint; False = only the last one.
    """

    arm: str
    seed: int
    target_fold: str
    num_trees: int
    evaluate_all: bool


def request_arguments(request: ArmRequest) -> list[str]:
    """The command-line arguments of scripts/m9_lgb_trial.py for one training."""
    arguments = [
        "--arm", request.arm,
        "--seed", str(request.seed),
        "--target", request.target_fold,
        "--num-trees", str(request.num_trees),
    ]  # fmt: skip
    if request.evaluate_all:
        arguments.append("--evaluate-all")
    return arguments


def subprocess_launcher(trial_script: Path) -> Callable[[ArmRequest], None]:
    """A launcher that runs one training with scripts/m9_lgb_trial.py in a new process.

    A non-zero exit (a failed check, a memory peak above the limit) raises
    subprocess.CalledProcessError and so stops the run.
    """

    def launch(request: ArmRequest) -> None:
        command = [sys.executable, str(trial_script), *request_arguments(request)]
        subprocess.run(command, check=True)

    return launch


@dataclass(frozen=True)
class ArmsPlan:
    """What the run of the two arms needs to know.

    Attributes:
        paths: Project paths (the two result tables).
        folds: The target folds (V1 ... V4).
        seeds: The seeds of both arms; the first is the locked seed.
        metrics: The metrics of the metric table, the selection metric first.
        locked_num_trees: Trees of the lambdarank arm (the locked number).
        checkpoint_every: Trees between two checkpoints (the locked value).
        binary_num_trees: Trees the binary arm's first seed is trained to.
        tolerance: Relative tolerance of the number-of-trees rule.
        launcher: Starts one training and returns when it is finished and recorded.
    """

    paths: ProjectPaths
    folds: tuple[str, ...]
    seeds: tuple[int, ...]
    metrics: tuple[str, ...]
    locked_num_trees: int
    checkpoint_every: int
    binary_num_trees: int
    tolerance: float
    launcher: Callable[[ArmRequest], None]

    @property
    def binary_checkpoints(self) -> tuple[int, ...]:
        """The checkpoints of the binary arm's first seed: every, 2 * every, ..., the last."""
        return tuple(range(self.checkpoint_every, self.binary_num_trees + 1, self.checkpoint_every))


def ensure_trainings(plan: ArmsPlan, requests: Sequence[ArmRequest]) -> int:
    """Start every requested training that is not recorded yet; returns how many."""
    started = 0
    for request in requests:
        recorded = read_trial_rows(plan.paths.m9_lgb_trials_file)
        if is_recorded(recorded, request.arm, request.seed, request.target_fold):
            continue
        LOGGER.info(
            "training %s seed %d %s (%d trees)",
            request.arm,
            request.seed,
            request.target_fold,
            request.num_trees,
        )
        plan.launcher(request)
        started += 1
    return started


def binary_choice(plan: ArmsPlan) -> NumTreesChoice:
    """k* of the binary arm, from the recorded rows of its first seed."""
    rows = read_metric_rows(plan.paths.m9_lgb_metrics_file, plan.metrics)
    return choose_binary_num_trees(
        rows, plan.seeds[0], plan.folds, plan.metrics[0], plan.tolerance, plan.binary_checkpoints
    )


def run_arms(plan: ArmsPlan) -> NumTreesChoice:
    """Run every training of the two arms that is missing, in the order of the module docstring.

    Returns:
        The number-of-trees choice of the binary arm.
    """
    first_seed, *other_seeds = plan.seeds
    ensure_trainings(
        plan,
        [
            ArmRequest(ARM_LAMBDARANK, seed, fold, plan.locked_num_trees, False)
            for seed in other_seeds
            for fold in plan.folds
        ],
    )
    ensure_trainings(
        plan,
        [
            ArmRequest(ARM_BINARY, first_seed, fold, plan.binary_num_trees, True)
            for fold in plan.folds
        ],
    )
    choice = binary_choice(plan)
    LOGGER.info(
        "binary arm: k* = %d trees (pooled %.6f; best %.6f at %d trees)",
        choice.num_trees,
        choice.score,
        choice.best_score,
        choice.best_num_trees,
    )
    ensure_trainings(
        plan,
        [
            ArmRequest(ARM_BINARY, seed, fold, choice.num_trees, False)
            for seed in other_seeds
            for fold in plan.folds
        ],
    )
    return choice
