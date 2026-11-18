"""The driver of the final evaluation (M8): which steps run, in which order, each in a process.

Every step runs in its own process (scripts/m8_step.py), so its max RSS is its
own true memory peak and a model never shares a process with another model. A
step that is already recorded is skipped, so an interrupted run can be continued
(with the same code: hmrec.final.gate).

    rehearsal   on the configured V fold: recall monitoring, RRF baseline, then for
                the full system and every secondary experiment a Dataset and a
                trial; then the comparison with M7 (hmrec.final.rehearsal), whose
                outcome is recorded. A failed comparison raises.
    formal      FIRST the gate (a passed rehearsal with this code; the start is
                recorded before any T label is read). Then, on the T folds:
                recall monitoring, RRF baseline, the full system (new samples,
                Datasets, trials), the secondary experiments, and the verdict on
                G1-G3.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from hmrec.csv_records import read_rows
from hmrec.eval.monitor import MONITOR_COLUMNS
from hmrec.final.context import MODE_FORMAL, MODE_REHEARSAL, MODES, FinalContext, area_of
from hmrec.final.gate import PASSED, FinalRunRefused, begin_formal_run, require_clean_tree
from hmrec.final.goals import GoalVerdict
from hmrec.final.monitor import recall_file
from hmrec.final.rehearsal import model_experiments, record_rehearsal_check
from hmrec.final.report import final_verdicts, record_verdicts, summary_lines
from hmrec.final.trial import experiment_config_id, new_sample_folds, sample_file
from hmrec.hpo.v_records import (
    RRF_BASELINE,
    find_trial,
    read_metric_rows,
    read_trial_rows,
    results_files,
)
from hmrec.logging_utils import get_logger
from hmrec.ranker.ablation import EXPERIMENTS, FULL
from hmrec.ranker.dataset import dataset_is_built
from hmrec.ranker.sampling import record_file

LOGGER = get_logger("final.run")

WORKER_SCRIPT = "m8_step.py"
STEP_MONITOR = "monitor"
STEP_BASELINE = "baseline"
STEP_SAMPLE = "sample"
STEP_DATASET = "dataset"
STEP_TRIAL = "trial"
STEPS = (STEP_MONITOR, STEP_BASELINE, STEP_SAMPLE, STEP_DATASET, STEP_TRIAL)

# Runs a script in a new process and waits (hmrec.pipeline.run_script).
Launcher = Callable[[Path, Sequence[str]], float]


@dataclass(frozen=True)
class StepRequest:
    """One step for one fold.

    Attributes:
        step: "monitor", "baseline", "sample", "dataset" or "trial".
        mode: "rehearsal" or "formal".
        fold: The target fold; for "sample" the training fold to sample.
        experiment: "full" or "minus_r6" (ignored by "monitor" and "baseline").
    """

    step: str
    mode: str
    fold: str
    experiment: str = FULL


def check_request(request: StepRequest) -> None:
    """Fail on an unknown step, mode or experiment, or a sampling step outside the formal run."""
    if request.step not in STEPS or request.mode not in MODES:
        raise ValueError(f"unknown step or mode in {request}")
    if request.experiment not in EXPERIMENTS:
        raise ValueError(f"experiment must be one of {EXPERIMENTS}, got {request.experiment!r}")
    if request.step == STEP_SAMPLE and request.mode != MODE_FORMAL:
        # The rehearsal trains on M7's existing sample files only.
        raise ValueError("samples are drawn only in the formal run")


def step_arguments(request: StepRequest) -> list[str]:
    """Command-line arguments of scripts/m8_step.py for one step."""
    return [
        "--step", request.step,
        "--mode", request.mode,
        "--fold", request.fold,
        "--experiment", request.experiment,
    ]  # fmt: skip


def step_is_done(context: FinalContext, request: StepRequest) -> bool:
    """True if the step's output is already recorded (then it is not run again)."""
    check_request(request)
    paths, area = context.paths, area_of(request.mode)
    trial_file, metrics_file = results_files(paths, area)
    if request.step == STEP_MONITOR:
        rows = read_rows(recall_file(context, request.mode), MONITOR_COLUMNS)
        return any(row["fold"] == request.fold for row in rows)
    if request.step == STEP_BASELINE:
        rows = read_metric_rows(metrics_file, context.config.metrics)
        return any(r.experiment == RRF_BASELINE and r.fold == request.fold for r in rows)
    if request.step == STEP_SAMPLE:
        return record_file(sample_file(context, request.experiment, request.fold)).exists()
    ratio = context.locked.tuned.negatives_per_positive
    if request.step == STEP_DATASET:
        directory = paths.final_dataset_dir(area, request.experiment, request.fold, ratio)
        return dataset_is_built(directory)
    config_id = experiment_config_id(context, request.experiment)
    rows = read_trial_rows(trial_file)
    return find_trial(rows, request.experiment, config_id, request.fold) is not None


def ensure_step(
    context: FinalContext, launch: Launcher, scripts_dir: Path, request: StepRequest
) -> None:
    """Run a step in its own process unless it is already recorded."""
    if step_is_done(context, request):
        LOGGER.info("already recorded, skipped: %s", " ".join(step_arguments(request)))
        return
    launch(scripts_dir / WORKER_SCRIPT, step_arguments(request))


def run_rehearsal(
    context: FinalContext, launch: Launcher, scripts_dir: Path, now: str
) -> Mapping[str, str]:
    """Run the rehearsal steps and compare the result with M7.

    Args:
        context: The context of the driver.
        launch: How a step's process is started.
        scripts_dir: The folder of scripts/m8_step.py.
        now: The current time as ISO text (written to the check row).

    Returns:
        The recorded row of results/m8_rehearsal_check.csv (status "passed").

    Raises:
        FinalRunRefused: If the working tree is dirty or a comparison failed
            (the failed row is recorded first).
    """
    require_clean_tree(context, "the rehearsal")
    fold = context.config.rehearsal_target_fold
    ensure_step(context, launch, scripts_dir, StepRequest(STEP_MONITOR, MODE_REHEARSAL, fold))
    ensure_step(context, launch, scripts_dir, StepRequest(STEP_BASELINE, MODE_REHEARSAL, fold))
    for experiment in model_experiments(context):
        for step in (STEP_DATASET, STEP_TRIAL):
            request = StepRequest(step, MODE_REHEARSAL, fold, experiment)
            ensure_step(context, launch, scripts_dir, request)
    row = record_rehearsal_check(context, now)
    if row["status"] != PASSED:
        raise FinalRunRefused(
            f"the rehearsal did NOT reproduce M7 on {fold}: {row['failed_checks']}. "
            f"Details are in {context.paths.m8_rehearsal_check_file}. The formal run cannot start."
        )
    LOGGER.info("rehearsal PASSED: %s checks on %s are exactly the same", row["n_checks"], fold)
    return row


def run_formal(
    context: FinalContext, launch: Launcher, scripts_dir: Path, now: str
) -> list[GoalVerdict]:
    """Run the formal evaluation on the T folds and judge G1-G3.

    The gate is passed FIRST (hmrec.final.gate.begin_formal_run): a passed
    rehearsal with this code, and the start of the run recorded before any step
    runs. Every step's process passes the gate again.

    Returns:
        The verdicts on G1, G2 and G3 (also written to results/t_phase_goals.csv).

    Raises:
        FinalRunRefused: If the gate refuses; then no step has run.
    """
    permit = begin_formal_run(context, now)
    LOGGER.info(
        "formal run: first read of T labels recorded at %s (commit %s; rehearsal of %s)",
        permit.first_label_read_at,
        permit.git_commit[:12],
        permit.rehearsal_checked_at,
    )
    targets = context.config.target_folds
    for step in (STEP_MONITOR, STEP_BASELINE):
        for fold in targets:
            ensure_step(context, launch, scripts_dir, StepRequest(step, MODE_FORMAL, fold))
    # The main result first (full), then the pre-registered secondary experiments.
    for experiment in model_experiments(context):
        for fold in new_sample_folds(context):
            request = StepRequest(STEP_SAMPLE, MODE_FORMAL, fold, experiment)
            ensure_step(context, launch, scripts_dir, request)
        for fold in targets:
            for step in (STEP_DATASET, STEP_TRIAL):
                request = StepRequest(step, MODE_FORMAL, fold, experiment)
                ensure_step(context, launch, scripts_dir, request)
    verdicts = final_verdicts(context)
    record_verdicts(context, verdicts, now)
    for line in summary_lines(context, verdicts):
        LOGGER.info(line)
    return verdicts
