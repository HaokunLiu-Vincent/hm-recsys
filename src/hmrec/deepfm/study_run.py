"""Run the DeepFM part of the M9 study and record what its rules decided (proposal §16.8).

Every training is ONE process (scripts/m9_deepfm_trial.py), started by the
`launcher` of the context; a training that is already recorded in
results/deepfm_trials.csv (for the configured device) is not started again, so
an interrupted study continues where it stopped. After the trainings of a step,
the step's decision is computed from results/deepfm_metrics.csv
(hmrec.deepfm.study).

Order (run_study; every stage needs the ones before it, and repeating them costs
nothing because their trainings are recorded):

    step0       the default configuration with the noise seeds and every schedule
                length T; T by the epoch rule on the seed mean; the baseline stop
                check; the noise rule decides the seeds of tuning
    tuning      pointwise arm: learning rate (every candidate with every T, its
                own T by the epoch rule), then embedding dimension, then dropout
                (at the T that came with the learning rate)
    pointwise   the selected pointwise configuration with the final seeds
    listwise    the listwise arm: its own learning-rate step, then the final seeds
    ablations   the ablations of the selected pointwise configuration

Nothing here selects on T folds: every training goes through the trial script,
which refuses them. This module imports neither torch nor LightGBM.
"""

import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from hmrec.config import DeepFMConfig, DeepFMStudyConfig, DeepFMStudyStep, SelectionRule
from hmrec.csv_records import append_row
from hmrec.deepfm.records import PooledReference, now, read_csv_rows, read_pooled_reference
from hmrec.deepfm.study import (
    ConfigurationScore,
    ScheduleChoice,
    StepOutcome,
    Variant,
    ablation_variants,
    check_above_baseline,
    choose_schedule,
    decide_step,
    listwise_start,
    listwise_step,
    noise_mark,
    score_configuration,
    step_variants,
    tuning_seeds,
    variant_config_id,
    variant_of,
)
from hmrec.hpo.selection import relative_change
from hmrec.logging_utils import get_logger
from hmrec.paths import ProjectPaths
from hmrec.run_records import RunStamp

LOGGER = get_logger("deepfm.study_run")

AREA_V_PHASE = "v_phase"
STAGES = ("step0", "tuning", "pointwise", "listwise", "ablations")
ARM_POINTWISE = "deepfm_pointwise"
ARM_LISTWISE = "deepfm_listwise"

STUDY_COLUMNS: tuple[str, ...] = (
    "stage",
    "arm",
    "name",
    "parameter",
    "value",
    "config_id",
    "epochs",
    "seeds",
    "pooled",
    "pooled_by_seed",
    "fold_scores",
    "seed_range",
    "reference",
    "relative_change",
    "folds_better",
    "passes",
    "selected",
    "on_grid_edge",
    "note",
    "git_commit",
    "recorded_at",
)


@dataclass(frozen=True)
class TrialRequest:
    """One training the study needs.

    Attributes:
        area: "v_phase".
        target_fold: The V fold.
        seed: The training seed.
        variant: The configuration's seven study settings (its epochs included).
    """

    area: str
    target_fold: str
    seed: int
    variant: Variant


def trial_arguments(request: TrialRequest) -> list[str]:
    """The command-line arguments of scripts/m9_deepfm_trial.py for one training."""
    variant = request.variant
    return [
        "--area", request.area,
        "--target", request.target_fold,
        "--seed", str(request.seed),
        "--epochs", str(variant.epochs),
        "--learning-rate", repr(variant.learning_rate),
        "--embedding-dim", str(variant.embedding_dim),
        "--dropout", repr(variant.dropout),
        "--components", variant.components,
        "--numeric-input", variant.numeric_input,
        "--loss", variant.loss,
    ]  # fmt: skip


@dataclass(frozen=True)
class StudyContext:
    """What the study works with.

    Attributes:
        paths: Project paths.
        config: configs/deepfm.yaml (the default configuration).
        study: configs/hpo/deepfm.yaml (the rules).
        rule: configs/selection_rule.yaml.
        launcher: Starts one training and returns when it has finished and is recorded.
    """

    paths: ProjectPaths
    config: DeepFMConfig
    study: DeepFMStudyConfig
    rule: SelectionRule
    launcher: Callable[[TrialRequest], None]

    @property
    def default(self) -> Variant:
        """The default configuration's study settings."""
        return variant_of(self.config)

    @property
    def metric(self) -> str:
        """The selection metric."""
        return self.study.metric


def subprocess_launcher(trial_script: Path) -> Callable[[TrialRequest], None]:
    """A launcher that runs one training with scripts/m9_deepfm_trial.py in a new process.

    The training's process records its own rows. A non-zero exit (a failed check,
    a diverged training, a memory peak above the limit) raises
    subprocess.CalledProcessError and so stops the study.
    """

    def launch(request: TrialRequest) -> None:
        command = [sys.executable, str(trial_script), *trial_arguments(request)]
        subprocess.run(command, check=True)

    return launch


# ---------------------------------------------------------------- trainings and scores


def recorded_trials(paths: ProjectPaths) -> set[tuple[str, str, int, str, str]]:
    """(area, config_id, seed, fold, device) of every training in results/deepfm_trials.csv."""
    rows = read_csv_rows(paths.deepfm_trials_file)
    return {
        (row["area"], row["config_id"], int(row["seed"]), row["fold"], row["device"])
        for row in rows
    }


def ensure_trials(context: StudyContext, variant: Variant, seeds: Sequence[int]) -> int:
    """Start every training of a configuration that is not recorded yet; returns how many.

    A recorded training counts only if it ran on the configured device: a run on
    another device is not the same experiment (on MPS it does not even repeat).
    """
    config_id = variant_config_id(context.config, variant)
    device = context.config.training.device
    started = 0
    for seed in seeds:
        for fold in context.study.target_folds:
            if (AREA_V_PHASE, config_id, seed, fold, device) in recorded_trials(context.paths):
                continue
            LOGGER.info("training %s seed %d %s (%d epochs)", config_id, seed, fold, variant.epochs)
            context.launcher(TrialRequest(AREA_V_PHASE, fold, seed, variant))
            started += 1
    return started


def score(context: StudyContext, variant: Variant, seeds: Sequence[int]) -> ConfigurationScore:
    """The recorded result of a configuration with the given seeds."""
    rows = read_csv_rows(context.paths.deepfm_metrics_file)
    return score_configuration(
        rows,
        AREA_V_PHASE,
        variant_config_id(context.config, variant),
        variant.epochs,
        seeds,
        context.study.target_folds,
        context.metric,
    )


def run_configuration(
    context: StudyContext, variant: Variant, seeds: Sequence[int]
) -> ConfigurationScore:
    """Train what is missing of a configuration with the given seeds, then score it."""
    ensure_trials(context, variant, seeds)
    return score(context, variant, seeds)


def run_schedule(context: StudyContext, variant: Variant, seeds: Sequence[int]) -> ScheduleChoice:
    """Run the settings of `variant` with every schedule length and apply the epoch rule.

    Returns:
        The choice; T -> result for every T of the grid (variant.epochs is ignored).
    """
    study = context.study
    scores = {
        epochs: run_configuration(context, replace(variant, epochs=epochs), seeds)
        for epochs in study.schedule_epochs
    }
    return choose_schedule(scores, study.epoch_tolerance)


def reference(context: StudyContext) -> PooledReference:
    """The recorded RRF baseline and LightGBM scores, pooled over the study's folds."""
    return read_pooled_reference(context.paths, context.study.target_folds, context.metric)


# ---------------------------------------------------------------- the stages


@dataclass(frozen=True)
class Step0Result:
    """Step 0: the default configuration with the noise seeds and every schedule length.

    Attributes:
        schedule: The epoch rule on the seed means (T -> result with all noise seeds).
        seed_range: (max - min) / min of the seeds' pooled scores at the selected T.
        baseline: The pooled RRF baseline.
        tuning_seeds: The seeds every tuning candidate is run with (noise rule).
    """

    schedule: ScheduleChoice
    seed_range: float
    baseline: float
    tuning_seeds: tuple[int, ...]

    @property
    def multi_seed(self) -> bool:
        """True if the noise rule makes tuning run with all noise seeds."""
        return len(self.tuning_seeds) > 1


def run_step0(context: StudyContext) -> Step0Result:
    """Train the default configuration with every noise seed and T; apply the step-0 rules.

    Raises:
        StudyStop: If the seed-mean pooled score at the selected T is below the
            RRF baseline.
    """
    study = context.study
    schedule = run_schedule(context, context.default, study.noise_seeds)
    baseline = reference(context).rrf_baseline
    check_above_baseline(schedule.selected, baseline)
    seed_range = schedule.selected.seed_range
    return Step0Result(schedule, seed_range, baseline, tuning_seeds(study, seed_range))


def run_learning_rate_step(
    context: StudyContext, step: DeepFMStudyStep, current: Variant, seeds: Sequence[int]
) -> StepOutcome:
    """A learning-rate step: every candidate with every T, compared at its own T."""
    schedules = {
        value: run_schedule(context, variant, seeds)
        for value, variant in step_variants(step, current).items()
    }
    variants = {
        value: replace(variant, epochs=schedules[value].selected_epochs)
        for value, variant in step_variants(step, current).items()
    }
    scores = {value: schedules[value].selected for value in step.candidates}
    return decide_step(step, current, variants, scores, context.rule, schedules)


def run_fixed_schedule_step(
    context: StudyContext, step: DeepFMStudyStep, current: Variant, seeds: Sequence[int]
) -> StepOutcome:
    """A step whose candidates all use the T of the configuration it starts from."""
    variants = step_variants(step, current)
    scores = {
        value: run_configuration(context, variant, seeds) for value, variant in variants.items()
    }
    return decide_step(step, current, variants, scores, context.rule)


def run_tuning(context: StudyContext, step0: Step0Result) -> tuple[StepOutcome, ...]:
    """The tuning steps of the pointwise arm; every step starts from the previous selection."""
    current = context.default
    outcomes = []
    for position, step in enumerate(context.study.steps):
        # The first step is the learning rate (checked by the config): it brings its T.
        runner = run_learning_rate_step if position == 0 else run_fixed_schedule_step
        outcome = runner(context, step, current, step0.tuning_seeds)
        LOGGER.info(
            "step %s: selected %s, T = %d (%s)",
            step.name,
            outcome.selected_value,
            outcome.selected.epochs,
            outcome.decision.reason,
        )
        outcomes.append(outcome)
        current = outcome.selected
    return tuple(outcomes)


def selected_variant(outcomes: Sequence[StepOutcome], default: Variant) -> Variant:
    """The configuration after the last tuning step (the default if there is no step)."""
    return outcomes[-1].selected if outcomes else default


def run_final_seeds(context: StudyContext, variant: Variant) -> ConfigurationScore:
    """A selected configuration with every final seed (the missing ones are trained)."""
    return run_configuration(context, variant, context.study.final_seeds)


def run_listwise(context: StudyContext, pointwise: Variant, seeds: Sequence[int]) -> StepOutcome:
    """The learning-rate step of the listwise arm, starting from the pointwise selection."""
    study = context.study
    outcome = run_learning_rate_step(
        context, listwise_step(study), listwise_start(study, pointwise), seeds
    )
    LOGGER.info(
        "listwise: selected learning rate %s, T = %d (%s)",
        outcome.selected_value,
        outcome.selected.epochs,
        outcome.decision.reason,
    )
    return outcome


def run_ablations(context: StudyContext, selected: Variant) -> dict[str, ConfigurationScore]:
    """Train and score the ablations of the selected pointwise configuration (selection seed)."""
    seeds = [context.study.selection_seed]
    return {
        name: run_configuration(context, variant, seeds)
        for name, variant in ablation_variants(context.study, selected).items()
    }


@dataclass(frozen=True)
class StudyResult:
    """Everything the stages decided, up to the stage the study was run through.

    Attributes:
        step0: Step 0.
        tuning: The pointwise tuning steps (None if the study stopped earlier).
        pointwise_variant: The selected pointwise configuration.
        pointwise: Its result with the final seeds.
        listwise_step: The listwise learning-rate step.
        listwise_variant: The selected listwise configuration.
        listwise: Its result with the final seeds.
        ablations: Name -> result of the pointwise ablations (selection seed).
    """

    step0: Step0Result
    tuning: tuple[StepOutcome, ...] | None = None
    pointwise_variant: Variant | None = None
    pointwise: ConfigurationScore | None = None
    listwise_step: StepOutcome | None = None
    listwise_variant: Variant | None = None
    listwise: ConfigurationScore | None = None
    ablations: dict[str, ConfigurationScore] | None = None


def run_study(context: StudyContext, through: str) -> StudyResult:
    """Run the stages in order up to and including `through` (one of STAGES).

    Raises:
        ValueError: If `through` is not a stage.
        StudyStop: If a stop condition is met.
    """
    if through not in STAGES:
        raise ValueError(f"through must be one of {STAGES}, got {through!r}")
    last = STAGES.index(through)
    result = StudyResult(step0=run_step0(context))
    if last >= STAGES.index("tuning"):
        tuning = run_tuning(context, result.step0)
        pointwise_variant = selected_variant(tuning, context.default)
        result = replace(result, tuning=tuning, pointwise_variant=pointwise_variant)
    if last >= STAGES.index("pointwise") and result.pointwise_variant is not None:
        result = replace(result, pointwise=run_final_seeds(context, result.pointwise_variant))
    if last >= STAGES.index("listwise") and result.pointwise_variant is not None:
        step = run_listwise(context, result.pointwise_variant, result.step0.tuning_seeds)
        result = replace(
            result,
            listwise_step=step,
            listwise_variant=step.selected,
            listwise=run_final_seeds(context, step.selected),
        )
    if last >= STAGES.index("ablations") and result.pointwise_variant is not None:
        result = replace(result, ablations=run_ablations(context, result.pointwise_variant))
    return result


# ---------------------------------------------------------------- the study table


def _row(
    stage: str, arm: str, name: str, result: ConfigurationScore, **extra: str
) -> dict[str, str]:
    """One row of results/deepfm_study.csv for a scored configuration."""
    folds = " ".join(f"{fold}={value!r}" for fold, value in result.fold_scores.items())
    by_seed = " ".join(f"{seed.seed}={seed.pooled!r}" for seed in result.seeds)
    row = dict.fromkeys(STUDY_COLUMNS, "")
    row.update(
        {
            "stage": stage,
            "arm": arm,
            "name": name,
            "config_id": result.config_id,
            "epochs": str(result.epochs),
            "seeds": " ".join(str(seed.seed) for seed in result.seeds),
            "pooled": repr(result.pooled),
            "pooled_by_seed": by_seed,
            "fold_scores": folds,
            "seed_range": repr(result.seed_range) if len(result.seeds) > 1 else "",
        }
    )
    row.update(extra)
    return row


def step0_rows(result: Step0Result) -> list[dict[str, str]]:
    """The rows of step 0: one per schedule length."""
    schedule = result.schedule
    note = (
        f"seed range at the selected T {result.seed_range!r}; RRF baseline {result.baseline!r}; "
        f"tuning seeds {list(result.tuning_seeds)}"
    )
    return [
        _row(
            "step0",
            ARM_POINTWISE,
            f"T={epochs}",
            score_,
            selected=str(epochs == schedule.selected_epochs),
            on_grid_edge=str(schedule.on_grid_edge) if epochs == schedule.selected_epochs else "",
            note=note if epochs == schedule.selected_epochs else "",
        )
        for epochs, score_ in schedule.scores.items()
    ]


def step_rows(outcome: StepOutcome, arm: str) -> list[dict[str, str]]:
    """The rows of one tuning step: the schedule rows of every candidate, then the candidates."""
    step = outcome.step
    stage = f"tune:{step.name}"
    rows = []
    for value, schedule in outcome.schedules.items():
        for epochs, score_ in schedule.scores.items():
            rows.append(
                _row(
                    f"{stage}:schedule",
                    arm,
                    f"{step.parameter}={value!r} T={epochs}",
                    score_,
                    parameter=step.parameter,
                    value=repr(value),
                    selected=str(epochs == schedule.selected_epochs),
                    on_grid_edge=(
                        str(schedule.on_grid_edge) if epochs == schedule.selected_epochs else ""
                    ),
                )
            )
    start = outcome.scores[step.candidates[0]]
    checks = {check.trial_id: check for check in outcome.decision.checks}
    for value in step.candidates:
        result = outcome.scores[value]
        check = checks.get(result.config_id)
        is_selected = value == outcome.selected_value
        rows.append(
            _row(
                stage,
                arm,
                f"{step.parameter}={value!r}",
                result,
                parameter=step.parameter,
                value=repr(value),
                reference=start.config_id,
                relative_change=repr(relative_change(start.pooled, result.pooled)),
                folds_better="" if check is None else str(check.folds_better),
                passes="" if check is None else str(check.passes),
                selected=str(is_selected),
                on_grid_edge=str(outcome.on_grid_edge) if is_selected else "",
                note=outcome.decision.reason if is_selected else "",
            )
        )
    return rows


def ablation_rows(
    ablations: dict[str, ConfigurationScore],
    full: ConfigurationScore,
    seed_range: float,
    lightgbm: float,
) -> list[dict[str, str]]:
    """The rows of the ablations, compared with the full model of the same seed.

    Args:
        ablations: Name -> result (selection seed).
        full: The selected pointwise configuration with the SAME single seed.
        seed_range: The default configuration's 3-seed range (the noise mark).
        lightgbm: The recorded pooled score of the locked LightGBM ranker.
    """
    return [
        _row(
            "ablation",
            ARM_POINTWISE,
            name,
            result,
            reference=full.config_id,
            relative_change=repr(relative_change(full.pooled, result.pooled)),
            note="; ".join(
                part
                for part in (
                    noise_mark(full.pooled, result.pooled, seed_range),
                    f"relative to LightGBM {relative_change(lightgbm, result.pooled)!r}",
                )
                if part
            ),
        )
        for name, result in ablations.items()
    ]


def study_rows(result: StudyResult, selection_seed: int, lightgbm: float) -> list[dict[str, str]]:
    """All rows of results/deepfm_study.csv for a study that ran through its last stage.

    Raises:
        ValueError: If the study did not run through "ablations".
    """
    if (
        result.tuning is None
        or result.pointwise is None
        or result.listwise_step is None
        or result.listwise is None
        or result.ablations is None
    ):
        raise ValueError("the study table records the finished study: run it through ablations")
    rows = step0_rows(result.step0)
    for outcome in result.tuning:
        rows += step_rows(outcome, ARM_POINTWISE)
    rows.append(_row("final", ARM_POINTWISE, "selected", result.pointwise, selected="True"))
    rows += step_rows(result.listwise_step, ARM_LISTWISE)
    rows.append(_row("final", ARM_LISTWISE, "selected", result.listwise, selected="True"))
    full = result.pointwise.only_seed(selection_seed)
    rows += ablation_rows(result.ablations, full, result.step0.seed_range, lightgbm)
    return rows


def write_study_table(paths: ProjectPaths, rows: Sequence[dict[str, str]], stamp: RunStamp) -> None:
    """Write results/deepfm_study.csv once (it is the record of the finished study).

    Raises:
        FileExistsError: If the table exists.
    """
    if paths.deepfm_study_file.exists():
        raise FileExistsError(f"{paths.deepfm_study_file} already exists; it is never replaced")
    recorded_at = now()
    for row in rows:
        row["git_commit"], row["recorded_at"] = stamp.git_commit, recorded_at
        append_row(paths.deepfm_study_file, STUDY_COLUMNS, row)
