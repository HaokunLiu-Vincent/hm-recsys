"""Run the DeepFM study step by step and record what its rules decided (proposal §16.5).

Every training is ONE process (scripts/m9_deepfm_trial.py), started by the
`launcher` of the context; a training that is already recorded in
results/deepfm_trials.csv is not started again, so an interrupted study
continues where it stopped. After the trainings of a step, the step's decision
is computed from results/deepfm_metrics.csv (hmrec.deepfm.study).

Order (each function runs the ones before it again; that costs nothing, because
their trainings are recorded):

    run_step0       the default configuration with the noise seeds; the two stop
                    checks (below the RRF baseline; seed range above the limit)
    run_tuning      learning rate, then embedding dimension, then dropout
    run_final       the epoch rule of the selected configuration; if it selects
                    the last epoch, the configuration is trained once more for
                    `extension_epochs` epochs (area "v_phase_extended")
    run_ablations   the ablations of the selected configuration
    run_seeds       the other seeds of the selected configuration

Nothing here selects on T folds: every training goes through the trial script,
which refuses them. This module imports neither torch nor LightGBM.
"""

import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from hmrec.config import DeepFMConfig, DeepFMStudyConfig, SelectionRule
from hmrec.csv_records import append_row
from hmrec.deepfm.records import PooledReference, now, read_csv_rows, read_pooled_reference
from hmrec.deepfm.study import (
    ConfigurationScore,
    StepOutcome,
    Variant,
    ablation_variants,
    check_above_baseline,
    check_noise,
    decide_step,
    score_configuration,
    step_variants,
    variant_config_id,
    variant_of,
)
from hmrec.hpo.selection import relative_change
from hmrec.logging_utils import get_logger
from hmrec.paths import ProjectPaths
from hmrec.run_records import RunStamp

LOGGER = get_logger("deepfm.study_run")

AREA_V_PHASE = "v_phase"
AREA_V_EXTENDED = "v_phase_extended"

STUDY_COLUMNS: tuple[str, ...] = (
    "stage",
    "name",
    "parameter",
    "value",
    "config_id",
    "seed",
    "area",
    "epochs_trained",
    "selected_epoch",
    "best_epoch",
    "pooled",
    "fold_scores",
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
        area: "v_phase" or "v_phase_extended".
        target_fold: The V fold.
        seed: The training seed.
        epochs: Epochs to train.
        variant: The configuration's six study settings.
    """

    area: str
    target_fold: str
    seed: int
    epochs: int
    variant: Variant


def trial_arguments(request: TrialRequest) -> list[str]:
    """The command-line arguments of scripts/m9_deepfm_trial.py for one training."""
    variant = request.variant
    return [
        "--area", request.area,
        "--target", request.target_fold,
        "--seed", str(request.seed),
        "--epochs", str(request.epochs),
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


def recorded_trials(paths: ProjectPaths) -> set[tuple[str, str, int, str]]:
    """(area, config_id, seed, fold) of every training recorded in results/deepfm_trials.csv."""
    rows = read_csv_rows(paths.deepfm_trials_file)
    return {(row["area"], row["config_id"], int(row["seed"]), row["fold"]) for row in rows}


def ensure_trials(
    context: StudyContext, variant: Variant, seeds: Sequence[int], area: str, epochs: int
) -> int:
    """Start every training of a configuration that is not recorded yet; returns how many."""
    config_id = variant_config_id(context.config, variant)
    started = 0
    for seed in seeds:
        for fold in context.study.target_folds:
            if (area, config_id, seed, fold) in recorded_trials(context.paths):
                continue
            LOGGER.info(
                "training %s seed %d %s (%s, %d epochs)", config_id, seed, fold, area, epochs
            )
            context.launcher(TrialRequest(area, fold, seed, epochs, variant))
            started += 1
    return started


def score(context: StudyContext, variant: Variant, seed: int, area: str) -> ConfigurationScore:
    """The recorded result of a configuration and seed, after the epoch rule."""
    rows = read_csv_rows(context.paths.deepfm_metrics_file)
    return score_configuration(
        rows,
        area,
        variant_config_id(context.config, variant),
        seed,
        context.study.target_folds,
        context.metric,
        context.study.epoch_tolerance,
    )


def reference(context: StudyContext) -> PooledReference:
    """The recorded RRF baseline and LightGBM scores, pooled over the study's folds."""
    return read_pooled_reference(context.paths, context.study.target_folds, context.metric)


# ---------------------------------------------------------------- the stages


@dataclass(frozen=True)
class Step0Result:
    """Step 0: the default configuration with the noise seeds.

    Attributes:
        scores: One score per seed, in the order of the config.
        seed_range: (max - min) / min of their pooled scores.
        baseline: The pooled RRF baseline.
    """

    scores: tuple[ConfigurationScore, ...]
    seed_range: float
    baseline: float


def run_step0(context: StudyContext) -> Step0Result:
    """Train the default configuration with every noise seed and apply the two stop checks.

    Raises:
        StudyStop: If the selection seed's pooled score is below the RRF baseline,
            or the range over the seeds is above the limit.
    """
    study, default = context.study, context.default
    epochs = context.config.training.epochs
    ensure_trials(context, default, study.noise_seeds, AREA_V_PHASE, epochs)
    scores = tuple(score(context, default, seed, AREA_V_PHASE) for seed in study.noise_seeds)
    baseline = reference(context).rrf_baseline
    selection = scores[study.noise_seeds.index(study.selection_seed)]
    check_above_baseline(selection, baseline)
    return Step0Result(scores, check_noise(scores, study.max_seed_range), baseline)


def run_tuning(context: StudyContext) -> tuple[StepOutcome, ...]:
    """The tuning steps in order; every step starts from the previous step's selection."""
    study = context.study
    epochs = context.config.training.epochs
    current = context.default
    outcomes = []
    for step in study.steps:
        variants = step_variants(step, current)
        for variant in variants.values():
            ensure_trials(context, variant, [study.selection_seed], AREA_V_PHASE, epochs)
        scores = {
            value: score(context, variant, study.selection_seed, AREA_V_PHASE)
            for value, variant in variants.items()
        }
        outcome = decide_step(step, current, scores, context.rule)
        LOGGER.info(
            "step %s: selected %s (%s)", step.name, outcome.selected_value, outcome.decision.reason
        )
        outcomes.append(outcome)
        current = outcome.selected
    return tuple(outcomes)


@dataclass(frozen=True)
class FinalChoice:
    """The selected configuration and the number of epochs its results use.

    Attributes:
        variant: The selected configuration.
        area: Where its final rows are: "v_phase", or "v_phase_extended" after the extension.
        epochs: Epochs of its final trainings (the default number, or the extension's).
        score: Its result with the selection seed in that area.
        default_epochs_score: Its result with the default number of epochs.
        extended: True if the epoch-cap extension was triggered.
    """

    variant: Variant
    area: str
    epochs: int
    score: ConfigurationScore
    default_epochs_score: ConfigurationScore
    extended: bool


def run_final(context: StudyContext, selected: Variant) -> FinalChoice:
    """Apply the epoch rule to the selected configuration; extend its epochs if it hits the cap.

    The extension (proposal §16.5 item 7) is done once: if the rule selects the
    LAST epoch of the default number of epochs, the configuration is trained for
    `extension_epochs` epochs and the rule is applied again to the longer curve.
    """
    study = context.study
    seed = study.selection_seed
    first = score(context, selected, seed, AREA_V_PHASE)
    if not first.uses_last_epoch:
        return FinalChoice(selected, AREA_V_PHASE, first.epochs_trained, first, first, False)
    ensure_trials(context, selected, [seed], AREA_V_EXTENDED, study.extension_epochs)
    extended = score(context, selected, seed, AREA_V_EXTENDED)
    return FinalChoice(selected, AREA_V_EXTENDED, study.extension_epochs, extended, first, True)


def run_ablations(context: StudyContext, final: FinalChoice) -> dict[str, ConfigurationScore]:
    """Train and score the ablations of the selected configuration (selection seed only)."""
    seed = context.study.selection_seed
    results = {}
    for name, variant in ablation_variants(context.study, final.variant).items():
        ensure_trials(context, variant, [seed], final.area, final.epochs)
        results[name] = score(context, variant, seed, final.area)
    return results


def run_seeds(context: StudyContext, final: FinalChoice) -> tuple[ConfigurationScore, ...]:
    """The selected configuration with every final seed (the missing ones are trained)."""
    seeds = context.study.final_seeds
    ensure_trials(context, final.variant, seeds, final.area, final.epochs)
    return tuple(score(context, final.variant, seed, final.area) for seed in seeds)


# ---------------------------------------------------------------- the study table


def _row(stage: str, name: str, result: ConfigurationScore, **extra: str) -> dict[str, str]:
    """One row of results/deepfm_study.csv for a scored configuration."""
    folds = " ".join(f"{fold}={value!r}" for fold, value in result.fold_scores.items())
    row = dict.fromkeys(STUDY_COLUMNS, "")
    row.update(
        {
            "stage": stage,
            "name": name,
            "config_id": result.config_id,
            "seed": str(result.seed),
            "area": result.area,
            "epochs_trained": str(result.epochs_trained),
            "selected_epoch": str(result.selected_epoch),
            "best_epoch": str(result.best_epoch),
            "pooled": repr(result.pooled),
            "fold_scores": folds,
        }
    )
    row.update(extra)
    return row


def step0_rows(result: Step0Result) -> list[dict[str, str]]:
    """The rows of step 0: one per seed."""
    note = f"seed range {result.seed_range!r}; RRF baseline {result.baseline!r}"
    return [_row("step0", f"seed {score.seed}", score, note=note) for score in result.scores]


def tuning_rows(outcomes: Sequence[StepOutcome]) -> list[dict[str, str]]:
    """The rows of the tuning steps: one per candidate, the step's start first."""
    rows = []
    for outcome in outcomes:
        step = outcome.step
        start = outcome.scores[step.candidates[0]]
        checks = {check.trial_id: check for check in outcome.decision.checks}
        for value in step.candidates:
            result = outcome.scores[value]
            check = checks.get(result.config_id)
            is_selected = value == outcome.selected_value
            rows.append(
                _row(
                    f"tune:{step.name}",
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


def final_rows(
    final: FinalChoice,
    ablations: dict[str, ConfigurationScore],
    seeds: Sequence[ConfigurationScore],
    lightgbm: float,
) -> list[dict[str, str]]:
    """The rows of the selected configuration, its ablations and its seeds."""
    note = f"extension triggered: {final.extended}; LightGBM pooled {lightgbm!r}"
    rows = [_row("final", "selected, default epochs", final.default_epochs_score, note=note)]
    rows.append(_row("final", "selected", final.score, selected="True", note=note))
    for name, result in ablations.items():
        rows.append(
            _row(
                "ablation",
                name,
                result,
                reference=final.score.config_id,
                relative_change=repr(relative_change(final.score.pooled, result.pooled)),
                note=f"relative to LightGBM {relative_change(lightgbm, result.pooled)!r}",
            )
        )
    rows += [_row("seeds", f"seed {result.seed}", result) for result in seeds]
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


def selected_variant(outcomes: Sequence[StepOutcome], default: Variant) -> Variant:
    """The configuration after the last tuning step (the default if there is no step)."""
    return outcomes[-1].selected if outcomes else default
