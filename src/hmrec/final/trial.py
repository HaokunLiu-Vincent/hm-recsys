"""The steps of the final evaluation that train and score (M8): sampling, Dataset, trial, baseline.

Every step takes a mode:

    rehearsal   the configured V fold, with the M7 window (F and V folds); labels
                through load_rehearsal_evaluator. No T fold can be reached.
    formal      a T fold, with the window of hmrec.ranker.final_window (V and T
                folds); labels only with a FormalRunPermit (hmrec.final.gate).

Both modes then run the SAME functions of hmrec.hpo.v_trial that M7 ran
(plan_dataset_for_folds, train_planned_model, score_and_save,
rrf_baseline_metric_row) with the locked configuration, which is why the
rehearsal can prove that this code path reproduces the M7 numbers.

Samples (proposal §9.1):
    - The folds that are in the rehearsal's window AND train a T model (V1, V2, V3)
      already have sample files from M7. They are REUSED and never sampled again:
      the formal run checks each file's sha256 against the one the passed
      rehearsal trained on (results/m8_rehearsal_check.csv).
    - The other training folds (V4 and the first three T folds) are sampled now,
      with the method, seed and ratio of the Ranker lock.
"""

from pathlib import Path

from hmrec.data.folds import Fold
from hmrec.eval.final_eval import (
    FormalRunPermit,
    final_training_label_phases,
    load_final_evaluator,
    load_rehearsal_evaluator,
)
from hmrec.eval.report import FoldEvaluator
from hmrec.final.context import MODE_FORMAL, MODE_REHEARSAL, FinalContext, area_of
from hmrec.final.gate import FinalRunRefused, passed_rehearsal, rehearsal_sample_sha256
from hmrec.hpo.v_records import MetricRow
from hmrec.hpo.v_trial import (
    DatasetPlan,
    TrialOutcome,
    TrialSpec,
    build_and_save_dataset,
    check_trial_settings,
    plan_dataset_for_folds,
    rrf_baseline_metric_row,
    score_and_save,
    train_planned_model,
    trial_output_files,
)
from hmrec.locks import file_sha256
from hmrec.ranker.ablation import FULL, build_experiment, wide_table_columns
from hmrec.ranker.dataset import DatasetRecord
from hmrec.ranker.final_window import (
    final_sampling_folds,
    final_training_window,
    rehearsal_training_window,
)
from hmrec.ranker.identity import model_config, ranker_config_id
from hmrec.ranker.sampling import SampleJob, SampleRecord, sample_fold


def training_window(context: FinalContext, mode: str, target_fold: str) -> tuple[Fold, ...]:
    """The training folds of the model that predicts `target_fold`, with the mode's fold check.

    Raises:
        ValueError: If the mode may not predict this fold (rehearsal: only the
            configured V fold; formal: only T folds) or train on one of its folds.
    """
    weeks = context.locked.config.window_weeks
    if mode == MODE_REHEARSAL:
        rehearsal_fold = context.config.rehearsal_target_fold
        return rehearsal_training_window(context.manifest, target_fold, rehearsal_fold, weeks)
    if mode == MODE_FORMAL:
        return final_training_window(context.manifest, target_fold, weeks)
    raise ValueError(f"unknown mode {mode!r}")


def experiment_config_id(context: FinalContext, experiment: str) -> str:
    """The configuration id of an experiment's models with the locked settings.

    For "full" this must be the id the lock records; for "minus_r6" it is the id
    M7's ablation trials carry (only the experiment name differs).

    Raises:
        ValueError: If the full experiment does not give the locked config_id.
    """
    locked = context.locked
    settings = model_config(locked.config, locked.tuned, experiment, locked.histogram_mode)
    config_id = ranker_config_id(settings)
    if experiment == FULL and config_id != locked.config_id:
        raise ValueError(
            f"the locked settings give config_id {config_id}, the lock records {locked.config_id}"
        )
    return config_id


def trial_spec(context: FinalContext, mode: str, experiment: str, target_fold: str) -> TrialSpec:
    """The trial of one target fold: the locked tuned values, trees and histogram mode."""
    locked = context.locked
    return TrialSpec(
        area=area_of(mode),
        experiment=experiment,
        step=mode,
        target_fold=target_fold,
        tuned=locked.tuned,
        num_trees=locked.num_trees,
        histogram_mode=locked.histogram_mode,
        evaluate=True,
    )


def dataset_plan(
    context: FinalContext, mode: str, experiment: str, target_fold: str
) -> DatasetPlan:
    """The Dataset of one target fold: its window's sample files, in the mode's own folder."""
    folds = training_window(context, mode, target_fold)
    ratio = context.locked.tuned.negatives_per_positive
    directory = context.paths.final_dataset_dir(area_of(mode), experiment, target_fold, ratio)
    return plan_dataset_for_folds(
        context.inputs,
        experiment,
        target_fold,
        folds,
        ratio,
        context.stamp.git_commit,
        directory,
    )


# ---------------------------------------------------------------- samples


def reused_sample_folds(context: FinalContext) -> tuple[str, ...]:
    """The training folds of the formal run whose M7 sample files are reused (V1, V2, V3).

    These are the folds that are in the rehearsal's window and also train a T
    model. The rehearsal trained on their sample files, so their sha256 is on
    record and the formal run can check that it reads the same files.
    """
    config, weeks = context.config, context.locked.config.window_weeks
    rehearsal_fold = config.rehearsal_target_fold
    rehearsal_window = rehearsal_training_window(
        context.manifest, rehearsal_fold, rehearsal_fold, weeks
    )
    rehearsed = {fold.name for fold in rehearsal_window}
    needed = final_sampling_folds(context.manifest, config.target_folds, weeks)
    return tuple(fold.name for fold in needed if fold.name in rehearsed)


def new_sample_folds(context: FinalContext) -> tuple[str, ...]:
    """The training folds the formal run samples itself (V4 and the first three T folds)."""
    weeks = context.locked.config.window_weeks
    needed = final_sampling_folds(context.manifest, context.config.target_folds, weeks)
    reused = set(reused_sample_folds(context))
    return tuple(fold.name for fold in needed if fold.name not in reused)


def sample_file(context: FinalContext, experiment: str, fold_name: str) -> Path:
    """The sample file of a training fold at the locked ratio."""
    ratio = context.locked.tuned.negatives_per_positive
    return context.paths.ranker_sample_file(experiment, fold_name, ratio)


def check_reused_samples(context: FinalContext, experiment: str, fold_names: list[str]) -> None:
    """Fail unless the reused sample files are the ones the passed rehearsal trained on.

    Args:
        context: The context of this process.
        experiment: "full" or "minus_r6".
        fold_names: The reused folds to check (a subset of reused_sample_folds).

    Raises:
        FinalRunRefused: If no rehearsal passed, a fold has no recorded sha256, or
            a file's sha256 differs from the recorded one.
        FileNotFoundError: If a reused sample file is missing.
    """
    recorded = rehearsal_sample_sha256(passed_rehearsal(context)).get(experiment, {})
    for fold_name in fold_names:
        path = sample_file(context, experiment, fold_name)
        if fold_name not in recorded:
            raise FinalRunRefused(
                f"the rehearsal recorded no sha256 for the {experiment} sample of {fold_name}"
            )
        if not path.is_file():
            raise FileNotFoundError(f"reused sample file not found: {path}")
        actual = file_sha256(path)
        if actual != recorded[fold_name]:
            raise FinalRunRefused(
                f"{path}: sha256 is {actual}, but the passed rehearsal trained on "
                f"{recorded[fold_name]}. This is not the M7 sample file; it is not used."
            )


def sample_job(
    context: FinalContext, experiment: str, fold_name: str, permit: FormalRunPermit
) -> SampleJob:
    """The sampling job of one NEW training fold of the formal run.

    Raises:
        ValueError: If the fold is a reused fold (its M7 sample is never drawn
            again) or does not train any T model.
        TypeError: If the fold is a T fold and no FormalRunPermit is given.
    """
    if fold_name not in new_sample_folds(context):
        raise ValueError(
            f"fold {fold_name} is not sampled by the formal run: new samples are drawn only "
            f"for {new_sample_folds(context)}; {reused_sample_folds(context)} reuse M7's files"
        )
    fold = context.manifest.get(fold_name)
    ratio = context.locked.tuned.negatives_per_positive
    return SampleJob(
        fold=fold,
        features_dir=context.paths.features_dir(fold_name),
        labels_file=context.paths.labels_file(fold_name),
        experiment=build_experiment(experiment, context.inputs.lock),
        wide_columns=wide_table_columns(context.inputs.lock),
        sample_files={ratio: sample_file(context, experiment, fold_name)},
        # ("V",) for a V fold; ("T",) for a T fold, and only with the permit.
        allowed_phases=final_training_label_phases(fold, permit),
        git_commit=context.stamp.git_commit,
    )


def run_sampling(
    context: FinalContext, experiment: str, fold_name: str, permit: FormalRunPermit
) -> SampleRecord:
    """Draw the sample of one new training fold with the locked seed and ratio."""
    job = sample_job(context, experiment, fold_name, permit)
    ratio = context.locked.tuned.negatives_per_positive
    return sample_fold(job, context.locked.config.sampling)[ratio]


# ---------------------------------------------------------------- Dataset, trial, baseline


def build_final_dataset(
    context: FinalContext, mode: str, experiment: str, target_fold: str
) -> DatasetRecord:
    """Build and save the binary Dataset of one target fold.

    In the formal run the reused sample files of the window are first checked
    against the sha256 the passed rehearsal recorded. Every sample file is then
    checked against its own record (build_and_save_dataset).
    """
    plan = dataset_plan(context, mode, experiment, target_fold)
    if mode == MODE_FORMAL:
        reused = set(reused_sample_folds(context))
        in_window = [source.fold for source in plan.sources if source.fold in reused]
        check_reused_samples(context, experiment, in_window)
    return build_and_save_dataset(plan, context.locked.config.sampling.row_group_rows)


def target_evaluator(
    context: FinalContext, mode: str, target_fold: str, permit: FormalRunPermit | None
) -> FoldEvaluator:
    """The evaluator of a target fold, through the mode's guarded loader.

    Raises:
        TypeError: In the formal mode without a FormalRunPermit.
        ValueError: If the mode may not evaluate this fold.
    """
    fold = context.manifest.get(target_fold)
    labels_file = context.paths.labels_file(target_fold)
    if mode == MODE_REHEARSAL:
        return load_rehearsal_evaluator(fold, labels_file, context.config.rehearsal_target_fold)
    if mode == MODE_FORMAL:
        if permit is None:
            raise TypeError("the formal mode needs a FormalRunPermit to read labels")
        return load_final_evaluator(fold, labels_file, permit)
    raise ValueError(f"unknown mode {mode!r}")


def evaluated_trees(context: FinalContext, mode: str) -> tuple[int, ...] | None:
    """The checkpoints a trial evaluates and saves.

    The rehearsal evaluates every checkpoint (100, ..., the locked number), so all
    of them can be compared with M7. The formal run evaluates ONLY the locked
    number of trees: metrics of a T fold at other numbers of trees are not
    needed, and not computing them leaves nothing to choose from afterwards.
    """
    if mode == MODE_REHEARSAL:
        return None
    return (context.locked.num_trees,)


def run_final_trial(
    context: FinalContext,
    mode: str,
    experiment: str,
    target_fold: str,
    permit: FormalRunPermit | None,
) -> TrialOutcome:
    """Train the model of one target fold from scratch, score the fold and evaluate it.

    The fold window is checked first (training_window); the target's labels are
    read after training, through the mode's guarded loader.

    Args:
        context: The context of this process.
        mode: "rehearsal" or "formal".
        experiment: "full" or "minus_r6".
        target_fold: The fold to predict.
        permit: The FormalRunPermit (formal mode); None in a rehearsal.

    Returns:
        The outcome; the caller writes the result rows.
    """
    plan = dataset_plan(context, mode, experiment, target_fold)  # checks the folds
    spec = trial_spec(context, mode, experiment, target_fold)
    inputs, locked = context.inputs, context.locked
    check_trial_settings(inputs.ranker, spec)
    settings = model_config(inputs.ranker, locked.tuned, experiment, locked.histogram_mode)
    config_id = experiment_config_id(context, experiment)
    files = trial_output_files(inputs, spec, config_id)
    booster, outcome, train_metric = train_planned_model(inputs, spec, plan, settings)
    evaluator = target_evaluator(context, mode, target_fold, permit)
    return score_and_save(
        inputs,
        spec,
        plan.experiment,
        booster,
        outcome,
        train_metric,
        files,
        evaluator,
        evaluated_trees(context, mode),
    )


def baseline_row(
    context: FinalContext, mode: str, target_fold: str, permit: FormalRunPermit | None
) -> MetricRow:
    """The RRF baseline of one target fold: no model, the same candidates and metrics."""
    training_window(context, mode, target_fold)  # the mode's fold check, before any label
    evaluator = target_evaluator(context, mode, target_fold, permit)
    return rrf_baseline_metric_row(context.inputs, target_fold, evaluator)
