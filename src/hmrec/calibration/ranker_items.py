"""Calibration item D: LightGBM training and evaluation in the container (LightGBM side).

NO new computation happens here. The item runs the functions of
hmrec.hpo.v_trial that M7 and the M8 rehearsal ran (plan_dataset_for_folds,
build_and_save_dataset, train_planned_model, score_and_save) with the locked
Ranker configuration, on the FROZEN stage-1 inputs:

    training   the four recorded sample files of the weeks before the
               calibration week (sha256 checked against the M8 rehearsal record)
    scoring    the frozen wide table of the calibration week (every part's
               sha256 is checked by hmrec.feature_files.verify_feature_files)

So only LightGBM and the OpenMP runtime it runs on differ from stage 1.

The configuration comes from hmrec.ranker.platform_entry in mode "calibration"
(the tolerance-accepted entry): the strict entry refuses a Linux process by
design. Outputs go to artifacts/m10_calibration/{platform}/{run}/main/ and to
results/m10_calibration/; no stage-1 file or table is written.

Two processes, as in stage 1: "ranker-dataset" builds the binary Dataset,
"ranker" trains, scores and evaluates.

The fold is checked with the M7 window (hmrec.ranker.window) and the labels
come through the M7 evaluator (hmrec.eval.ranker_eval): a T fold is refused.

Item E (end to end; run_end_to_end_ranker) is the second half of
hmrec.calibration.feature_items: the model item D trained scores the wide table
built from the container's OWN candidates, and the result is evaluated. Nothing
is trained there.

This module is on the LightGBM side; it must not import torch.
"""

import platform
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from hmrec.calibration.compare import (
    compare_top_k,
    leading_identical_trees,
    split_model_trees,
    within_relative,
)
from hmrec.calibration.context import CalibrationContext, ItemOutcome, timed
from hmrec.calibration.records import Measurement
from hmrec.calibration.references import (
    FULL_EXPERIMENT,
    stage1_dataset_row,
    stage1_ranker_metrics,
    stage1_ranker_trial,
    verify_files,
)
from hmrec.data.folds import load_fold_manifest
from hmrec.eval.ranker_eval import load_m7_evaluator
from hmrec.hpo.v_records import MetricRow
from hmrec.hpo.v_trial import (
    DatasetPlan,
    FoldEvaluation,
    TrialInputs,
    TrialOutcome,
    TrialSpec,
    build_and_save_dataset,
    check_trial_settings,
    evaluate_checkpoints,
    plan_dataset_for_folds,
    score_and_save,
    score_fold,
    train_planned_model,
)
from hmrec.locks import check_file_sha256, file_sha256, load_features_lock
from hmrec.logging_utils import get_logger
from hmrec.memory import read_process_peak_memory
from hmrec.paths import calibration_output_paths
from hmrec.ranker.ablation import build_experiment
from hmrec.ranker.dataset import read_dataset_record
from hmrec.ranker.identity import model_config, ranker_config_id
from hmrec.ranker.platform_entry import (
    MODE_CALIBRATION,
    PlatformLockedRanker,
    load_locked_ranker_for_platform,
)
from hmrec.ranker.runtime import read_lightgbm_runtime
from hmrec.ranker.train import load_model
from hmrec.ranker.window import m7_training_window

LOGGER = get_logger("calibration.ranker_items")

ITEM_D = "D"
ITEM_E = "E"
AREA = "m10_calibration"
# The results area of item E's Top-12 (its own folder next to item D's).
END_TO_END_AREA = "m10_end_to_end"
RANKING_SCOPE = "ranking"
MAIN_TREE = "main"
STEP = "calibration"
DATASET_SCOPE = "dataset"
MODEL_SCOPE = "model"
NUM_TREES = "num_trees"
TOP_COLUMNS = ["query_id", "rank", "article_idx"]

M8_REHEARSAL = "results/m8_rehearsal.csv"
M8_REHEARSAL_METRICS = "results/m8_rehearsal_metrics.csv"
M8_RUNS = "results/m8_runs.csv"


@dataclass(frozen=True)
class RankerSetup:
    """What both Ranker processes start with.

    Attributes:
        platform_locked: The locked configuration from the tolerance-accepted
            entry, with the recorded runtime.
        inputs: The configuration in the shape the shared trial code takes;
            inputs.paths are the STAGE-1 locations (samples and features are read there).
        plan: The Dataset of the calibration week's training window, in the
            calibration tree.
        spec: The trial: the locked tuned values, number of trees and histogram mode.
    """

    platform_locked: PlatformLockedRanker
    inputs: TrialInputs
    plan: DatasetPlan
    spec: TrialSpec


def load_ranker_setup(context: CalibrationContext) -> RankerSetup:
    """Get the locked configuration for this platform and plan the Dataset.

    Raises:
        RuntimeError: On macOS (the tolerance-accepted entry is for Linux only).
        ValueError: If a lock check fails, the fold is not a V fold with a full
            window of F / V folds, or a sample file has another sha256.
    """
    paths, config = context.paths, context.config
    platform_locked = load_locked_ranker_for_platform(
        paths,
        context.base,
        read_lightgbm_runtime(),
        sys.platform,
        platform.machine(),
        MODE_CALIBRATION,
        context.main_paths,
    )
    locked = platform_locked.locked
    inputs = TrialInputs(
        paths=paths,
        base=context.base,
        ranker=locked.config,
        manifest=load_fold_manifest(paths.fold_manifest_file),
        lock=load_features_lock(paths.features_lock_file),
        metrics=config.metrics,
    )
    # The M7 window: the target must be a V fold and every training fold an F or V fold.
    folds = m7_training_window(inputs.manifest, config.fold, locked.config.window_weeks)
    ratio = locked.tuned.negatives_per_positive
    sample_names = [
        paths.ranker_sample_file(FULL_EXPERIMENT, fold.name, ratio)
        .relative_to(paths.artifacts_root)
        .as_posix()
        for fold in folds
    ]
    verify_files(paths.artifacts_root, config.artifact_sha256, sample_names)
    plan = plan_dataset_for_folds(
        inputs,
        FULL_EXPERIMENT,
        config.fold,
        folds,
        ratio,
        context.stamp.git_commit,
        context.main_paths.ranker_dataset_dir(FULL_EXPERIMENT, config.fold, ratio),
    )
    spec = TrialSpec(
        area=AREA,
        experiment=FULL_EXPERIMENT,
        step=STEP,
        target_fold=config.fold,
        tuned=locked.tuned,
        num_trees=locked.num_trees,
        histogram_mode=locked.histogram_mode,
        evaluate=True,
    )
    check_trial_settings(locked.config, spec)
    return RankerSetup(platform_locked, inputs, plan, spec)


def require_new_outputs(files: tuple[Path, ...]) -> None:
    """Fail if an output of this run already exists (a calibration never reuses one)."""
    existing = [str(path) for path in files if path.exists()]
    if existing:
        raise FileExistsError(
            f"outputs of this calibration run already exist: {existing}. Nothing is reused "
            "or replaced; look at the earlier attempt first."
        )


# ---------------------------------------------------------------- the Dataset


def run_ranker_dataset(context: CalibrationContext) -> ItemOutcome:
    """Item D, first process: build the binary Dataset from the four recorded sample files."""
    setup = load_ranker_setup(context)
    plan, config = setup.plan, context.config
    require_new_outputs((plan.directory,))
    block_rows = setup.inputs.ranker.sampling.row_group_rows
    record, seconds, sampled_peak = timed(
        context, f"{config.fold} dataset", lambda: build_and_save_dataset(plan, block_rows)
    )
    item_peak = read_process_peak_memory()
    stage1 = stage1_dataset_row(context.paths, config)
    reference_dir = context.paths.final_dataset_dir(
        config.ranker_reference_area, FULL_EXPERIMENT, config.fold, record.ratio
    )
    recorded = read_dataset_record(reference_dir)
    where = reference_dir.relative_to(context.paths.artifacts_root.parent).as_posix()

    def row(
        quantity: str, value: float | int | str, stage1_value: float | int | str, source: str
    ) -> Measurement:
        return Measurement(ITEM_D, DATASET_SCOPE, quantity, value, stage1_value, source=source)

    rows = [
        row("train_rows", record.n_rows, recorded.n_rows, where),
        row("train_queries", record.n_queries, recorded.n_queries, where),
        row("sample_sha256_equal", record.sample_sha256 == recorded.sample_sha256, True, where),
        row("binary_sha256_equal", record.binary_sha256 == recorded.binary_sha256, True, where),
        row("seconds", seconds, float(stage1["generate_seconds"]), M8_RUNS),
        row(
            "max_rss_gb", item_peak.process_max_rss_gb, float(stage1["process_max_rss_gb"]), M8_RUNS
        ),
    ]
    return ItemOutcome(
        item=ITEM_D,
        scope=DATASET_SCOPE,
        item_seconds=seconds,
        sampled_peak_rss_gb=sampled_peak,
        item_peak=item_peak,
        measurements=rows,
        detail=(
            f"{record.n_rows} rows, {record.n_queries} queries, "
            f"folds {' '.join(record.train_folds)}"
        ),
    )


# ---------------------------------------------------------------- train, score, evaluate


def metric_measurements(
    context: CalibrationContext,
    metric_rows: Sequence[MetricRow],
    recorded: dict[int, dict[str, float]],
    item: str = ITEM_D,
) -> list[Measurement]:
    """One row per metric and checkpoint, next to the stage-1 record.

    The provisional tolerance of docs/mlops.md §8.3 names MAP@12 of a week: it
    is marked on the main metric at the locked number of trees only.
    """
    main_metric = context.config.metrics[0]
    limit = context.config.tolerance.max_map_relative_difference
    final_trees = max(row.num_trees for row in metric_rows)
    rows = []
    for metric_row in metric_rows:
        if metric_row.num_trees not in recorded:
            raise ValueError(f"stage 1 recorded no metrics at {metric_row.num_trees} trees")
        for metric in context.config.metrics:
            value, stage1 = metric_row.scores[metric], recorded[metric_row.num_trees][metric]
            judged = metric == main_metric and metric_row.num_trees == final_trees
            rows.append(
                Measurement(
                    item,
                    f"{metric_row.num_trees} trees",
                    metric,
                    value,
                    stage1,
                    f"relative difference <= {limit}" if judged else "",
                    within_relative(value, stage1, limit) if judged else None,
                    M8_REHEARSAL_METRICS,
                )
            )
    return rows


def model_measurements(
    model_file: Path, reference_model_file: Path, model_sha256: str, recorded_sha256: str
) -> list[Measurement]:
    """The container's model text against the stage-1 model, tree by tree.

    Args:
        model_file: The container's model.txt.
        reference_model_file: The stage-1 model.txt of the same configuration and fold.
        model_sha256: sha256 of the container's file.
        recorded_sha256: sha256 of the stage-1 file as its trial row records it.
    """
    measured_trees = split_model_trees(model_file.read_text(encoding="utf-8"))
    recorded_trees = split_model_trees(reference_model_file.read_text(encoding="utf-8"))
    identical = leading_identical_trees(recorded_trees, measured_trees)
    all_identical = identical == len(recorded_trees) == len(measured_trees)

    def row(quantity: str, value: float | int | str, stage1: float | int | str) -> Measurement:
        return Measurement(ITEM_D, MODEL_SCOPE, quantity, value, stage1, source=M8_REHEARSAL)

    return [
        row("trees", len(measured_trees), len(recorded_trees)),
        row("leading_identical_trees", identical, len(recorded_trees)),
        # Trees are numbered from 0; "none" if every tree is the same text.
        row("first_differing_tree", "none" if all_identical else identical, "none"),
        row("model_sha256_equal", model_sha256 == recorded_sha256, True),
    ]


def top_k_measurements(
    predictions_file: Path, reference_file: Path, num_trees: int, source: str, item: str = ITEM_D
) -> list[Measurement]:
    """The Top-K of every query at the locked number of trees, against the stage-1 Top-K."""
    measured = pl.read_parquet(predictions_file).filter(pl.col(NUM_TREES) == num_trees)
    recorded = pl.read_parquet(reference_file).filter(pl.col(NUM_TREES) == num_trees)
    comparison = compare_top_k(recorded.select(TOP_COLUMNS), measured.select(TOP_COLUMNS))
    scope = f"{num_trees} trees"

    def row(quantity: str, value: float | int) -> Measurement:
        stage1 = comparison.n_queries if quantity == "queries" else 1.0
        return Measurement(item, scope, quantity, value, stage1, source=source)

    return [
        row("queries", comparison.n_queries),
        row("identical_top12_list_query_share", comparison.identical_list_share),
        row("identical_top12_set_query_share", comparison.identical_set_share),
        row("mean_top12_overlap", comparison.mean_overlap),
    ]


def run_ranker(context: CalibrationContext) -> ItemOutcome:
    """Item D, second process: train the locked model, score the frozen week, evaluate, compare.

    Raises:
        ValueError: If the locked settings do not give the locked config_id, or a
            sha256 of an input or of the stage-1 model differs.
        FileExistsError: If the run already has a model or predictions.
    """
    paths, config, output = context.paths, context.config, context.main_paths
    setup = load_ranker_setup(context)
    locked, spec, inputs = setup.platform_locked.locked, setup.spec, setup.inputs
    settings = model_config(locked.config, locked.tuned, FULL_EXPERIMENT, locked.histogram_mode)
    config_id = ranker_config_id(settings)
    if config_id != locked.config_id:
        raise ValueError(
            f"the locked settings give {config_id}, the lock records {locked.config_id}"
        )
    feature_names = [
        paths.features_metadata_file(config.fold).relative_to(paths.artifacts_root).as_posix(),
        paths.category_codes_file.relative_to(paths.artifacts_root).as_posix(),
        paths.labels_file(config.fold).relative_to(paths.artifacts_root).as_posix(),
    ]
    verify_files(paths.artifacts_root, config.artifact_sha256, feature_names)
    key = (AREA, FULL_EXPERIMENT, config_id, config.fold)
    files = (
        output.ranker_predictions_file(*key),
        output.ranker_per_query_file(*key),
        output.ranker_model_file(*key),
    )
    require_new_outputs(files)

    def train_score_evaluate() -> TrialOutcome:
        booster, trained, train_metric = train_planned_model(inputs, spec, setup.plan, settings)
        # The M7 evaluator checks the phase before the label file is opened (V folds only).
        evaluator = load_m7_evaluator(
            inputs.manifest.get(config.fold), paths.labels_file(config.fold)
        )
        return score_and_save(
            inputs, spec, setup.plan.experiment, booster, trained, train_metric, files, evaluator
        )

    outcome, seconds, sampled_peak = timed(context, f"{config.fold} ranker", train_score_evaluate)
    item_peak = read_process_peak_memory()
    stage1 = stage1_ranker_trial(paths, config, config_id)
    reference_key = (config.ranker_reference_area, FULL_EXPERIMENT, config_id, config.fold)
    reference_model = paths.ranker_model_file(*reference_key)
    check_file_sha256(reference_model, stage1["model_sha256"])
    recorded_metrics = stage1_ranker_metrics(paths, config, config_id)
    rows = metric_measurements(context, outcome.metric_rows, recorded_metrics)
    rows.extend(
        model_measurements(files[2], reference_model, outcome.model_sha256, stage1["model_sha256"])
    )
    reference_predictions = paths.ranker_predictions_file(*reference_key)
    where = reference_predictions.relative_to(paths.artifacts_root.parent).as_posix()
    rows.extend(top_k_measurements(files[0], reference_predictions, spec.num_trees, where))

    def row(quantity: str, value: float | int, column: str) -> Measurement:
        return Measurement(
            ITEM_D, MODEL_SCOPE, quantity, value, float(stage1[column]), source=M8_REHEARSAL
        )

    rows.extend(
        [
            row("train_rows", outcome.n_train_rows, "n_train_rows"),
            row("scored_rows", outcome.n_scored_rows, "n_scored_rows"),
            row("load_seconds", outcome.load_seconds, "load_seconds"),
            row("train_seconds", outcome.train_seconds, "train_seconds"),
            row("seconds_per_tree", outcome.seconds_per_tree, "seconds_per_tree"),
            row("score_seconds", outcome.score_seconds, "score_seconds"),
            row("seconds", seconds, "total_seconds"),
            row("max_rss_gb", item_peak.process_max_rss_gb, "process_max_rss_gb"),
        ]
    )
    runtime = setup.platform_locked.runtime
    return ItemOutcome(
        item=ITEM_D,
        scope=MODEL_SCOPE,
        item_seconds=seconds,
        sampled_peak_rss_gb=sampled_peak,
        item_peak=item_peak,
        measurements=rows,
        detail=(
            f"config {config_id}, {spec.num_trees} trees, model sha256 {outcome.model_sha256}; "
            f"LightGBM {runtime.lightgbm_version} on {runtime.libomp_version} "
            f"(entry mode {setup.platform_locked.mode})"
        ),
    )


# ---------------------------------------------------------------- item E: the run's own features


def run_end_to_end_ranker(context: CalibrationContext) -> ItemOutcome:
    """Item E, second process: score this run's own wide table with the model of item D.

    The model is the one item D trained (configs/m10_calibration.yaml:
    end_to_end.ranker_model_run); nothing is trained here. The wide table is the
    one hmrec.calibration.feature_items wrote for this run; every part's sha256
    is checked against the record written with it. Scoring and evaluation are the
    functions of hmrec.hpo.v_trial that M7 used (score_fold, evaluate_checkpoints).

    Raises:
        ValueError: If a lock check fails, the fold is not a V fold, or a feature
            part has another sha256 than its record.
        FileNotFoundError: If the model of item D or the features are missing.
        FileExistsError: If the run already has an end-to-end Top-12.
    """
    paths, config, output = context.paths, context.config, context.main_paths
    setup = load_ranker_setup(context)
    locked, inputs = setup.platform_locked.locked, setup.inputs
    config_id, fold = locked.config_id, config.fold
    labels_name = paths.labels_file(fold).relative_to(paths.artifacts_root).as_posix()
    verify_files(paths.artifacts_root, config.artifact_sha256, [labels_name])
    key = (END_TO_END_AREA, FULL_EXPERIMENT, config_id, fold)
    files = (output.ranker_predictions_file(*key), output.ranker_per_query_file(*key))
    require_new_outputs(files)
    model_paths = calibration_output_paths(
        paths, context.platform_name, config.ranker_model_run, MAIN_TREE
    )
    model_file = model_paths.ranker_model_file(AREA, FULL_EXPERIMENT, config_id, fold)
    stage1 = stage1_ranker_trial(paths, config, config_id)
    same_model = file_sha256(model_file) == stage1["model_sha256"]
    experiment = build_experiment(FULL_EXPERIMENT, inputs.lock)

    def score_and_evaluate() -> tuple[FoldEvaluation, int, float]:
        started = time.perf_counter()
        booster = load_model(model_file)
        # score_fold verifies the feature files against features.json before reading them.
        tops, n_scored = score_fold(
            booster, output.features_dir(fold), experiment, inputs, locked.num_trees
        )
        score_seconds = time.perf_counter() - started
        # The M7 evaluator checks the phase before the label file is opened (V folds only).
        evaluator = load_m7_evaluator(inputs.manifest.get(fold), paths.labels_file(fold))
        evaluation = evaluate_checkpoints(
            evaluator, tops, inputs.metrics, FULL_EXPERIMENT, config_id, {}
        )
        return evaluation, n_scored, score_seconds

    (evaluation, n_scored, score_seconds), seconds, sampled_peak = timed(
        context, f"{fold} end-to-end ranking", score_and_evaluate
    )
    item_peak = read_process_peak_memory()
    files[0].parent.mkdir(parents=True, exist_ok=True)
    evaluation.predictions.write_parquet(files[0])
    evaluation.per_query.write_parquet(files[1])
    rows = metric_measurements(
        context, evaluation.metric_rows, stage1_ranker_metrics(paths, config, config_id), ITEM_E
    )
    reference_key = (config.ranker_reference_area, FULL_EXPERIMENT, config_id, fold)
    reference_predictions = paths.ranker_predictions_file(*reference_key)
    where = reference_predictions.relative_to(paths.artifacts_root.parent).as_posix()
    rows.extend(
        top_k_measurements(files[0], reference_predictions, locked.num_trees, where, ITEM_E)
    )

    def row(quantity: str, value: float | int | str, reference: float | int | str) -> Measurement:
        return Measurement(ITEM_E, RANKING_SCOPE, quantity, value, reference, source=M8_REHEARSAL)

    rows.extend(
        [
            row("ranker_model_sha256_equal_to_stage1", same_model, True),
            row("scored_rows", n_scored, int(stage1["n_scored_rows"])),
            row("score_seconds", score_seconds, float(stage1["score_seconds"])),
            row("seconds", seconds, float(stage1["score_seconds"])),
            row("max_rss_gb", item_peak.process_max_rss_gb, float(stage1["process_max_rss_gb"])),
        ]
    )
    return ItemOutcome(
        item=ITEM_E,
        scope=RANKING_SCOPE,
        item_seconds=seconds,
        sampled_peak_rss_gb=sampled_peak,
        item_peak=item_peak,
        measurements=rows,
        detail=(
            f"model of item D from {config.ranker_model_run} (sha256 equal to stage 1: "
            f"{same_model}); {n_scored} rows of this run's own features scored"
        ),
    )
