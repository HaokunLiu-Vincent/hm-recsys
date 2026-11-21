"""Calibration items A, B, C1, C2 and the union: recall in the container (torch side).

NO new computation happens here. Every item calls the functions the stage-1
script scripts/m6_generate_candidates.py calls (hmrec.recall.generate,
hmrec.recall.locked, hmrec.recall.chain_records), with the locked configuration
of configs/locks/. Two things differ from stage 1, and only these:

    where the outputs go    a second ProjectPaths points into
                            artifacts/m10_calibration/{platform}/{run}/, so no
                            stage-1 file can be written;
    what happens afterwards the output is compared with the stage-1 output of the
                            same week (hmrec.calibration.compare) and the numbers
                            are written to results/m10_calibration/.

Items (docs/m10_calibration_plan.md §4):

    A    R1, R2, R3, R4 candidates of the week
    B    one R5 (WALS) fit as of the week, with its candidates
    C1   R6 candidates of the week from the RECORDED model of the week before
         (no training): the same model applied on another platform
    C2   what one tick of the weekly pipeline does: the recorded model of two
         weeks before predicts the week before and is fine-tuned on it (exactly
         the stage-1 process of that week); the resulting model then predicts
         the week
    union  the six channels of the container's own run, fused

Labels of the calibration week (a V fold) are read only through
hmrec.eval.monitor.load_label_evaluator, the M6 guard that refuses a T fold,
and only AFTER the candidates are written. Nothing is selected.

This module is on the torch side (hmrec.recall loads torch); it must not import
lightgbm.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import torch

from hmrec.calibration.compare import (
    CandidateComparison,
    compare_candidates,
    max_abs_matrix_difference,
    within_relative,
)
from hmrec.calibration.context import CalibrationContext, ItemOutcome, timed
from hmrec.calibration.records import Measurement
from hmrec.calibration.references import (
    stage1_candidates_row,
    stage1_run_row,
    verify_files,
)
from hmrec.config import (
    CatalogConfig,
    PreprocessConfig,
    load_candidate_run_config,
    load_catalog_config,
    load_preprocess_config,
)
from hmrec.data.folds import FoldManifest, load_fold_manifest
from hmrec.eval.metrics import macro_mean
from hmrec.eval.monitor import label_metrics, load_label_evaluator
from hmrec.eval.report import FoldEvaluator
from hmrec.fusion.union import fuse_candidates
from hmrec.locks import (
    ALL_CHANNELS,
    F1_START_CHECKPOINT,
    R6_CHANNEL,
    STATISTICAL_CHANNELS,
    check_file_sha256,
    file_sha256,
)
from hmrec.logging_utils import get_logger
from hmrec.memory import ProcessPeakMemory, read_process_peak_memory
from hmrec.paths import ProjectPaths
from hmrec.recall.chain_records import ChainRecord, read_chain_records, start_model_for
from hmrec.recall.generate import (
    R5_CHANNEL,
    TWOTOWER_SOURCE,
    WALS_SOURCE,
    FoldJob,
    check_fine_tune_loss,
    fine_tune_twotower_fold,
    generate_statistical_channel,
    predict_twotower_fold,
)
from hmrec.recall.inputs import SharedTables, build_recall_inputs, load_shared_tables, read_queries
from hmrec.recall.locked import (
    LockedRecall,
    ModelFile,
    load_locked_recall,
    load_locked_twotower_chain,
    locked_checkpoint_file,
    post_lock_folds,
)
from hmrec.run_records import SKIPPED
from hmrec.vectors import load_vectors

LOGGER = get_logger("calibration.recall_items")

ITEM_A = "A"
ITEM_B = "B"
ITEM_C1 = "C1"
ITEM_C2 = "C2"
ITEM_UNION = "union"
START_RECORDED = "recorded"
START_CALIBRATION = "calibration"
STARTS = (START_RECORDED, START_CALIBRATION)
FINE_TUNE_SCOPE = "fine_tune"
UNION_SCOPE = "union"
MONITOR_STEP = "monitor"
CANDIDATE_COLUMNS = ["query_id", "article_idx", "rank"]

M6_CANDIDATES = "results/m6_candidates.csv"
M6_RUNS = "results/m6_runs.csv"
M6_CHAIN = "results/m6_twotower_chain.csv"


@dataclass(frozen=True)
class RecallSetup:
    """What every recall item reads before it starts (the same things as stage 1).

    Attributes:
        preprocess: configs/preprocess.yaml.
        catalog: configs/catalog.yaml (the active pool).
        manifest: configs/fold_manifest.yaml.
        fold_names: The post-lock folds in time order (F1 ... T4).
        locked: Both recall lock files.
        tables: The processed tables (transactions lazy).
    """

    preprocess: PreprocessConfig
    catalog: CatalogConfig
    manifest: FoldManifest
    fold_names: tuple[str, ...]
    locked: LockedRecall
    tables: SharedTables


def load_recall_setup(context: CalibrationContext) -> RecallSetup:
    """Check the processed tables' sha256, then read configs, locks and tables.

    Also sets torch's thread count to num_threads of configs/base.yaml, as the
    stage-1 script does in every recall process.
    """
    paths, config = context.paths, context.config
    torch.set_num_threads(context.base.num_threads)
    verify_files(paths.processed_dir, config.processed_sha256, list(config.processed_sha256))
    preprocess = load_preprocess_config(paths.preprocess_config_file)
    manifest = load_fold_manifest(paths.fold_manifest_file)
    return RecallSetup(
        preprocess=preprocess,
        catalog=load_catalog_config(paths.catalog_config_file),
        manifest=manifest,
        fold_names=tuple(fold.name for fold in post_lock_folds(manifest)),
        locked=load_locked_recall(paths),
        tables=load_shared_tables(paths, preprocess),
    )


def fold_job(
    context: CalibrationContext, setup: RecallSetup, fold_name: str, output_paths: ProjectPaths
) -> FoldJob:
    """The stage-1 FoldJob of a fold, with outputs redirected to the calibration tree.

    Queries and inputs are read from the stage-1 locations (the query file's
    sha256 is checked first); job.paths, which the stage-1 functions use ONLY
    for their outputs of R1-R5, is the calibration tree.
    """
    paths = context.paths
    queries_name = paths.queries_file(fold_name).relative_to(paths.artifacts_root).as_posix()
    verify_files(paths.artifacts_root, context.config.artifact_sha256, [queries_name])
    fold = setup.manifest.get(fold_name)
    return FoldJob(
        paths=output_paths,
        base=context.base,
        preprocess=setup.preprocess,
        locked=setup.locked,
        fold=fold,
        fold_names=setup.fold_names,
        queries=read_queries(paths, fold_name),
        inputs=build_recall_inputs(setup.tables, fold, setup.preprocess, setup.catalog),
    )


def require_new_output(path: Path) -> None:
    """Fail if an output of this run already exists.

    A calibration run measures; it never reuses a file of an earlier attempt.

    Raises:
        FileExistsError: If `path` exists.
    """
    if path.exists():
        raise FileExistsError(
            f"{path} already exists. A calibration run never reuses or replaces an output; "
            "look at the earlier attempt first (nothing is deleted automatically)."
        )


# ---------------------------------------------------------------- comparison with stage 1


def week_evaluator(context: CalibrationContext, setup: RecallSetup) -> FoldEvaluator:
    """The evaluator of the calibration week, through the M6 guard (F and V folds only)."""
    paths, fold = context.paths, setup.manifest.get(context.config.fold)
    labels_name = paths.labels_file(fold.name).relative_to(paths.artifacts_root).as_posix()
    verify_files(paths.artifacts_root, context.config.artifact_sha256, [labels_name])
    # load_label_evaluator checks the phase before it opens the file: a T fold raises.
    return load_label_evaluator(fold.name, fold.phase, paths.labels_file(fold.name))


def read_reference_candidates(context: CalibrationContext, channel: str) -> pl.DataFrame:
    """The stage-1 candidates of the calibration week, after checking their sha256."""
    paths = context.paths
    file = paths.candidates_file(context.config.fold, channel)
    name = file.relative_to(paths.artifacts_root).as_posix()
    verify_files(paths.artifacts_root, context.config.reference_sha256, [name])
    return pl.read_parquet(file)


def candidate_measurements(
    context: CalibrationContext, item: str, channel: str, comparison: CandidateComparison
) -> list[Measurement]:
    """The rows of one channel's candidates comparison.

    The provisional tolerance of docs/mlops.md §8.3 is "identical" for R1-R4 and
    "at least 99% of the recorded pairs present" for R5 and R6.
    """
    source = f"artifacts/candidates/{context.config.fold}/{channel}.parquet"
    minimum = context.config.tolerance.min_recorded_pair_share
    is_statistical = channel in STATISTICAL_CHANNELS and channel != R5_CHANNEL

    def row(quantity: str, measured: float | int | str, stage1: float | int | str) -> Measurement:
        return Measurement(item, channel, quantity, measured, stage1, source=source)

    rows = [
        row("rows", comparison.measured_rows, comparison.recorded_rows),
        Measurement(
            item,
            channel,
            "rank_rows_identical",
            comparison.rank_rows_identical,
            True,
            "identical (customer, article, rank) rows" if is_statistical else "",
            comparison.rank_rows_identical if is_statistical else None,
            source,
        ),
        row("differing_rank_rows", comparison.differing_rank_rows, 0),
        Measurement(
            item,
            channel,
            "recorded_pairs_present_share",
            comparison.recorded_pairs_present_share,
            1.0,
            "" if is_statistical else f">= {minimum}",
            None if is_statistical else comparison.recorded_pairs_present_share >= minimum,
            source,
        ),
        row("new_pairs_share", comparison.new_pairs_share, 0.0),
        row("common_pairs_same_rank_share", comparison.common_pairs_same_rank_share, 1.0),
        row("scores_bit_identical", comparison.scores_bit_identical, True),
        row("max_abs_score_difference", comparison.max_abs_score_difference, 0.0),
        row("max_rel_score_difference", comparison.max_rel_score_difference, 0.0),
    ]
    return rows


def recall_measurement(
    context: CalibrationContext, item: str, scope: str, recall: float
) -> Measurement:
    """A recall next to the stage-1 recall of results/m6_candidates.csv."""
    recorded = float(stage1_candidates_row(context.paths, context.config.fold, scope)["recall"])
    limit = context.config.tolerance.max_recall_relative_difference
    return Measurement(
        item,
        scope,
        "recall",
        recall,
        recorded,
        f"relative difference <= {limit}",
        within_relative(recall, recorded, limit),
        M6_CANDIDATES,
    )


def resource_measurements(
    item: str,
    scope: str,
    seconds: float,
    peak: ProcessPeakMemory,
    stage1_row: dict[str, str],
    note: str = "",
) -> list[Measurement]:
    """Wall time and true memory peak of an item next to the stage-1 process's values.

    Args:
        item: The calibration item.
        scope: What inside the item.
        seconds: Wall time of the item's computation.
        peak: ru_maxrss right after the computation.
        stage1_row: The stage-1 row of results/m6_runs.csv.
        note: Added to the source when the stage-1 process did more or less than the item.
    """
    source = M6_RUNS + (f" ({note})" if note else "")
    return [
        Measurement(
            item, scope, "seconds", seconds, float(stage1_row["generate_seconds"]), source=source
        ),
        Measurement(
            item,
            scope,
            "max_rss_gb",
            peak.process_max_rss_gb,
            float(stage1_row["process_max_rss_gb"]),
            source=source,
        ),
    ]


def vector_measurements(
    context: CalibrationContext,
    item: str,
    scope: str,
    source: str,
    measured_dir: Path,
    recorded_dir: Path,
) -> list[Measurement]:
    """Largest absolute difference of the saved user and item vectors (information only).

    Args:
        context: The calibration context.
        item: The calibration item.
        scope: The channel.
        source: "wals" or "twotower" (hmrec.vectors).
        measured_dir: The container's vectors folder.
        recorded_dir: The stage-1 vectors folder of the same week.
    """
    fold = context.config.fold
    measured = load_vectors(measured_dir, source, fold)
    recorded = load_vectors(recorded_dir, source, fold)
    user_difference = max_abs_matrix_difference(
        recorded.user_customer_idx,
        recorded.user_vectors,
        measured.user_customer_idx,
        measured.user_vectors,
    )
    item_difference = max_abs_matrix_difference(
        recorded.item_article_idx,
        recorded.item_vectors,
        measured.item_article_idx,
        measured.item_vectors,
    )
    where = recorded_dir.relative_to(context.paths.artifacts_root.parent).as_posix()
    return [
        Measurement(
            item, scope, "max_abs_user_vector_difference", user_difference, 0.0, source=where
        ),
        Measurement(
            item, scope, "max_abs_item_vector_difference", item_difference, 0.0, source=where
        ),
    ]


def compare_channel_with_stage1(
    context: CalibrationContext,
    setup: RecallSetup,
    item: str,
    channel: str,
    candidates_file: Path,
) -> list[Measurement]:
    """Everything that is compared for one channel's candidates of the calibration week."""
    measured = pl.read_parquet(candidates_file)
    recorded = read_reference_candidates(context, channel)
    rows = candidate_measurements(context, item, channel, compare_candidates(recorded, measured))
    reference_name = (
        context.paths.candidates_file(context.config.fold, channel)
        .relative_to(context.paths.artifacts_root)
        .as_posix()
    )
    same_file = file_sha256(candidates_file) == context.config.reference_sha256[reference_name]
    rows.append(Measurement(item, channel, "file_sha256_equal", same_file, True))
    evaluator = week_evaluator(context, setup)
    metric = setup.locked.channel(channel).d_phase_record.metric
    # The same expression as hmrec.eval.monitor.label_metrics uses for a channel.
    recall = macro_mean(evaluator.per_query(measured.select(CANDIDATE_COLUMNS), metric))
    rows.append(recall_measurement(context, item, channel, recall))
    return rows


# ---------------------------------------------------------------- items A and B


def run_statistical_channel(context: CalibrationContext, channel: str) -> ItemOutcome:
    """Items A (R1-R4) and B (R5): generate one channel of the week, then compare.

    Args:
        context: The calibration context.
        channel: One of the five statistical channels.

    Raises:
        ValueError: If `channel` is R6 or unknown.
        FileExistsError: If the run already has this channel's candidates.
    """
    if channel not in STATISTICAL_CHANNELS:
        raise ValueError(f"expected one of {STATISTICAL_CHANNELS}, got {channel!r}")
    item = ITEM_B if channel == R5_CHANNEL else ITEM_A
    setup = load_recall_setup(context)
    fold = context.config.fold
    job = fold_job(context, setup, fold, context.main_paths)
    candidates_file = context.main_paths.candidates_file(fold, channel)
    require_new_output(candidates_file)
    # The stage-1 function: builds the channel from the lock, generates, validates, writes.
    result = generate_statistical_channel(job, channel)
    if result.status == SKIPPED:
        raise FileExistsError(f"{candidates_file} appeared during the run; nothing was generated")
    item_peak = read_process_peak_memory()
    rows = compare_channel_with_stage1(context, setup, item, channel, candidates_file)
    if channel == R5_CHANNEL:
        rows.extend(
            vector_measurements(
                context,
                item,
                channel,
                WALS_SOURCE,
                context.main_paths.wals_vectors_dir(fold),
                context.paths.wals_vectors_dir(fold),
            )
        )
    stage1 = stage1_run_row(context.paths, fold, channel)
    rows.extend(resource_measurements(item, channel, result.generate_seconds, item_peak, stage1))
    return ItemOutcome(
        item=item,
        scope=channel,
        item_seconds=result.generate_seconds,
        sampled_peak_rss_gb=result.sampled_peak_rss_gb,
        item_peak=item_peak,
        measurements=rows,
        detail=f"{result.n_rows} rows; {result.checked_rows} scores checked",
    )


# ---------------------------------------------------------------- R6: items C1 and C2


def locked_f1_start(context: CalibrationContext, setup: RecallSetup) -> ModelFile:
    """The locked model that predicts the first post-lock fold, with its locked sha256.

    start_model_for needs it as an argument; it is only used for the first
    fold (F1), which a calibration never predicts.
    """
    lock = setup.locked.twotower
    return ModelFile(
        locked_checkpoint_file(context.paths, lock, F1_START_CHECKPOINT),
        lock.checkpoints[F1_START_CHECKPOINT].sha256,
    )


def run_r6_predict(context: CalibrationContext, start_kind: str) -> ItemOutcome:
    """Items C1 and C2 (second half): R6 candidates of the week from a checked model.

    Args:
        context: The calibration context.
        start_kind: "recorded" = the stage-1 model of the week before, as
            recorded in results/m6_twotower_chain.csv (item C1). "calibration" =
            the model this run fine-tuned itself, as recorded in the run's own
            chain table (item C2).

    Raises:
        ValueError: On an unknown start, or if the start model's sha256 differs
            from its record (nothing is loaded).
        FileExistsError: If the run already has these candidates.
    """
    if start_kind not in STARTS:
        raise ValueError(f"--start must be one of {STARTS}, got {start_kind!r}")
    paths, fold = context.paths, context.config.fold
    setup = load_recall_setup(context)
    if start_kind == START_RECORDED:
        item, output = ITEM_C1, context.recorded_model_paths
        chain_file = paths.m6_twotower_chain_results_file
        checkpoint_file: Callable[[str], Path] = paths.twotower_locked_chain_checkpoint_file
    else:
        item, output = ITEM_C2, context.main_paths
        chain_file = paths.calibration_chain_file(context.platform_name, context.run)
        checkpoint_file = context.main_paths.twotower_locked_chain_checkpoint_file
    job = fold_job(context, setup, fold, output)
    candidates_file = output.candidates_file(fold, R6_CHANNEL)
    require_new_output(candidates_file)
    # start_model_for checks that the model of the week before exists and has the sha256
    # its record gives; the loader hashes it again before any weight is read.
    start = start_model_for(
        fold,
        setup.fold_names,
        locked_f1_start(context, setup),
        read_chain_records(chain_file),
        checkpoint_file,
    )
    chain = load_locked_twotower_chain(setup.locked, context.base, paths, setup.preprocess, start)
    LOGGER.info("%s %s: predicting %s with %s", item, R6_CHANNEL, fold, start.path)
    vectors_dir = output.twotower_vectors_dir(fold)
    (_, checked), seconds, sampled_peak = timed(
        context,
        f"{fold} {R6_CHANNEL} predict",
        lambda: predict_twotower_fold(
            chain, fold, job.queries, job.inputs, candidates_file, vectors_dir
        ),
    )
    item_peak = read_process_peak_memory()
    rows = compare_channel_with_stage1(context, setup, item, R6_CHANNEL, candidates_file)
    rows.extend(
        vector_measurements(
            context,
            item,
            R6_CHANNEL,
            TWOTOWER_SOURCE,
            vectors_dir,
            paths.twotower_vectors_dir(fold),
        )
    )
    stage1 = stage1_run_row(paths, fold, R6_CHANNEL)
    note = "seconds = prediction only; the stage-1 process also fine-tuned on the week"
    rows.extend(resource_measurements(item, R6_CHANNEL, seconds, item_peak, stage1, note))
    return ItemOutcome(
        item=item,
        scope=R6_CHANNEL,
        item_seconds=seconds,
        sampled_peak_rss_gb=sampled_peak,
        item_peak=item_peak,
        measurements=rows,
        detail=f"start model {start.path.name} sha256 {start.sha256}; {checked} scores checked",
    )


@dataclass(frozen=True)
class ParameterDifference:
    """Two saved models of the same architecture, compared tensor by tensor.

    Attributes:
        max_abs_difference: Largest |measured - recorded| over all parameters (float64).
        n_values: Number of parameter values compared.
        n_different_values: How many of them differ.
    """

    max_abs_difference: float
    n_values: int
    n_different_values: int


def parameter_difference(recorded_file: Path, measured_file: Path) -> ParameterDifference:
    """Compare the weights of two checkpoints written by the Two-Tower chain.

    Args:
        recorded_file: The stage-1 checkpoint.
        measured_file: The container's checkpoint of the same fold.

    Raises:
        ValueError: If the two checkpoints do not hold the same tensors.
    """
    recorded = torch.load(recorded_file, map_location="cpu", weights_only=True)["state_dict"]
    measured = torch.load(measured_file, map_location="cpu", weights_only=True)["state_dict"]
    if list(recorded) != list(measured):
        raise ValueError(f"{recorded_file} and {measured_file} hold different tensors")
    max_abs, n_values, n_different = 0.0, 0, 0
    for name, recorded_tensor in recorded.items():
        measured_tensor = measured[name]
        if recorded_tensor.shape != measured_tensor.shape:
            raise ValueError(f"tensor {name} has two shapes in the two checkpoints")
        # float64: the difference of two float32 values is exact in float64.
        difference = (measured_tensor.double() - recorded_tensor.double()).abs()
        n_values += difference.numel()
        n_different += int(torch.count_nonzero(difference).item())
        if difference.numel() > 0:
            max_abs = max(max_abs, float(difference.max().item()))
    return ParameterDifference(max_abs, n_values, n_different)


def fine_tune_measurements(
    measured: ChainRecord, recorded: ChainRecord, parameters: ParameterDifference
) -> list[Measurement]:
    """The rows of the fine-tuning step: its record next to the stage-1 record of the fold."""

    def row(quantity: str, value: float | int | str, stage1: float | int | str) -> Measurement:
        return Measurement(ITEM_C2, FINE_TUNE_SCOPE, quantity, value, stage1, source=M6_CHAIN)

    return [
        row("start_sha256_equal", measured.start_sha256 == recorded.start_sha256, True),
        row("n_new_rows", measured.n_new_rows, recorded.n_new_rows),
        row("n_replay_rows", measured.n_replay_rows, recorded.n_replay_rows),
        row("steps", measured.steps, recorded.steps),
        row("mean_train_loss", measured.mean_train_loss, recorded.mean_train_loss),
        row("alarm_loss", measured.alarm_loss, recorded.alarm_loss),
        row("model_sha256_equal", measured.sha256 == recorded.sha256, True),
        row("max_abs_parameter_difference", parameters.max_abs_difference, 0.0),
        row("parameter_values", parameters.n_values, parameters.n_values),
        row("different_parameter_values", parameters.n_different_values, 0),
        row("fine_tune_seconds", measured.fine_tune_seconds, recorded.fine_tune_seconds),
    ]


def run_r6_fine_tune(context: CalibrationContext) -> ItemOutcome:
    """Item C2 (first half): replay the stage-1 R6 process of the week before the calibration week.

    Exactly what stage 1 did for that week (hmrec.recall.generate.generate_twotower_fold):
    the recorded model of the week before it is loaded after its sha256 check,
    predicts the week (candidates and vectors go to the calibration tree), and
    is then fine-tuned on the week's purchases with the locked settings. The
    new model is saved in the calibration tree and recorded in the run's own
    chain table; results/m6_twotower_chain.csv is not written.

    Raises:
        ValueError: If a sha256 differs from its record, or the fine-tune loss
            is not finite or above the stop limit of configs/m6_candidates.yaml.
        FileExistsError: If the run already has this fold's candidates or model.
    """
    paths, fold_name = context.paths, context.config.fine_tune_fold
    setup = load_recall_setup(context)
    output = context.main_paths
    job = fold_job(context, setup, fold_name, output)
    candidates_file = output.candidates_file(fold_name, R6_CHANNEL)
    model_file = output.twotower_locked_chain_checkpoint_file(fold_name)
    require_new_output(candidates_file)
    require_new_output(model_file)
    stage1_records = read_chain_records(paths.m6_twotower_chain_results_file)
    start = start_model_for(
        fold_name,
        setup.fold_names,
        locked_f1_start(context, setup),
        stage1_records,
        paths.twotower_locked_chain_checkpoint_file,
    )
    chain = load_locked_twotower_chain(setup.locked, context.base, paths, setup.preprocess, start)
    # The chain saves its fine-tuned model where this function points: the calibration
    # tree. The loader set it to the stage-1 folder, which a calibration never writes.
    chain.checkpoint_file = output.twotower_locked_chain_checkpoint_file
    LOGGER.info("C2: %s predicts %s, then is fine-tuned on it", start.path, fold_name)
    _, predict_seconds, sampled_peak = timed(
        context,
        f"{fold_name} {R6_CHANNEL} predict",
        lambda: predict_twotower_fold(
            chain,
            fold_name,
            job.queries,
            job.inputs,
            candidates_file,
            output.twotower_vectors_dir(fold_name),
        ),
    )
    # Only now may the model see the fold's own week (as in stage 1).
    chain_file = paths.calibration_chain_file(context.platform_name, context.run)
    record = fine_tune_twotower_fold(chain, job.fold, job.inputs, start, context.stamp, chain_file)
    item_peak = read_process_peak_memory()
    monitoring = load_candidate_run_config(paths.m6_config_file)
    check_fine_tune_loss(record, monitoring.max_fine_tune_loss)
    recorded = stage1_records[fold_name]
    recorded_model = paths.twotower_locked_chain_checkpoint_file(fold_name)
    check_file_sha256(recorded_model, recorded.sha256)
    rows = fine_tune_measurements(
        record, recorded, parameter_difference(recorded_model, model_file)
    )
    stage1 = stage1_run_row(paths, fold_name, R6_CHANNEL)
    note = f"the stage-1 process of {fold_name}: prediction seconds; peak of prediction + fine-tune"
    rows.extend(
        resource_measurements(ITEM_C2, FINE_TUNE_SCOPE, predict_seconds, item_peak, stage1, note)
    )
    return ItemOutcome(
        item=ITEM_C2,
        scope=FINE_TUNE_SCOPE,
        item_seconds=predict_seconds + record.fine_tune_seconds,
        sampled_peak_rss_gb=max(sampled_peak, record.sampled_peak_rss_gb),
        item_peak=item_peak,
        measurements=rows,
        detail=(
            f"start {start.path.name} sha256 {start.sha256}; model {model_file.name} "
            f"sha256 {record.sha256}; train loss {record.mean_train_loss!r}"
        ),
    )


# ---------------------------------------------------------------- the union


def run_union(context: CalibrationContext) -> ItemOutcome:
    """Fuse the run's own six channels of the week; compare the count and the union recall.

    The six files are the run's own: R1-R5 of items A and B, and R6 of item C2
    (from the model the run fine-tuned itself). The fused table is built with
    hmrec.fusion.union.fuse_candidates, the function the wide table is built
    with; the recalls with hmrec.eval.monitor.label_metrics, as in stage 1.

    Raises:
        FileNotFoundError: If a channel of the run is missing.
    """
    paths, fold = context.paths, context.config.fold
    setup = load_recall_setup(context)

    def fuse() -> tuple[int, dict[str, pl.DataFrame]]:
        tables = {
            channel: pl.read_parquet(context.main_paths.candidates_file(fold, channel))
            for channel in ALL_CHANNELS
        }
        # One row per distinct (query_id, article_idx) over the six channels.
        return fuse_candidates(tables).height, tables

    (n_rows, tables), seconds, sampled_peak = timed(context, f"{fold} union", fuse)
    item_peak = read_process_peak_memory()
    evaluator = week_evaluator(context, setup)
    metrics = {
        channel: setup.locked.channel(channel).d_phase_record.metric for channel in ALL_CHANNELS
    }
    ranked = {channel: table.select(CANDIDATE_COLUMNS) for channel, table in tables.items()}
    labels = label_metrics(evaluator, ranked, metrics)
    stage1 = stage1_candidates_row(paths, fold, UNION_SCOPE)
    tolerance = context.config.tolerance
    recorded_rows = int(stage1["n_rows"])
    rows = [
        Measurement(
            ITEM_UNION,
            UNION_SCOPE,
            "deduplicated_rows",
            n_rows,
            recorded_rows,
            f"relative difference <= {tolerance.max_count_relative_difference}",
            within_relative(n_rows, recorded_rows, tolerance.max_count_relative_difference),
            M6_CANDIDATES,
        ),
        recall_measurement(context, ITEM_UNION, UNION_SCOPE, labels.union_recall),
        Measurement(
            ITEM_UNION,
            UNION_SCOPE,
            "zero_hit_query_share",
            labels.zero_hit_query_share,
            float(stage1["zero_hit_query_share"]),
            source=M6_CANDIDATES,
        ),
    ]
    note = "the stage-1 monitor step; it counted the union without building the fused table"
    rows.extend(
        resource_measurements(
            ITEM_UNION,
            UNION_SCOPE,
            seconds,
            item_peak,
            stage1_run_row(paths, fold, MONITOR_STEP),
            note,
        )
    )
    return ItemOutcome(
        item=ITEM_UNION,
        scope=UNION_SCOPE,
        item_seconds=seconds,
        sampled_peak_rss_gb=sampled_peak,
        item_peak=item_peak,
        measurements=rows,
        detail=f"{n_rows} deduplicated rows from the run's own six channels",
    )
