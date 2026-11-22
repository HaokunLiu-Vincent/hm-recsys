"""The Ranker steps of a tick: C2 evaluate, C3 sample, O3 dataset and train, O4 score.

This module is on the LightGBM side; it must not import torch.

NO new computation happens here. Every step calls the functions M7 and M8 ran
(hmrec.ranker.sampling, hmrec.hpo.v_trial) with the locked Ranker configuration
and the tick's paths (hmrec.paths.PipelinePaths).

Where the configuration comes from:

    Linux (the container)   hmrec.ranker.platform_entry, mode "accepted": the
                            process must be the platform a record under
                            configs/platforms/ describes (docs/mlops.md §8.3).
    macOS (the laptop)      hmrec.ranker.locked.load_locked_ranker, the strict
                            entry, as in stage 1. The other entry refuses macOS.

A model that predicts a week is trained from scratch on the 4 weeks before it
(the locked window), which must all be closed: their samples exist. F1-F4 come
from the bootstrap state, every later week from step C3 of the tick that closed it.

Labels: C2 and C3 read the answers of the week the tick closes. They are built
from the tick's ingested transactions (hmrec.weekly.data_steps) and read through
hmrec.eval.pipeline_labels, which checks the clock. Nothing is selected or tuned.
"""

import json
import platform
import shutil
import sys
from dataclasses import asdict, dataclass

import polars as pl

from hmrec.eval.hit_value import hit_value
from hmrec.eval.monitor import label_metrics
from hmrec.eval.pipeline_labels import load_closed_week_evaluator, require_week_is_over
from hmrec.eval.ranker_eval import ranking_scores
from hmrec.eval.report import FoldEvaluator
from hmrec.feature_files import read_feature_files
from hmrec.hpo.v_trial import (
    DatasetPlan,
    TrialInputs,
    TrialSpec,
    build_and_save_dataset,
    check_trial_settings,
    plan_dataset_for_folds,
    rrf_baseline_metric_row,
    score_fold,
    train_planned_model,
)
from hmrec.locks import (
    ALL_CHANNELS,
    R6_CHANNEL,
    LockedChannel,
    check_file_sha256,
    load_channels_lock,
    load_features_lock,
    load_twotower_lock,
)
from hmrec.logging_utils import get_logger
from hmrec.paths import ProjectPaths
from hmrec.ranker.ablation import build_experiment, wide_table_columns
from hmrec.ranker.identity import model_config, ranker_config_id
from hmrec.ranker.locked import LockedRanker, load_locked_ranker
from hmrec.ranker.platform_entry import (
    MODE_ACCEPTED,
    MODE_CALIBRATION,
    load_locked_ranker_for_platform,
)
from hmrec.ranker.runtime import LightGBMRuntime, read_lightgbm_runtime
from hmrec.ranker.sampling import SampleJob, SampleRecord, sample_fold
from hmrec.ranker.train import first_trees_sha256, load_model, save_model
from hmrec.run_manifest import MetricRecord
from hmrec.weekly.calendar import weeks_before
from hmrec.weekly.context import OutputFile, TickContext
from hmrec.weekly.data_steps import ClosedWeekAnswers, closed_week_answers
from hmrec.weekly.outputs import (
    ARTICLE_IDX,
    CUSTOMER_IDX,
    PREDICTION_FILE,
    QUERY_ID,
    RANK,
    TOP_COLUMNS,
    WeekEvaluation,
    signal_metrics,
    write_checked_top,
    write_week_evaluation,
)
from hmrec.weekly.shard_features import FEATURE_STATS_FILE, QUERY_STATS_FILE
from hmrec.weekly.steps import Step

LOGGER = get_logger("weekly.ranker_steps")

# The `area` of the pipeline's trials (hmrec.hpo.v_trial.TrialSpec). It is not "v_phase":
# nothing is tuned here.
AREA_WEEKLY = "weekly"
DARWIN = "darwin"
CANDIDATE_COLUMNS = [QUERY_ID, ARTICLE_IDX, RANK]
RRF_PREFIX = "rrf_baseline"

StepOutcome = tuple[str, list[MetricRecord], list[OutputFile]]


@dataclass(frozen=True)
class RankerSetup:
    """What every Ranker step works with.

    Attributes:
        locked: The locked Ranker configuration (already checked by its entry).
        inputs: The same configuration in the shape the shared trial code takes;
            inputs.paths are the tick's paths.
        runtime: The LightGBM version and OpenMP runtime of this process.
    """

    locked: LockedRanker
    inputs: TrialInputs
    runtime: LightGBMRuntime

    @property
    def ratio(self) -> int:
        """The locked negatives-per-positive ratio."""
        return self.locked.tuned.negatives_per_positive


def load_locked_for_this_platform(context: TickContext, runtime: LightGBMRuntime) -> LockedRanker:
    """The locked Ranker configuration through the entry of this platform (module docstring).

    A calibration tick (docs/mlops.md §8.3) measures a platform BEFORE it has a
    record: on Linux it uses mode "calibration", which records the OpenMP
    runtime it finds and accepts only the calibration state as output location.
    On macOS the strict entry is the only one, also for a calibration tick.

    Raises:
        hmrec.ranker.platform_entry.PlatformNotAccepted: On Linux, in mode
            "accepted": the platform has no record or the process is not what
            its record describes.
        ValueError: If a lock check fails, or a calibration tick would write
            outside the calibration state.
    """
    if sys.platform == DARWIN:
        return load_locked_ranker(context.paths, context.base, runtime)
    mode = MODE_CALIBRATION if context.calibration else MODE_ACCEPTED
    entered = load_locked_ranker_for_platform(
        context.paths,
        context.base,
        runtime,
        sys.platform,
        platform.machine(),
        mode,
        context.view,
    )
    return entered.locked


def load_ranker_setup(context: TickContext) -> RankerSetup:
    """Get the locked configuration for this platform and wrap it for the trial code."""
    runtime = read_lightgbm_runtime()
    locked = load_locked_for_this_platform(context, runtime)
    inputs = TrialInputs(
        paths=context.view,
        base=context.base,
        ranker=locked.config,
        manifest=context.manifest,
        lock=load_features_lock(context.paths.features_lock_file),
        metrics=context.pipeline.metrics,
    )
    return RankerSetup(locked, inputs, runtime)


# ---------------------------------------------------------------- C2: evaluate the closed week


def locked_channels(paths: ProjectPaths) -> dict[str, LockedChannel]:
    """Channel -> its lock (quota, selection metric), in the order R1 ... R6.

    Read with hmrec.locks only: hmrec.recall.locked would load torch.
    """
    channels = dict(load_channels_lock(paths.channels_lock_file).channels)
    channels[R6_CHANNEL] = load_twotower_lock(paths.twotower_lock_file).channel
    return {channel: channels[channel] for channel in ALL_CHANNELS}


def published_top(context: TickContext, answers: ClosedWeekAnswers) -> pl.DataFrame:
    """Last week's PUBLISHED predictions as a ranked table by query.

    Read from the prediction set predictions/{anchor}/ (what was put out), not
    from the scoring step's own file.

    Returns:
        One row per (query_id, rank): query_id, rank, article_idx.

    Raises:
        ValueError: If a published customer is not a query of the closed week.
    """
    folder = context.layout.prediction_set_dir(context.closing.anchor)
    published = pl.read_parquet(folder / PREDICTION_FILE)
    ranked = published.join(
        answers.queries.select(QUERY_ID, CUSTOMER_IDX), on=CUSTOMER_IDX, how="left"
    )
    if ranked[QUERY_ID].null_count() > 0:
        raise ValueError(
            f"the prediction set of {context.closing.name} holds customers that are not "
            "queries of the week"
        )
    return ranked.select(QUERY_ID, RANK, ARTICLE_IDX).sort(QUERY_ID, RANK)


def evaluate_closed_week(
    context: TickContext,
    setup: RankerSetup,
    evaluator: FoldEvaluator,
    answers: ClosedWeekAnswers,
) -> tuple[WeekEvaluation, pl.DataFrame]:
    """Measure the closed week: recall of the channels, the Ranker, the RRF baseline, hit value.

    Returns:
        (the content of evaluation/metrics.json, the Ranker's per-query metric table).
    """
    closing, view = context.closing, context.view
    channels = locked_channels(context.paths)
    candidates = {
        channel: pl.read_parquet(
            view.candidates_file(closing.name, channel), columns=CANDIDATE_COLUMNS
        )
        for channel in ALL_CHANNELS
    }
    recall_metrics = {channel: lock.d_phase_record.metric for channel, lock in channels.items()}
    recall = label_metrics(evaluator, candidates, recall_metrics)
    top = published_top(context, answers)
    ranker = ranking_scores(evaluator, top, context.pipeline.metrics)
    baseline = rrf_baseline_metric_row(setup.inputs, closing.name, evaluator)
    evaluation = WeekEvaluation(
        week=closing.name,
        anchor=closing.anchor.isoformat(),
        n_queries=evaluator.n_queries,
        n_label_pairs=evaluator.labels.height,
        channel_recall_metric=recall_metrics,
        channel_recall=recall.channel_recall,
        union_recall=recall.union_recall,
        zero_hit_query_share=recall.zero_hit_query_share,
        ranker=ranker.means,
        rrf_baseline=dict(baseline.scores),
        hit_value=hit_value(top, answers.queries, answers.purchase_values),
    )
    return evaluation, ranker.per_query


def evaluation_metrics(context: TickContext, evaluation: WeekEvaluation) -> list[MetricRecord]:
    """The numbers of evaluation/metrics.json as manifest metrics of the closed week."""
    week = context.closing.anchor
    metrics = [
        MetricRecord("queries", week, float(evaluation.n_queries)),
        MetricRecord("union_recall", week, evaluation.union_recall),
        MetricRecord("hit_value", week, evaluation.hit_value),
    ]
    for channel, value in evaluation.channel_recall.items():
        metrics.append(MetricRecord(f"recall/{channel}", week, value))
    for name, value in evaluation.ranker.items():
        metrics.append(MetricRecord(name, week, value))
    for name, value in evaluation.rrf_baseline.items():
        metrics.append(MetricRecord(f"{RRF_PREFIX}/{name}", week, value))
    return metrics


def run_evaluate(context: TickContext, setup: RankerSetup) -> StepOutcome:
    """C2: score last week's published predictions against last week's purchases.

    Writes evaluation/labels.parquet (the week's label sets, built from the
    tick's ingested transactions), metrics.json and per_query.parquet.

    Raises:
        ValueError: If the week is not over at the tick's anchor, or the query
            list written when the week was opened is not the one its purchases give.
    """
    closing, view = context.closing, context.view
    require_week_is_over(closing.fold, context.tick.anchor)
    answers = closed_week_answers(context)
    labels_file = view.labels_file(closing.name)
    labels_file.parent.mkdir(parents=True, exist_ok=True)
    answers.labels.write_parquet(labels_file)
    evaluator = load_closed_week_evaluator(closing.fold, context.tick.anchor, labels_file)
    evaluation, per_query = evaluate_closed_week(context, setup, evaluator, answers)
    metrics_file = view.evaluation_file(closing.name)
    write_week_evaluation(metrics_file, evaluation)
    per_query.write_parquet(metrics_file.with_name("per_query.parquet"))
    first_metric = context.pipeline.metrics[0]
    detail = (
        f"{closing.name}: {evaluation.n_queries} queries, union recall "
        f"{evaluation.union_recall:.6f}, {first_metric} {evaluation.ranker[first_metric]:.6f} "
        f"(RRF baseline {evaluation.rrf_baseline[first_metric]:.6f}), hit value "
        f"{evaluation.hit_value:.4f}"
    )
    LOGGER.info("C2: %s", detail)
    files = [
        OutputFile("closed_week/labels", labels_file.name),
        OutputFile("closed_week/metrics", metrics_file.name),
    ]
    return detail, evaluation_metrics(context, evaluation), files


# ---------------------------------------------------------------- C3: sample the closed week


def check_sample_is_plausible(record: SampleRecord, ratio: int, n_week_queries: int) -> None:
    """Fail if the sample's row counts cannot be right (docs/mlops.md §6.1, C3).

    Every sampled query has at least one positive row, and at most `ratio`
    negatives per positive (proposal §9.1).

    Raises:
        ValueError: With the count that is impossible.
    """
    problems = []
    if not 0 < record.n_queries <= n_week_queries:
        problems.append(f"{record.n_queries} sampled queries of {n_week_queries} queries")
    if record.n_positives < record.n_queries:
        problems.append(f"{record.n_positives} positives for {record.n_queries} queries")
    if not record.n_positives < record.n_rows <= record.n_positives * (ratio + 1):
        problems.append(f"{record.n_rows} rows for {record.n_positives} positives at ratio {ratio}")
    if problems:
        raise ValueError(f"sample of {record.fold} is not plausible: {'; '.join(problems)}")


def run_sample(context: TickContext, setup: RankerSetup) -> StepOutcome:
    """C3: the Ranker training sample of the closed week, with the locked seed and ratio.

    Raises:
        ValueError: If the week is not over, or the sample's counts are not plausible.
    """
    closing, view, lock = context.closing, context.view, setup.inputs.lock
    require_week_is_over(closing.fold, context.tick.anchor)
    experiment = build_experiment(context.ranker_experiment, lock)
    sample_file = view.ranker_sample_file(experiment.name, closing.name, setup.ratio)
    job = SampleJob(
        fold=closing.fold,
        features_dir=view.features_dir(closing.name),
        labels_file=view.labels_file(closing.name),
        experiment=experiment,
        wide_columns=wide_table_columns(lock),
        sample_files={setup.ratio: sample_file},
        # The clock was checked above; the phase guard gets the week's own phase.
        allowed_phases=(closing.fold.phase,),
        git_commit=context.stamp.git_commit,
    )
    record = sample_fold(job, setup.locked.config.sampling)[setup.ratio]
    n_week_queries = read_feature_files(view.features_dir(closing.name)).n_queries
    check_sample_is_plausible(record, setup.ratio, n_week_queries)
    detail = (
        f"{closing.name}: {record.n_rows} rows, {record.n_queries} queries, "
        f"{record.n_positives} positives, ratio {setup.ratio}, sha256 {record.sha256}"
    )
    metrics = [
        MetricRecord("sample/rows", closing.anchor, float(record.n_rows)),
        MetricRecord("sample/queries", closing.anchor, float(record.n_queries)),
        MetricRecord("sample/positives", closing.anchor, float(record.n_positives)),
    ]
    files = [
        OutputFile("closed_week/sample", sample_file.name),
        OutputFile("closed_week/sample_record", sample_file.with_suffix(".json").name),
    ]
    return detail, metrics, files


# ---------------------------------------------------------------- O3: Dataset and model


def dataset_plan(context: TickContext, setup: RankerSetup) -> DatasetPlan:
    """The Dataset of the model that predicts the tick's week: the 4 weeks before it.

    Raises:
        FileNotFoundError: If a training week has no finished sample (it is not closed).
        ValueError: If a sample has other columns or another seed than the lock.
    """
    window = weeks_before(context.tick, context.manifest, setup.locked.config.window_weeks)
    experiment = context.ranker_experiment
    return plan_dataset_for_folds(
        setup.inputs,
        experiment,
        context.tick.name,
        tuple(week.fold for week in window),
        setup.ratio,
        context.stamp.git_commit,
        context.view.ranker_dataset_dir(experiment, context.tick.name, setup.ratio),
    )


def run_dataset(context: TickContext, setup: RankerSetup) -> StepOutcome:
    """O3 (first part): build and save the binary Dataset (sample files checked by sha256)."""
    plan = dataset_plan(context, setup)
    record = build_and_save_dataset(plan, setup.locked.config.sampling.row_group_rows)
    detail = (
        f"training weeks {' '.join(record.train_folds)}: {record.n_rows} rows, "
        f"{record.n_queries} queries"
    )
    metrics = [
        MetricRecord("ranker/train_rows", context.tick.anchor, float(record.n_rows)),
        MetricRecord("ranker/train_queries", context.tick.anchor, float(record.n_queries)),
    ]
    return detail, metrics, [OutputFile("ranker_dataset_record", "dataset.json")]


def trial_spec(context: TickContext, setup: RankerSetup) -> TrialSpec:
    """The one trial of a tick: the locked tuned values, trees and histogram mode."""
    locked = setup.locked
    return TrialSpec(
        area=AREA_WEEKLY,
        experiment=context.ranker_experiment,
        step=context.run_id,
        target_fold=context.tick.name,
        tuned=locked.tuned,
        num_trees=locked.num_trees,
        histogram_mode=locked.histogram_mode,
        evaluate=True,
    )


def run_train(context: TickContext, setup: RankerSetup) -> StepOutcome:
    """O3 (second part): train the locked configuration from scratch and save the model.

    Raises:
        ValueError: If the locked settings do not give the locked config_id.
        FileExistsError: If the model file exists (it is never replaced).
    """
    locked, inputs, week = setup.locked, setup.inputs, context.tick
    experiment = context.ranker_experiment
    spec = trial_spec(context, setup)
    check_trial_settings(inputs.ranker, spec)
    settings = model_config(inputs.ranker, locked.tuned, experiment, locked.histogram_mode)
    if ranker_config_id(settings) != locked.config_id:
        raise ValueError(
            f"the locked settings give config_id {ranker_config_id(settings)}, the lock "
            f"records {locked.config_id}"
        )
    booster, outcome, _ = train_planned_model(inputs, spec, dataset_plan(context, setup), settings)
    model_file = context.view.ranker_model_file(
        AREA_WEEKLY, experiment, locked.config_id, week.name
    )
    model_sha256 = save_model(booster, model_file)
    record = {
        "week": week.name,
        "sha256": model_sha256,
        # Read from the SAVED file, so the hash describes what a later step loads.
        "trees_sha256": first_trees_sha256(load_model(model_file), locked.num_trees),
        "config_id": locked.config_id,
        "num_trees": locked.num_trees,
        "train_folds": list(outcome.train_folds),
        "n_train_rows": outcome.n_train_rows,
        "n_train_queries": outcome.n_train_queries,
        "train_seconds": outcome.train_seconds,
        **asdict(setup.runtime),
    }
    record_file = context.view.model_record_file(week.name)
    record_file.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    detail = (
        f"config {locked.config_id}, {locked.num_trees} trees on {' '.join(outcome.train_folds)} "
        f"({outcome.n_train_rows} rows), {outcome.train_seconds:.1f} s, sha256 {model_sha256}"
    )
    metrics = [MetricRecord("ranker/train_seconds", week.anchor, outcome.train_seconds)]
    files = [
        OutputFile("ranker_model", model_file.name),
        OutputFile("ranker_model_record", record_file.name),
    ]
    return detail, metrics, files


# ---------------------------------------------------------------- O4: score


def score_with_week_model(
    context: TickContext, setup: RankerSetup, features_view: ProjectPaths
) -> tuple[pl.DataFrame, int]:
    """Score all candidates of a week (or a shard) with the week's saved model.

    The model is loaded from its file after the sha256 of its record was checked;
    the frozen wide table is verified and scored by hmrec.hpo.v_trial.score_fold,
    the function M7 and M8 scored a fold with. Only the Top-K at the locked
    number of trees is kept.

    Args:
        context: The process's context.
        setup: The locked configuration.
        features_view: The paths whose features_dir holds the wide table: the
            tick's paths, or the paths of one shard.

    Returns:
        (Top-K: query_id, rank, article_idx, score; number of candidate rows scored).
    """
    view, week, locked = context.view, context.tick.name, setup.locked
    model_file = view.ranker_model_file(
        AREA_WEEKLY, context.ranker_experiment, locked.config_id, week
    )
    record = json.loads(view.model_record_file(week).read_text(encoding="utf-8"))
    check_file_sha256(model_file, record["sha256"])
    experiment = build_experiment(context.ranker_experiment, setup.inputs.lock)
    tops, n_rows = score_fold(
        load_model(model_file),
        features_view.features_dir(week),
        experiment,
        setup.inputs,
        locked.num_trees,
    )
    return tops[locked.num_trees].select(TOP_COLUMNS).sort(QUERY_ID, RANK), n_rows


def run_score(context: TickContext, setup: RankerSetup) -> StepOutcome:
    """O4 in `eval` mode: the Top-K of every query of the week, checked, with the output signals.

    Raises:
        ValueError: If a query does not have top_k distinct valid articles.
    """
    top, n_rows = score_with_week_model(context, setup, context.view)
    signals, files = write_checked_top(context, top, n_rows)
    detail = (
        f"scored {n_rows} candidates; Top-{setup.locked.config.top_k} of {signals.n_queries} "
        f"queries; {signals.distinct_articles} distinct articles"
    )
    return detail, signal_metrics(context.tick.anchor, signals), files


def run_shard_score(context: TickContext, setup: RankerSetup, step: Step) -> StepOutcome:
    """O4 in `serve` mode: the Top-K of one shard.

    Also copies the shard's two small statistics files next to the Top-K: the
    shard's features are removed after this step (docs/mlops.md §8.4).

    Raises:
        ValueError: If the step has no shard.
    """
    if step.shard is None:
        raise ValueError(f"step {step.name} has no shard")
    week = context.tick.name
    shard_view = context.view.for_shard(step.shard)
    top, n_rows = score_with_week_model(context, setup, shard_view)
    top_file = shard_view.top_file(week)
    top_file.parent.mkdir(parents=True, exist_ok=True)
    top.write_parquet(top_file)
    for name in (QUERY_STATS_FILE, FEATURE_STATS_FILE):
        shutil.copyfile(shard_view.features_dir(week) / name, top_file.with_name(name))
    detail = f"shard {step.shard}: scored {n_rows} candidates of {top[QUERY_ID].n_unique()} queries"
    return detail, [], []
