"""One Ranker trial of M7: train on 4 weeks, score one V fold, evaluate (proposal §9).

A trial = one configuration on ONE target fold, run in its own process
(scripts/m7_train_eval.py), so the process's max RSS is the trial's true memory
peak. Also here: the plan of a binary Dataset (shared by the build script and
the trial, so both mean the same Dataset) and the RRF baseline of a fold.

Every function that takes a fold checks it FIRST (hmrec.ranker.window): a T fold
raises before any file is opened.

Outputs of an evaluated trial, written in this order (never replaced):
    artifacts/predictions/{area}/{experiment}/{config_id}/{fold}.parquet
        the Top-K of every query at every checkpoint: num_trees, query_id, rank,
        article_idx, score; sorted by (num_trees, query_id, rank)
    .../{fold}_per_query.parquet
        every query's metric values at every checkpoint: query_id, fold,
        num_trees, then one column per metric
    artifacts/models/ranker/{area}/{experiment}/{config_id}/{fold}/model.txt
The rows of the results tables are written by the script afterwards; the trial
row is written last and marks the trial as done.
"""

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import lightgbm as lgb
import polars as pl

from hmrec.config import BaseConfig, RankerConfig, RankerTunedParams
from hmrec.data.folds import FoldManifest
from hmrec.eval.ranker_eval import load_m7_evaluator, ranking_scores
from hmrec.eval.report import FoldEvaluator
from hmrec.feature_files import verify_feature_files
from hmrec.fusion.rrf import BEST_RANK, RRF_SCORE, rrf_baseline
from hmrec.hpo.v_records import AREA_V_PHASE, NO_TREES, RRF_BASELINE, MetricRow
from hmrec.locks import FeaturesLock
from hmrec.logging_utils import get_logger
from hmrec.paths import ProjectPaths
from hmrec.ranker.ablation import (
    ARTICLE_IDX,
    QUERY_ID,
    Experiment,
    build_experiment,
    wide_table_columns,
)
from hmrec.ranker.dataset import (
    DatasetRecord,
    DatasetSource,
    build_dataset,
    load_dataset,
    read_dataset_record,
    save_dataset,
)
from hmrec.ranker.identity import model_config, ranker_config_id
from hmrec.ranker.predict import RANK, SCORE, query_chunks, score_rows
from hmrec.ranker.sampling import LABEL, read_sample_record, record_file, verify_sample_file
from hmrec.ranker.train import (
    dataset_params,
    save_model,
    train_ranker,
    training_params,
)
from hmrec.ranker.window import m7_training_window, require_m7_target_fold

LOGGER = get_logger("hpo.v_trial")

NUM_TREES = "num_trees"


@dataclass(frozen=True)
class TrialInputs:
    """The configuration every M7 step works with.

    Attributes:
        paths: Project paths.
        base: configs/base.yaml (threads, memory limit).
        ranker: configs/ranker.yaml.
        manifest: configs/fold_manifest.yaml.
        lock: configs/locks/features.lock.yaml (feature list, rrf_k).
        metrics: The metrics to compute, the selection metric first
            (configs/hpo/v_phase.yaml), e.g. ("map@12", "recall@12", "ndcg@12").
    """

    paths: ProjectPaths
    base: BaseConfig
    ranker: RankerConfig
    manifest: FoldManifest
    lock: FeaturesLock
    metrics: tuple[str, ...]


@dataclass(frozen=True)
class TrialSpec:
    """One trial to run.

    Attributes:
        area: "v_phase" (the grid) or "speed_test".
        experiment: "full" or "minus_r6".
        step: Name of the step, written to the trial row.
        target_fold: The V fold to predict.
        tuned: num_leaves, min_data_in_leaf, negatives_per_positive.
        num_trees: Trees to train (the last checkpoint).
        histogram_mode: "col_wise" or "row_wise".
        evaluate: False = train only and time it (no scoring, no label is read,
            nothing is saved).
    """

    area: str
    experiment: str
    step: str
    target_fold: str
    tuned: RankerTunedParams
    num_trees: int
    histogram_mode: str
    evaluate: bool


# ---------------------------------------------------------------- the binary Dataset


@dataclass(frozen=True)
class DatasetPlan:
    """What a (experiment, target fold, ratio) Dataset is built from.

    Attributes:
        directory: Where train.bin and dataset.json go.
        experiment: The experiment's feature list.
        sources: The training folds' sample files in time order.
        params: The LightGBM parameters that define the bins.
        record: The record the Dataset must have (binary_sha256 is empty here).
    """

    directory: Path
    experiment: Experiment
    sources: tuple[DatasetSource, ...]
    params: dict[str, Any]
    record: DatasetRecord


def plan_dataset(
    inputs: TrialInputs, experiment_name: str, target_fold: str, ratio: int, git_commit: str
) -> DatasetPlan:
    """Describe the Dataset of a target fold from the CURRENT sample records and config.

    The fold window is checked first (M7: target V, training folds F / V).

    Raises:
        FileNotFoundError: If a training fold's sample is not finished.
        ValueError: If a sample was drawn with another seed or has other columns.
    """
    folds = m7_training_window(inputs.manifest, target_fold, inputs.ranker.window_weeks)
    experiment = build_experiment(experiment_name, inputs.lock)
    expected_columns = (QUERY_ID, ARTICLE_IDX, LABEL, *experiment.feature_names)
    sources = []
    for fold in folds:
        sample_file = inputs.paths.ranker_sample_file(experiment_name, fold.name, ratio)
        record = read_sample_record(record_file(sample_file))
        if record.columns != expected_columns or record.seed != inputs.ranker.sampling.seed:
            raise ValueError(
                f"{sample_file}: the sample has other columns or another seed than the config"
            )
        sources.append(DatasetSource(fold.name, sample_file, record.n_rows, record.sha256))
    params = dataset_params(inputs.ranker, inputs.base.num_threads)
    record = DatasetRecord(
        experiment=experiment_name,
        target_fold=target_fold,
        ratio=ratio,
        train_folds=tuple(fold.name for fold in folds),
        sample_sha256={source.fold: source.sha256 for source in sources},
        n_rows=sum(source.n_rows for source in sources),
        n_queries=0,
        feature_names=experiment.feature_names,
        categorical_names=experiment.categorical_names,
        params=params,
        binary_sha256="",
        git_commit=git_commit,
    )
    directory = inputs.paths.ranker_dataset_dir(experiment_name, target_fold, ratio)
    return DatasetPlan(directory, experiment, tuple(sources), params, record)


def build_and_save_dataset(plan: DatasetPlan, block_rows: int) -> DatasetRecord:
    """Build the planned Dataset from the sample files and save it as a binary file.

    Every sample file's sha256 is checked against its record first.

    Returns:
        The written record (with the number of queries and the binary's sha256).
    """
    for source in plan.sources:
        verify_sample_file(source.sample_file)
    dataset, n_queries = build_dataset(
        plan.sources,
        plan.experiment.feature_names,
        plan.experiment.categorical_names,
        plan.params,
        block_rows,
    )
    return save_dataset(dataset, plan.directory, replace(plan.record, n_queries=n_queries))


def load_planned_dataset(
    plan: DatasetPlan, train_params: Mapping[str, Any]
) -> tuple[lgb.Dataset, DatasetRecord]:
    """Load the binary Dataset of a plan after checking it against the plan."""
    written = read_dataset_record(plan.directory)
    # The number of queries is only known after building; everything else must match.
    expected = replace(plan.record, n_queries=written.n_queries)
    return load_dataset(plan.directory, expected, train_params), written


# ---------------------------------------------------------------- scoring and evaluation


def score_fold(
    booster: lgb.Booster,
    features_dir: Path,
    experiment: Experiment,
    inputs: TrialInputs,
    num_trees: int,
) -> tuple[dict[int, pl.DataFrame], int]:
    """Score ALL candidates of a fold (no sampling) and keep the Top-K at every checkpoint.

    The frozen wide table is verified first, then read one part file at a time
    (a part holds whole queries), so only one part is in memory. A part is scored
    in pieces of `scoring_query_chunk_size` whole queries: the float32 feature
    matrix and its copies exist for one piece only (memory only, same scores).

    Returns:
        (k -> Top-K table of the whole fold, number of candidate rows scored).
    """
    feature_files = verify_feature_files(features_dir, wide_table_columns(inputs.lock))
    tables: dict[int, list[pl.DataFrame]] = {}
    n_rows = 0
    for part in feature_files.parts:
        rows = experiment.apply(pl.read_parquet(features_dir / part.name))
        n_rows += rows.height
        for chunk in query_chunks(rows, inputs.ranker.scoring_query_chunk_size):
            tops = score_rows(
                booster,
                chunk,
                experiment.feature_names,
                num_trees,
                inputs.ranker.checkpoint_every,
                inputs.ranker.top_k,
                inputs.base.num_threads,
            )
            for k, top in tops.items():
                tables.setdefault(k, []).append(top)
        LOGGER.info("scored %s (%d rows)", part.name, rows.height)
    return {k: pl.concat(frames) for k, frames in tables.items()}, n_rows


@dataclass(frozen=True)
class FoldEvaluation:
    """The evaluation of one model on one fold at every checkpoint.

    Attributes:
        metric_rows: One MetricRow per checkpoint, ascending.
        per_query: query_id, fold, num_trees, then one column per metric.
        predictions: num_trees, query_id, rank, article_idx, score.
    """

    metric_rows: tuple[MetricRow, ...]
    per_query: pl.DataFrame
    predictions: pl.DataFrame


def evaluate_checkpoints(
    evaluator: FoldEvaluator,
    tops: Mapping[int, pl.DataFrame],
    metrics: Sequence[str],
    experiment: str,
    config_id: str,
    train_metric: Mapping[int, float],
) -> FoldEvaluation:
    """Compute the metrics of every checkpoint's Top-K table."""
    metric_rows = []
    per_query = []
    predictions = []
    for k in sorted(tops):
        scores = ranking_scores(evaluator, tops[k], metrics)
        metric_rows.append(
            MetricRow(
                experiment=experiment,
                config_id=config_id,
                fold=evaluator.fold,
                num_trees=k,
                n_queries=scores.n_queries,
                scores=scores.means,
                train_metric=train_metric.get(k),
            )
        )
        tree_column = pl.lit(k, dtype=pl.Int32).alias(NUM_TREES)
        per_query.append(scores.per_query.with_columns(tree_column))
        predictions.append(tops[k].select(tree_column, QUERY_ID, RANK, ARTICLE_IDX, SCORE))
    return FoldEvaluation(tuple(metric_rows), pl.concat(per_query), pl.concat(predictions))


# ---------------------------------------------------------------- one trial


@dataclass(frozen=True)
class TrialOutcome:
    """What a trial produced (the script adds time, memory and git state).

    Attributes:
        config_id: The model's configuration id.
        config: The configuration the id was computed from.
        train_folds: The training folds.
        n_train_rows: Rows of the training Dataset.
        n_train_queries: Queries of the training Dataset.
        n_scored_rows: Candidate rows scored (0 if not evaluated).
        n_queries: Queries of the target fold (0 if not evaluated).
        load_seconds: Seconds to verify and load the Dataset.
        train_seconds: Seconds of training.
        seconds_per_tree: Mean seconds per tree after the warm-up trees.
        score_seconds: Seconds of scoring and evaluation.
        model_sha256: sha256 of the saved model ("" if not saved).
        metric_rows: The fold's metric rows, one per checkpoint.
    """

    config_id: str
    config: dict[str, Any]
    train_folds: tuple[str, ...]
    n_train_rows: int
    n_train_queries: int
    n_scored_rows: int
    n_queries: int
    load_seconds: float
    train_seconds: float
    seconds_per_tree: float
    score_seconds: float
    model_sha256: str
    metric_rows: tuple[MetricRow, ...]


def _output_files(inputs: TrialInputs, spec: TrialSpec, config_id: str) -> tuple[Path, Path, Path]:
    """(predictions, per-query, model) files of a trial; none of them may exist yet."""
    key = (spec.area, spec.experiment, config_id, spec.target_fold)
    files = (
        inputs.paths.ranker_predictions_file(*key),
        inputs.paths.ranker_per_query_file(*key),
        inputs.paths.ranker_model_file(*key),
    )
    existing = [str(path) for path in files if path.exists()]
    if existing:
        raise FileExistsError(
            f"outputs of this trial already exist: {existing}. An earlier run stopped before "
            "its row was written; move the files away by hand, then run again."
        )
    return files


def _train(
    inputs: TrialInputs, spec: TrialSpec, plan: DatasetPlan, config: dict[str, Any]
) -> tuple[lgb.Booster, TrialOutcome, dict[int, float]]:
    """Load the Dataset and train; returns the model, the outcome so far and the train metric."""
    params = training_params(
        inputs.ranker, spec.tuned, spec.histogram_mode, inputs.base.num_threads
    )
    started = time.perf_counter()
    dataset, dataset_record = load_planned_dataset(plan, params)
    load_seconds = time.perf_counter() - started
    booster, log = train_ranker(dataset, params, spec.num_trees, inputs.ranker.train_metric_every)
    # The binned training data is not needed for scoring: release it first.
    del dataset
    booster.free_dataset()
    warmup = min(inputs.ranker.histogram.warmup_trees, spec.num_trees - 1)
    outcome = TrialOutcome(
        config_id=ranker_config_id(config),
        config=config,
        train_folds=dataset_record.train_folds,
        n_train_rows=dataset_record.n_rows,
        n_train_queries=dataset_record.n_queries,
        n_scored_rows=0,
        n_queries=0,
        load_seconds=load_seconds,
        train_seconds=log.total_seconds,
        seconds_per_tree=log.seconds_per_tree(warmup),
        score_seconds=0.0,
        model_sha256="",
        metric_rows=(),
    )
    return booster, outcome, log.train_metric


def _score_and_save(
    inputs: TrialInputs,
    spec: TrialSpec,
    experiment: Experiment,
    booster: lgb.Booster,
    outcome: TrialOutcome,
    train_metric: Mapping[int, float],
    files: tuple[Path, Path, Path],
) -> TrialOutcome:
    """Score the target fold, evaluate every checkpoint and write the trial's files."""
    started = time.perf_counter()
    target = inputs.manifest.get(spec.target_fold)
    evaluator = load_m7_evaluator(target, inputs.paths.labels_file(spec.target_fold))
    features_dir = inputs.paths.features_dir(spec.target_fold)
    tops, n_scored_rows = score_fold(booster, features_dir, experiment, inputs, spec.num_trees)
    evaluation = evaluate_checkpoints(
        evaluator, tops, inputs.metrics, spec.experiment, outcome.config_id, train_metric
    )
    predictions_file, per_query_file, model_file = files
    predictions_file.parent.mkdir(parents=True, exist_ok=True)
    evaluation.predictions.write_parquet(predictions_file)
    evaluation.per_query.write_parquet(per_query_file)
    model_sha256 = save_model(booster, model_file)
    return replace(
        outcome,
        n_scored_rows=n_scored_rows,
        n_queries=evaluator.n_queries,
        score_seconds=time.perf_counter() - started,
        model_sha256=model_sha256,
        metric_rows=evaluation.metric_rows,
    )


def check_trial_settings(ranker: RankerConfig, spec: TrialSpec) -> None:
    """Refuse a trial whose settings the config does not allow.

    A grid trial (area "v_phase") must use the histogram mode of
    configs/ranker.yaml, which must be decided; only the speed test may choose
    a mode itself. No trial trains more trees than num_trees, and an evaluated
    trial ends on a checkpoint.

    Raises:
        ValueError: On any violation.
    """
    if spec.area == AREA_V_PHASE and spec.histogram_mode != ranker.histogram.mode:
        raise ValueError(
            f"a grid trial must use histogram.mode of configs/ranker.yaml "
            f"({ranker.histogram.mode!r}), got {spec.histogram_mode!r}"
        )
    if not 0 < spec.num_trees <= ranker.num_trees:
        raise ValueError(f"num_trees must be in 1..{ranker.num_trees}, got {spec.num_trees}")
    # Only an evaluated trial needs checkpoints; a timing run may train any number of trees.
    if spec.evaluate and spec.num_trees % ranker.checkpoint_every != 0:
        raise ValueError(
            f"an evaluated trial needs num_trees to be a multiple of {ranker.checkpoint_every}, "
            f"got {spec.num_trees}"
        )
    if spec.tuned.negatives_per_positive not in ranker.sampling.ratios:
        raise ValueError(f"ratio {spec.tuned.negatives_per_positive} is not sampled")


def run_trial(inputs: TrialInputs, spec: TrialSpec, git_commit: str) -> TrialOutcome:
    """Train one model and, unless spec.evaluate is False, score and evaluate its fold.

    Args:
        inputs: Configuration.
        spec: The trial.
        git_commit: HEAD of this run.

    Returns:
        The outcome; the caller writes the result rows.
    """
    require_m7_target_fold(inputs.manifest.get(spec.target_fold))  # before anything is read
    check_trial_settings(inputs.ranker, spec)
    ratio = spec.tuned.negatives_per_positive
    plan = plan_dataset(inputs, spec.experiment, spec.target_fold, ratio, git_commit)
    config = model_config(inputs.ranker, spec.tuned, spec.experiment, spec.histogram_mode)
    files = _output_files(inputs, spec, ranker_config_id(config)) if spec.evaluate else None
    booster, outcome, train_metric = _train(inputs, spec, plan, config)
    if files is None:
        return outcome
    return _score_and_save(inputs, spec, plan.experiment, booster, outcome, train_metric, files)


# ---------------------------------------------------------------- RRF baseline


def rrf_baseline_row(inputs: TrialInputs, fold_name: str) -> MetricRow:
    """The RRF baseline of one V fold (proposal §9.5): no model, the same candidates.

    Every query's Top-K by rrf_score DESC, best_rank ASC, article_idx ASC
    (hmrec.fusion.rrf.rrf_baseline), evaluated with the same metrics as the Ranker.
    """
    fold = inputs.manifest.get(fold_name)
    evaluator = load_m7_evaluator(fold, inputs.paths.labels_file(fold_name))  # checks the phase
    features_dir = inputs.paths.features_dir(fold_name)
    feature_files = verify_feature_files(features_dir, wide_table_columns(inputs.lock))
    columns = [QUERY_ID, ARTICLE_IDX, RRF_SCORE, BEST_RANK]
    tops = [
        rrf_baseline(
            pl.read_parquet(features_dir / part.name, columns=columns), inputs.ranker.top_k
        )
        for part in feature_files.parts
    ]
    scores = ranking_scores(evaluator, pl.concat(tops), inputs.metrics)
    return MetricRow(
        experiment=RRF_BASELINE,
        config_id=RRF_BASELINE,
        fold=fold_name,
        num_trees=NO_TREES,
        n_queries=scores.n_queries,
        scores=scores.means,
        train_metric=None,
    )
