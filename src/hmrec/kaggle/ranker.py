"""The ranker of a Kaggle run: training labels, sample, Dataset, model, Top-K of every shard.

The configuration comes ONLY from hmrec.ranker.locked.load_locked_ranker
(configs/locks/ranker.lock.yaml); the steps are the ones M7 and M8 ran
(hmrec.ranker.sampling, hmrec.hpo.v_trial), so nothing is tuned or selected here.

The model is trained from scratch on the 4 folds before the predicted week
(hmrec.kaggle.plan.ranker_training_folds):

    submission   T1 T2 T3 T4. T1-T3: the sample files M8 wrote, reused. T4: sampled
                 now with the locked seed and ratio, from T4's frozen wide table.
    rehearsal    V4 T1 T2 T3: the four sample files M8 trained its T4 model on,
                 all reused. No label file is opened.

A reused sample file is used only if its sha256 is the one recorded in the
Dataset record of M8's T4 model (artifacts/lgb_datasets/t_phase/full/T4_r{ratio}/
dataset.json): that record is the proof of which files M8 trained on.

Labels: the T folds' labels are TRAINING labels here (the user's consent of
2026-11-19, docs/decisions.md). They are read in two places: the label file of a
newly sampled fold, and the label column of the sample files when the Dataset is
built. Each of these steps first appends a row to results/kaggle_label_access.csv
(record_label_access). No label is used to evaluate anything.

This module imports LightGBM and must never be imported together with torch
(tests/unit/test_kaggle_code_rules.py).
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import polars as pl

from hmrec.csv_records import append_row
from hmrec.data.folds import Fold
from hmrec.hpo.v_records import AREA_T_PHASE
from hmrec.hpo.v_trial import (
    DatasetPlan,
    TrialInputs,
    TrialSpec,
    build_and_save_dataset,
    check_trial_settings,
    score_fold,
    train_planned_model,
)
from hmrec.kaggle.context import KaggleContext
from hmrec.kaggle.plan import LAST_PHASE, RUN_REHEARSAL, ranker_training_folds, read_queries
from hmrec.kaggle.rehearsal import RehearsalOutcome, compare_top_lists
from hmrec.locks import file_sha256, load_features_lock
from hmrec.logging_utils import get_logger
from hmrec.ranker.ablation import ARTICLE_IDX, FULL, QUERY_ID, build_experiment, wide_table_columns
from hmrec.ranker.dataset import DatasetRecord, DatasetSource, read_dataset_record
from hmrec.ranker.identity import model_config, ranker_config_id
from hmrec.ranker.locked import LockedRanker, load_locked_ranker
from hmrec.ranker.predict import RANK, SCORE
from hmrec.ranker.runtime import read_lightgbm_runtime
from hmrec.ranker.sampling import (
    LABEL,
    SampleJob,
    SampleRecord,
    read_sample_record,
    record_file,
    sample_fold,
)
from hmrec.ranker.train import dataset_params, first_trees_sha256, load_model, save_model

LOGGER = get_logger("kaggle.ranker")

AREA_KAGGLE = "kaggle"
LABEL_ACCESS_COLUMNS: tuple[str, ...] = (
    "recorded_at",
    "git_commit",
    "git_dirty",
    "run",
    "folds",
    "purpose",
    "script",
)
TOP_COLUMNS = [QUERY_ID, RANK, ARTICLE_IDX, SCORE]


@dataclass(frozen=True)
class RankerContext:
    """What the ranker steps of a Kaggle run work with.

    Attributes:
        context: The process's context.
        locked: The locked ranker configuration (already checked).
        inputs: The same configuration in the shape the shared trial code takes.
        training_folds: The folds the model is trained on, in time order.
    """

    context: KaggleContext
    locked: LockedRanker
    inputs: TrialInputs
    training_folds: tuple[Fold, ...]

    @property
    def ratio(self) -> int:
        """The locked negatives-per-positive ratio."""
        return self.locked.tuned.negatives_per_positive


def load_ranker_context(context: KaggleContext) -> RankerContext:
    """Read the ranker lock (with all its checks) and work out the training folds.

    Raises:
        ValueError: If the ranker lock fails a check (config id, referenced lock
            files, LightGBM / OpenMP version, thread count).
    """
    paths, base = context.paths, context.base
    locked = load_locked_ranker(paths, base, read_lightgbm_runtime())
    inputs = TrialInputs(
        paths=paths,
        base=base,
        ranker=locked.config,
        manifest=context.manifest,
        lock=load_features_lock(paths.features_lock_file),
        metrics=(),
    )
    folds = ranker_training_folds(
        context.run, context.config, context.manifest, locked.config.window_weeks
    )
    return RankerContext(context, locked, inputs, folds)


# ---------------------------------------------------------------- label access record


def record_label_access(
    ranker: RankerContext, folds: tuple[str, ...], purpose: str, script: str
) -> None:
    """Append a row to results/kaggle_label_access.csv BEFORE training labels are read.

    Args:
        ranker: The ranker context (run and code state).
        folds: The folds whose labels the step is about to read.
        purpose: What they are read for, in a few words.
        script: The script that reads them, e.g. "scripts/kaggle_ranker.py --step sample".
    """
    stamp = ranker.context.stamp
    row = {
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": stamp.git_commit,
        "git_dirty": str(stamp.git_dirty),
        "run": ranker.context.run,
        "folds": " ".join(folds),
        "purpose": purpose,
        "script": script,
    }
    append_row(ranker.context.paths.kaggle_label_access_file, LABEL_ACCESS_COLUMNS, row)
    LOGGER.info("label access recorded: %s", row)


# ---------------------------------------------------------------- sample files


def m8_sample_sha256(ranker: RankerContext) -> dict[str, str]:
    """Fold -> sha256 of the sample file M8 trained its last model (target T4) on."""
    last_fold = ranker.context.manifest.folds[-1].name
    directory = ranker.context.paths.final_dataset_dir(AREA_T_PHASE, FULL, last_fold, ranker.ratio)
    return dict(read_dataset_record(directory).sample_sha256)


def new_sample_folds(ranker: RankerContext) -> tuple[Fold, ...]:
    """The training folds this run samples itself: those M8 has no sample file of (T4)."""
    recorded = m8_sample_sha256(ranker)
    return tuple(fold for fold in ranker.training_folds if fold.name not in recorded)


def sample_file_of(ranker: RankerContext, fold: Fold) -> Path:
    """The sample file of a training fold: M8's file, or this run's own for a new fold."""
    paths, run = ranker.context.paths, ranker.context.run
    if fold in new_sample_folds(ranker):
        return paths.kaggle_sample_file(run, fold.name, ranker.ratio)
    return paths.ranker_sample_file(FULL, fold.name, ranker.ratio)


def check_reused_sample(ranker: RankerContext, fold: Fold) -> None:
    """Fail unless a reused sample file is the one M8 trained on (sha256).

    Raises:
        FileNotFoundError: If the file is missing.
        ValueError: If its sha256 differs from the one in M8's Dataset record.
    """
    recorded = m8_sample_sha256(ranker)[fold.name]
    path = sample_file_of(ranker, fold)
    if not path.is_file():
        raise FileNotFoundError(f"reused sample file not found: {path}")
    actual = file_sha256(path)
    if actual != recorded:
        raise ValueError(
            f"{path}: sha256 is {actual}, but M8 trained on {recorded}. This is not the M8 "
            "sample file; it is not used."
        )


def sample_new_fold(ranker: RankerContext, fold: Fold, script: str) -> SampleRecord:
    """Draw the training sample of a fold M8 did not sample (T4), with the locked seed and ratio.

    Reads the fold's label file as TRAINING labels; the read is recorded first.
    The rows come from the fold's frozen wide table (artifacts/features/{fold}/),
    verified before it is read (hmrec.ranker.sampling.sample_fold).

    Raises:
        ValueError: If the fold is not one this run samples, or not a T fold.
    """
    if fold not in new_sample_folds(ranker):
        raise ValueError(f"fold {fold.name} is not sampled by this run: M8's sample file is reused")
    if fold.phase != LAST_PHASE:
        raise ValueError(f"only {LAST_PHASE} folds are sampled here, got {fold.name}")
    paths, lock = ranker.context.paths, ranker.inputs.lock
    job = SampleJob(
        fold=fold,
        features_dir=paths.features_dir(fold.name),
        labels_file=paths.labels_file(fold.name),
        experiment=build_experiment(FULL, lock),
        wide_columns=wide_table_columns(lock),
        sample_files={ranker.ratio: sample_file_of(ranker, fold)},
        allowed_phases=(LAST_PHASE,),
        git_commit=ranker.context.stamp.git_commit,
    )
    purpose = f"ranker training labels: negative sampling of {fold.name} (label file)"
    record_label_access(ranker, (fold.name,), purpose, script)
    return sample_fold(job, ranker.locked.config.sampling)[ranker.ratio]


# ---------------------------------------------------------------- Dataset and model


def dataset_plan(ranker: RankerContext) -> DatasetPlan:
    """The run's Dataset: the training folds' sample files, in the run's own folder.

    The same plan hmrec.hpo.v_trial.plan_dataset_for_folds makes, except that
    the sample file of a new fold is the run's own.

    Raises:
        FileNotFoundError: If a sample is not finished.
        ValueError: If a sample has other columns or another seed than the lock.
    """
    inputs, context = ranker.inputs, ranker.context
    experiment = build_experiment(FULL, inputs.lock)
    expected_columns = (QUERY_ID, ARTICLE_IDX, LABEL, *experiment.feature_names)
    sources = []
    for fold in ranker.training_folds:
        sample_file = sample_file_of(ranker, fold)
        record = read_sample_record(record_file(sample_file))
        if record.columns != expected_columns or record.seed != inputs.ranker.sampling.seed:
            raise ValueError(
                f"{sample_file}: the sample has other columns or another seed than the lock"
            )
        sources.append(DatasetSource(fold.name, sample_file, record.n_rows, record.sha256))
    params = dataset_params(inputs.ranker, inputs.base.num_threads)
    record = DatasetRecord(
        experiment=FULL,
        target_fold=context.target.name,
        ratio=ranker.ratio,
        train_folds=tuple(fold.name for fold in ranker.training_folds),
        sample_sha256={source.fold: source.sha256 for source in sources},
        n_rows=sum(source.n_rows for source in sources),
        n_queries=0,
        feature_names=experiment.feature_names,
        categorical_names=experiment.categorical_names,
        params=params,
        binary_sha256="",
        git_commit=context.stamp.git_commit,
    )
    directory = context.paths.kaggle_dataset_dir(context.run)
    return DatasetPlan(directory, experiment, tuple(sources), params, record)


def build_run_dataset(ranker: RankerContext, script: str) -> DatasetRecord:
    """Check the reused sample files against M8's record, then build and save the Dataset.

    Building reads the label column of every sample file (training labels), so
    the read is recorded first.
    """
    new_folds = new_sample_folds(ranker)
    for fold in ranker.training_folds:
        if fold not in new_folds:
            check_reused_sample(ranker, fold)
    plan = dataset_plan(ranker)
    names = tuple(fold.name for fold in ranker.training_folds)
    purpose = "ranker training labels: label column of the sample files (Dataset build)"
    record_label_access(ranker, names, purpose, script)
    return build_and_save_dataset(plan, ranker.locked.config.sampling.row_group_rows)


@dataclass(frozen=True)
class TrainedModel:
    """What training the run's model gave.

    Attributes:
        model_sha256: sha256 of the saved model file.
        trees_sha256: sha256 of the text of the model's trees (not of its header).
        n_train_rows: Rows of the training Dataset.
        n_train_queries: Queries of the training Dataset.
        train_seconds: Seconds of training.
    """

    model_sha256: str
    trees_sha256: str
    n_train_rows: int
    n_train_queries: int
    train_seconds: float


def train_run_model(ranker: RankerContext) -> TrainedModel:
    """Train the locked configuration from scratch on the run's Dataset and save the model.

    Raises:
        ValueError: If the locked settings do not give the locked config_id.
        FileExistsError: If the model file exists (it is never replaced).
    """
    inputs, locked, context = ranker.inputs, ranker.locked, ranker.context
    spec = TrialSpec(
        area=AREA_KAGGLE,
        experiment=FULL,
        step=context.run,
        target_fold=context.target.name,
        tuned=locked.tuned,
        num_trees=locked.num_trees,
        histogram_mode=locked.histogram_mode,
        evaluate=True,
    )
    check_trial_settings(inputs.ranker, spec)
    settings = model_config(inputs.ranker, locked.tuned, FULL, locked.histogram_mode)
    if ranker_config_id(settings) != locked.config_id:
        raise ValueError(
            f"the locked settings give config_id {ranker_config_id(settings)}, the lock "
            f"records {locked.config_id}"
        )
    model_file = context.paths.kaggle_model_file(context.run)
    if model_file.exists():
        raise FileExistsError(f"{model_file} already exists; models are never replaced")
    booster, outcome, _ = train_planned_model(inputs, spec, dataset_plan(ranker), settings)
    model_sha256 = save_model(booster, model_file)
    return TrainedModel(
        model_sha256=model_sha256,
        trees_sha256=saved_trees_sha256(model_file, locked.num_trees),
        n_train_rows=outcome.n_train_rows,
        n_train_queries=outcome.n_train_queries,
        train_seconds=outcome.train_seconds,
    )


def saved_trees_sha256(model_file: Path, num_trees: int) -> str:
    """sha256 of the first `num_trees` trees of a SAVED model (hmrec.ranker.train).

    Both models of the rehearsal's comparison are read from their files with
    this function, so the two texts are produced in the same way.
    """
    return first_trees_sha256(load_model(model_file), num_trees)


def m8_trees_sha256(ranker: RankerContext) -> str:
    """sha256 of the trees of the model M8 trained for the rehearsal's fold (T4).

    Raises:
        ValueError: If the run is not the rehearsal (only then is there an M8 model
            of the predicted week).
    """
    context, locked = ranker.context, ranker.locked
    if context.run != RUN_REHEARSAL:
        raise ValueError("only the rehearsal's week has a model trained by M8")
    model_file = context.paths.ranker_model_file(
        AREA_T_PHASE, FULL, locked.config_id, context.target.name
    )
    return saved_trees_sha256(model_file, locked.num_trees)


# ---------------------------------------------------------------- scoring


def load_run_model(ranker: RankerContext) -> lgb.Booster:
    """The run's saved model."""
    return load_model(ranker.context.paths.kaggle_model_file(ranker.context.run))


def score_shard(ranker: RankerContext, booster: lgb.Booster, shard: int) -> int:
    """Score ALL candidates of one shard and write the Top-K of every query.

    The shard's wide table is verified and scored by hmrec.hpo.v_trial.score_fold,
    the function M7 and M8 scored a fold with; only the Top-K at the locked number
    of trees is kept. Output columns: query_id (int32), rank (int32, 1 = best),
    article_idx (int32), score (float64); sorted by (query_id, rank).

    Returns:
        The number of candidate rows scored; 0 if the shard's file existed already.

    Raises:
        FileExistsError: Never; an existing file is left untouched and skipped.
    """
    context, locked = ranker.context, ranker.locked
    top_file = context.paths.kaggle_top_file(context.run, shard)
    if top_file.exists():
        LOGGER.warning("%s shard %d: %s exists; SKIPPED", context.run, shard, top_file)
        return 0
    experiment = build_experiment(FULL, ranker.inputs.lock)
    features_dir = context.paths.kaggle_features_dir(context.run, shard)
    tops, n_rows = score_fold(booster, features_dir, experiment, ranker.inputs, locked.num_trees)
    top = tops[locked.num_trees].select(TOP_COLUMNS).sort(QUERY_ID, RANK)
    top_file.parent.mkdir(parents=True, exist_ok=True)
    top.write_parquet(top_file)
    LOGGER.info(
        "%s shard %d: scored %d rows, Top-%d of %d queries -> %s",
        context.run,
        shard,
        n_rows,
        locked.config.top_k,
        top[QUERY_ID].n_unique(),
        top_file,
    )
    return n_rows


def read_top(path: Path) -> pl.DataFrame:
    """Read a shard's Top-K file written by score_shard."""
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found: the shard is not scored yet")
    return pl.read_parquet(path, columns=TOP_COLUMNS)


# ---------------------------------------------------------------- the rehearsal's checks


def m8_top(ranker: RankerContext) -> pl.DataFrame:
    """The Top-K M8 saved for the rehearsal's fold at the locked number of trees.

    Returns:
        query_id, rank, article_idx; one row per (query, rank).
    """
    context, locked = ranker.context, ranker.locked
    file = context.paths.ranker_predictions_file(
        AREA_T_PHASE, FULL, locked.config_id, context.target.name
    )
    saved = pl.read_parquet(file)
    at_locked_trees = saved.filter(pl.col("num_trees") == locked.num_trees)
    if at_locked_trees.height == 0:
        raise ValueError(f"{file} has no rows at {locked.num_trees} trees")
    return at_locked_trees.select(QUERY_ID, RANK, ARTICLE_IDX)


def rehearsal_outcome(
    ranker: RankerContext, new_top: pl.DataFrame, n_shards: int, format_checks_passed: bool
) -> RehearsalOutcome:
    """Compare a finished rehearsal with M8: the trees and the Top-K lists of the fold's queries.

    Args:
        ranker: The ranker context of the rehearsal.
        new_top: The rehearsal's Top-K of all its customers: query_id, rank, article_idx.
        n_shards: Shards the rehearsal ran in.
        format_checks_passed: The recorded outcome of the submission-format checks.

    Raises:
        ValueError: If the run is not the rehearsal.
    """
    context = ranker.context
    if context.run != RUN_REHEARSAL:
        raise ValueError("only a rehearsal is compared with M8")
    paths = context.paths
    new_queries = read_queries(paths.kaggle_queries_file(context.run))
    fold_queries = pl.read_parquet(
        paths.queries_file(context.target.name), columns=[QUERY_ID, "customer_idx"]
    )
    comparison = compare_top_lists(new_top, new_queries, m8_top(ranker), fold_queries)
    return RehearsalOutcome(
        n_customers=new_queries.height,
        n_shards=n_shards,
        new_trees_sha256=saved_trees_sha256(
            paths.kaggle_model_file(context.run), ranker.locked.num_trees
        ),
        m8_trees_sha256=m8_trees_sha256(ranker),
        comparison=comparison,
        min_identical_share=context.config.rehearsal.min_identical_top12_share,
        format_checks_passed=format_checks_passed,
    )
