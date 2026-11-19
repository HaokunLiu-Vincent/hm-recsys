"""One DeepFM training with its evaluations: a configuration, a seed, one V fold (proposal §16).

    1. Fold check FIRST (hmrec.ranker.window.m7_training_window): the target must
       be a V fold and every training fold an F or V fold. A T fold raises before
       any file is opened.
    2. The training rows: the M7 sample files of the window, checked against the
       LightGBM Dataset of the same target week (hmrec.deepfm.data).
    3. The input transform is fitted on those training rows only, then the rows
       are encoded and the raw values released.
    4. Training: `epochs` epochs; the model is saved after every epoch.
    5. Scoring: the target week's frozen wide table is verified, then read ONE
       part at a time; a part is encoded once and scored with the model of every
       epoch. Top-K by score DESC, article_idx ASC.
    6. Evaluation of every epoch with the existing code
       (hmrec.eval.ranker_eval): the labels of the target fold are read through
       load_m7_evaluator, which refuses every fold that is not a V fold.

Files of a trial, under artifacts/deepfm/{area}/{config_id}/seed{seed}/{fold}/:
    transform.json            the fitted bin edges or standardisation statistics
    epoch{k}.pt               the model after epoch k
    predictions.parquet       epoch, query_id, rank, article_idx, score (Top-K of every epoch)
    per_query.parquet         epoch, query_id, fold and one column per metric
    trial.json                written LAST: marks the trial as finished
Nothing is ever replaced: a folder that already exists is an error.
"""

import json
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import polars as pl
import torch

from hmrec.config import DEEPFM_UNDECIDED_DEVICE, BaseConfig, DeepFMConfig
from hmrec.data.folds import Fold, FoldManifest
from hmrec.deepfm.data import (
    FeatureColumns,
    TrainingRows,
    categorical_sizes,
    check_sample_files,
    feature_columns,
    lightgbm_sample_sha256,
    load_training_rows,
    read_candidate_part,
)
from hmrec.deepfm.inputs import InputTransform, encode_rows, fit_transform, save_transform
from hmrec.deepfm.model import DeepFM
from hmrec.deepfm.records import deepfm_config_id
from hmrec.deepfm.score import ARTICLE_IDX, QUERY_ID, RANK, SCORE, score_rows, top_k_per_query
from hmrec.deepfm.train import (
    EpochResult,
    TrainingSet,
    build_model,
    build_optimizer,
    train_epoch,
    training_set,
)
from hmrec.eval.ranker_eval import load_m7_evaluator, ranking_scores
from hmrec.feature_files import verify_feature_files
from hmrec.locks import FeaturesLock
from hmrec.logging_utils import get_logger
from hmrec.memory import log_memory
from hmrec.paths import ProjectPaths
from hmrec.ranker.ablation import wide_table_columns
from hmrec.ranker.window import m7_training_window

LOGGER = get_logger("deepfm.trial")

AREA_SPEED_TEST = "speed_test"
AREA_V_PHASE = "v_phase"
AREAS = (AREA_SPEED_TEST, AREA_V_PHASE)
EPOCH = "epoch"
DONE_FILE = "trial.json"
TRANSFORM_FILE = "transform.json"
PREDICTIONS_FILE = "predictions.parquet"
PER_QUERY_FILE = "per_query.parquet"


@dataclass(frozen=True)
class TrialInputs:
    """The configuration every DeepFM trial works with.

    Attributes:
        paths: Project paths.
        base: configs/base.yaml (threads, memory limit).
        config: configs/deepfm.yaml, possibly with a variant's values replaced.
        manifest: configs/fold_manifest.yaml.
        lock: configs/locks/features.lock.yaml (the 71 features).
    """

    paths: ProjectPaths
    base: BaseConfig
    config: DeepFMConfig
    manifest: FoldManifest
    lock: FeaturesLock


@dataclass(frozen=True)
class TrialSpec:
    """One training to run.

    Attributes:
        area: "speed_test" or "v_phase".
        target_fold: The V fold to predict.
        seed: The training seed.
        epochs: Epochs to train (a checkpoint and an evaluation after each).
    """

    area: str
    target_fold: str
    seed: int
    epochs: int


@dataclass(frozen=True)
class EpochMetrics:
    """The evaluation of the model after one epoch.

    Attributes:
        epoch: 1-based epoch number.
        n_queries: Queries of the target fold.
        scores: metric name -> mean over the fold's queries.
        train_loss: Mean training loss of the epoch.
    """

    epoch: int
    n_queries: int
    scores: dict[str, float]
    train_loss: float


@dataclass(frozen=True)
class TrialResult:
    """What a trial produced (the script adds the memory peak and the git state).

    Attributes:
        config_id: Id of the configuration.
        train_folds: The training folds.
        device: The device it ran on.
        n_train_rows: Training rows.
        n_train_queries: Training queries.
        n_scored_rows: Candidate rows of the target week (scored once per epoch model).
        prepare_seconds: Reading the sample files, fitting the transform, encoding.
        epochs: One EpochResult per epoch (steps, loss, seconds).
        score_seconds: Reading, encoding and scoring the target week with all epoch models.
        metrics: One EpochMetrics per epoch.
    """

    config_id: str
    train_folds: tuple[str, ...]
    device: str
    n_train_rows: int
    n_train_queries: int
    n_scored_rows: int
    prepare_seconds: float
    epochs: tuple[EpochResult, ...]
    score_seconds: float
    metrics: tuple[EpochMetrics, ...]


# ---------------------------------------------------------------- checks


def check_spec(inputs: TrialInputs, spec: TrialSpec) -> tuple[Fold, ...]:
    """Check a trial before anything is read and return its training folds.

    Raises:
        ValueError: If the area or the number of epochs is invalid, the device is
            still undecided, the target is not a V fold or a training fold is
            not an F / V fold (so every T fold is refused here).
    """
    if spec.area not in AREAS:
        raise ValueError(f"area must be one of {AREAS}, got {spec.area!r}")
    if spec.epochs <= 0:
        raise ValueError(f"epochs must be > 0, got {spec.epochs}")
    if inputs.config.training.device == DEEPFM_UNDECIDED_DEVICE:
        raise ValueError(
            "training.device of configs/deepfm.yaml is still undecided: run the device "
            "benchmark and write its outcome to the config first"
        )
    return m7_training_window(inputs.manifest, spec.target_fold, inputs.config.window_weeks)


def trial_dir(inputs: TrialInputs, spec: TrialSpec) -> Path:
    """The folder of a trial's files; it must not exist yet.

    Raises:
        FileExistsError: If it exists (a finished trial is never replaced; an
            unfinished one must be looked at and moved away by hand).
    """
    config_id = deepfm_config_id(inputs.config)
    directory = inputs.paths.deepfm_trial_dir(spec.area, config_id, spec.seed, spec.target_fold)
    if directory.exists():
        state = "finished" if (directory / DONE_FILE).exists() else "unfinished"
        raise FileExistsError(f"{directory} already holds a {state} trial; it is never replaced")
    return directory


# ---------------------------------------------------------------- training rows


@dataclass(frozen=True)
class PreparedTraining:
    """The training rows of a window, encoded, with the transform that encoded them.

    Attributes:
        transform: Fitted on these training rows only.
        data: The encoded rows, labels and query boundaries on the device.
        n_queries: Training queries.
    """

    transform: InputTransform
    data: TrainingSet
    n_queries: int


def load_window_rows(
    inputs: TrialInputs, target_fold: str, folds: tuple[Fold, ...], columns: FeatureColumns
) -> TrainingRows:
    """Check the window's sample files against LightGBM's Dataset record, then read them."""
    paths, config = inputs.paths, inputs.config
    ratio = config.negatives_per_positive
    sample_files = {
        fold.name: paths.ranker_sample_file(config.experiment, fold.name, ratio) for fold in folds
    }
    dataset_dir = paths.ranker_dataset_dir(config.experiment, target_fold, ratio)
    check_sample_files(sample_files, lightgbm_sample_sha256(dataset_dir))
    return load_training_rows(list(sample_files.values()), columns)


def fit_window_transform(
    inputs: TrialInputs, rows: TrainingRows, columns: FeatureColumns
) -> InputTransform:
    """Fit the input transform on the training rows of the window (and on nothing else)."""
    config = inputs.config
    sizes = categorical_sizes(inputs.paths.category_codes_file, columns.categorical)
    return fit_transform(
        rows.raw.numeric,
        columns.numeric,
        columns.categorical,
        sizes,
        config.model.numeric_input,
        config.n_bins,
        (config.clip_low_percentile, config.clip_high_percentile),
    )


def prepare_training(
    inputs: TrialInputs, target_fold: str, folds: tuple[Fold, ...], device: torch.device
) -> PreparedTraining:
    """Read, check and encode the training rows of the window that predicts `target_fold`."""
    columns = feature_columns(inputs.lock)
    rows = load_window_rows(inputs, target_fold, folds, columns)
    transform = fit_window_transform(inputs, rows, columns)
    encoded = encode_rows(rows.raw, transform)
    data = training_set(encoded, rows.labels, rows.query_ids, device)
    log_memory(LOGGER, f"{target_fold}: {rows.raw.n_rows} training rows encoded")
    return PreparedTraining(transform, data, data.boundaries.size - 1)


# ---------------------------------------------------------------- scoring and evaluation


def load_epoch_models(
    directory: Path, transform: InputTransform, inputs: TrialInputs, epochs: int
) -> dict[int, DeepFM]:
    """The saved model of every epoch, loaded from its file, in eval mode on the device."""
    device = torch.device(inputs.config.training.device)
    models = {}
    for epoch in range(1, epochs + 1):
        model = DeepFM(transform.layout, inputs.config.model).to(device)
        state = torch.load(directory / f"epoch{epoch}.pt", map_location=device, weights_only=True)
        model.load_state_dict(state)
        models[epoch] = model.eval()
    return models


def score_target_fold(
    inputs: TrialInputs, target_fold: str, transform: InputTransform, models: dict[int, DeepFM]
) -> tuple[dict[int, pl.DataFrame], int]:
    """Score ALL candidates of the target week with every epoch's model.

    The wide table is verified first. Each part is read and encoded once, then
    scored by every model, so only one part is in memory.

    Returns:
        (epoch -> Top-K table of the whole fold, number of candidate rows).
    """
    config = inputs.config
    features_dir = inputs.paths.features_dir(target_fold)
    feature_files = verify_feature_files(features_dir, wide_table_columns(inputs.lock))
    columns = feature_columns(inputs.lock)
    device = torch.device(config.training.device)
    tops: dict[int, list[pl.DataFrame]] = {epoch: [] for epoch in models}
    n_rows = 0
    for part in feature_files.parts:
        candidates = read_candidate_part(features_dir / part.name, columns)
        encoded = encode_rows(candidates.raw, transform)
        n_rows += encoded.n_rows
        for epoch, model in models.items():
            scores = score_rows(model, encoded, config.scoring_rows_per_chunk, device)
            tops[epoch].append(
                top_k_per_query(candidates.query_ids, candidates.article_idx, scores, config.top_k)
            )
        LOGGER.info("scored %s (%d rows, %d models)", part.name, encoded.n_rows, len(models))
    return {epoch: pl.concat(frames) for epoch, frames in tops.items()}, n_rows


def evaluate_and_save(
    inputs: TrialInputs,
    target: Fold,
    tops: dict[int, pl.DataFrame],
    losses: dict[int, float],
    directory: Path,
) -> tuple[EpochMetrics, ...]:
    """Evaluate every epoch's Top-K on the target V fold and write the trial's two tables.

    The labels are read through load_m7_evaluator, which checks the fold's phase
    before it opens the file.
    """
    evaluator = load_m7_evaluator(target, inputs.paths.labels_file(target.name))
    metrics, per_query, predictions = [], [], []
    for epoch in sorted(tops):
        scores = ranking_scores(evaluator, tops[epoch], inputs.config.metrics)
        metrics.append(EpochMetrics(epoch, scores.n_queries, scores.means, losses[epoch]))
        epoch_column = pl.lit(epoch, dtype=pl.Int32).alias(EPOCH)
        per_query.append(scores.per_query.with_columns(epoch_column))
        predictions.append(tops[epoch].select(epoch_column, QUERY_ID, RANK, ARTICLE_IDX, SCORE))
    pl.concat(predictions).write_parquet(directory / PREDICTIONS_FILE)
    pl.concat(per_query).write_parquet(directory / PER_QUERY_FILE)
    return tuple(metrics)


# ---------------------------------------------------------------- one trial


def train_and_save(
    inputs: TrialInputs, spec: TrialSpec, prepared: PreparedTraining, directory: Path
) -> tuple[EpochResult, ...]:
    """Train `spec.epochs` epochs and save the model after each."""
    training = replace(inputs.config.training, seed=spec.seed)
    device = torch.device(training.device)
    model = build_model(prepared.transform.layout, inputs.config.model, spec.seed, device)
    optimizer = build_optimizer(model, training)
    results = []
    for epoch in range(1, spec.epochs + 1):
        results.append(train_epoch(model, optimizer, prepared.data, training, spec.seed, epoch))
        torch.save(model.state_dict(), directory / f"epoch{epoch}.pt")
    return tuple(results)


def run_trial(inputs: TrialInputs, spec: TrialSpec) -> TrialResult:
    """Train one configuration for one V fold and evaluate it after every epoch.

    Args:
        inputs: Configuration.
        spec: The trial.

    Returns:
        The result; the caller writes the result rows.

    Raises:
        ValueError: If the fold check fails (e.g. a T fold) or a sample file is
            not the one LightGBM trained on.
        FileExistsError: If the trial's folder exists.
    """
    folds = check_spec(inputs, spec)  # before anything is read
    directory = trial_dir(inputs, spec)
    device = torch.device(inputs.config.training.device)
    torch.set_num_threads(inputs.base.num_threads)
    started = time.perf_counter()
    prepared = prepare_training(inputs, spec.target_fold, folds, device)
    directory.mkdir(parents=True)
    save_transform(prepared.transform, directory / TRANSFORM_FILE)
    prepare_seconds = time.perf_counter() - started
    n_train_rows = prepared.data.n_rows
    epochs = train_and_save(inputs, spec, prepared, directory)
    transform, n_train_queries = prepared.transform, prepared.n_queries
    # The training rows are not needed for scoring: release them first.
    del prepared
    started = time.perf_counter()
    models = load_epoch_models(directory, transform, inputs, spec.epochs)
    tops, n_scored_rows = score_target_fold(inputs, spec.target_fold, transform, models)
    losses = {result.epoch: result.mean_loss for result in epochs}
    target = inputs.manifest.get(spec.target_fold)
    metrics = evaluate_and_save(inputs, target, tops, losses, directory)
    score_seconds = time.perf_counter() - started
    result = TrialResult(
        config_id=deepfm_config_id(inputs.config),
        train_folds=tuple(fold.name for fold in folds),
        device=inputs.config.training.device,
        n_train_rows=n_train_rows,
        n_train_queries=n_train_queries,
        n_scored_rows=n_scored_rows,
        prepare_seconds=prepare_seconds,
        epochs=epochs,
        score_seconds=score_seconds,
        metrics=metrics,
    )
    summary = {"config_id": result.config_id, "seed": spec.seed, "epochs": spec.epochs}
    (directory / DONE_FILE).write_text(json.dumps(summary), encoding="utf-8")
    return result


def check_scores_are_finite(result: TrialResult) -> None:
    """Fail if a training loss is NaN or infinite (the model diverged)."""
    bad = [epoch.epoch for epoch in result.epochs if not np.isfinite(epoch.mean_loss)]
    if bad:
        raise ValueError(f"the training loss is not finite in epochs {bad}")
