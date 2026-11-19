"""One DeepFM training with its evaluation: a configuration, a seed, one V fold (proposal §16).

    1. Fold check FIRST (hmrec.ranker.window.m7_training_window): the target must
       be a V fold and every training fold an F or V fold. A T fold raises before
       any file is opened.
    2. The training rows: the M7 sample files of the window, checked against the
       LightGBM Dataset of the same target week (hmrec.deepfm.data).
    3. The input transform is fitted on those training rows only, then the rows
       are encoded and the raw values released.
    4. Training: `epochs` epochs with the configured learning-rate schedule
       (hmrec.deepfm.train). The model is saved ONCE, after the last epoch: with
       the cosine schedule a run of T epochs is its own model, and no
       mid-schedule checkpoint is kept or evaluated (proposal §16.8).
    5. Scoring: the target week's frozen wide table is verified, then read ONE
       part at a time and scored with the final model. Top-K by score DESC,
       article_idx ASC.
    6. Evaluation with the existing code (hmrec.eval.ranker_eval): the labels of
       the target fold are read through load_m7_evaluator, which refuses every
       fold that is not a V fold.

Steps 2, 3 and the encoding of step 5 give the same arrays for every training of
a target week, so they are cached on disk for the input modes listed in the
config (hmrec.deepfm.cache). The first training of a week does them and writes
the cache; the later ones read it after its sha256 check. Either way the
trial's own transform.json records the transform it used.

Files of a trial, under artifacts/deepfm/{area}/{config_id}/seed{seed}/{fold}/:
    transform.json            the fitted bin edges or standardisation statistics
    epoch{T}.pt               the model after the last epoch T
    predictions.parquet       epoch, query_id, rank, article_idx, score (Top-K; epoch = T)
    per_query.parquet         epoch, query_id, fold and one column per metric
    trial.json                written LAST: marks the trial as finished; it also
                              holds the mean training loss of every epoch
Nothing is ever replaced: a folder that already exists is an error.
"""

import json
import time
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch

from hmrec.config import DEEPFM_UNDECIDED_DEVICE, BaseConfig, DeepFMConfig
from hmrec.data.folds import Fold, FoldManifest
from hmrec.deepfm.cache import (
    CachedPart,
    CachedTraining,
    CacheReader,
    CacheWriter,
    open_cache,
    source_key,
)
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
    build_scheduler,
    count_steps,
    train_epoch,
    training_set,
)
from hmrec.eval.ranker_eval import load_m7_evaluator, ranking_scores
from hmrec.feature_files import METADATA_FILE, verify_feature_files
from hmrec.locks import FeaturesLock, file_sha256
from hmrec.logging_utils import get_logger
from hmrec.memory import log_memory
from hmrec.paths import ProjectPaths
from hmrec.ranker.ablation import wide_table_columns
from hmrec.ranker.sampling import read_sample_record, record_file
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
# Part of the cache key: changes whenever the way inputs are encoded changes.
ENCODING_VERSION = "2026-11-20 signed_log1p"


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
        epochs: Epochs to train = the schedule length; it must equal
            training.epochs of the trial's configuration (the configuration id
            of a cosine run depends on it).
    """

    area: str
    target_fold: str
    seed: int
    epochs: int


@dataclass(frozen=True)
class EpochMetrics:
    """The evaluation of the final model.

    Attributes:
        epoch: The last epoch of the run (1-based) = its number of epochs.
        n_queries: Queries of the target fold.
        scores: metric name -> mean over the fold's queries.
        train_loss: Mean training loss of that last epoch.
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
        n_scored_rows: Candidate rows of the target week.
        prepare_seconds: Getting the encoded training rows (from the sample files
            or from the cache).
        epochs: One EpochResult per epoch (steps, loss, seconds).
        score_seconds: Scoring and evaluating the target week with the final model.
        metrics: The evaluation of the final model (exactly one entry).
        input_cache: "written", "read" or "off": how the encoded inputs were obtained.
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
    input_cache: str


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
    if spec.epochs != inputs.config.training.epochs:
        raise ValueError(
            f"the trial trains {spec.epochs} epochs but its configuration says "
            f"{inputs.config.training.epochs}: the two must be equal"
        )
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


def window_sample_files(inputs: TrialInputs, folds: tuple[Fold, ...]) -> dict[str, Path]:
    """Training fold -> its M7 sample file of the configured ratio, in time order."""
    paths, config = inputs.paths, inputs.config
    ratio = config.negatives_per_positive
    return {
        fold.name: paths.ranker_sample_file(config.experiment, fold.name, ratio) for fold in folds
    }


def load_window_rows(
    inputs: TrialInputs, target_fold: str, folds: tuple[Fold, ...], columns: FeatureColumns
) -> TrainingRows:
    """Check the window's sample files against LightGBM's Dataset record, then read them."""
    paths, config = inputs.paths, inputs.config
    sample_files = window_sample_files(inputs, folds)
    dataset_dir = paths.ranker_dataset_dir(
        config.experiment, target_fold, config.negatives_per_positive
    )
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


def encode_window(inputs: TrialInputs, target_fold: str, folds: tuple[Fold, ...]) -> CachedTraining:
    """Read, check and encode the training rows of the window that predicts `target_fold`."""
    columns = feature_columns(inputs.lock)
    rows = load_window_rows(inputs, target_fold, folds, columns)
    transform = fit_window_transform(inputs, rows, columns)
    encoded = encode_rows(rows.raw, transform)
    log_memory(LOGGER, f"{target_fold}: {rows.raw.n_rows} training rows encoded")
    return CachedTraining(transform, encoded, rows.labels, rows.query_ids)


def encoded_parts(
    inputs: TrialInputs, target_fold: str, transform: InputTransform
) -> Iterator[CachedPart]:
    """The target week's candidates, one part at a time, encoded with `transform`.

    The frozen wide table is verified before the first part is read.
    """
    features_dir = inputs.paths.features_dir(target_fold)
    feature_files = verify_feature_files(features_dir, wide_table_columns(inputs.lock))
    columns = feature_columns(inputs.lock)
    for part in feature_files.parts:
        candidates = read_candidate_part(features_dir / part.name, columns)
        encoded = encode_rows(candidates.raw, transform)
        yield CachedPart(part.name, encoded, candidates.query_ids, candidates.article_idx)


# ---------------------------------------------------------------- the input cache


def cache_source(inputs: TrialInputs, target_fold: str, folds: tuple[Fold, ...]) -> dict[str, Any]:
    """Everything the encoded inputs of a target week are computed from (the cache key).

    Only small record files are read here. The sample files' sha256 come from
    their own records and must equal the ones LightGBM's Dataset recorded, so a
    cache can only belong to the rows LightGBM trained on.

    Raises:
        ValueError: If a sample record does not name the file LightGBM trained on.
    """
    paths, config = inputs.paths, inputs.config
    columns = feature_columns(inputs.lock)
    dataset_dir = paths.ranker_dataset_dir(
        config.experiment, target_fold, config.negatives_per_positive
    )
    lightgbm_sha256 = lightgbm_sample_sha256(dataset_dir)
    sample_sha256 = {
        fold: read_sample_record(record_file(file)).sha256
        for fold, file in window_sample_files(inputs, folds).items()
    }
    if sample_sha256 != lightgbm_sha256:
        raise ValueError(
            f"{target_fold}: a sample file is not the one LightGBM trained on: the sample "
            f"records say {sample_sha256}, the LightGBM Dataset {lightgbm_sha256}"
        )
    return {
        "encoding": ENCODING_VERSION,
        "experiment": config.experiment,
        "negatives_per_positive": config.negatives_per_positive,
        "target_fold": target_fold,
        "train_folds": [fold.name for fold in folds],
        "sample_sha256": sample_sha256,
        "features_record_sha256": file_sha256(paths.features_dir(target_fold) / METADATA_FILE),
        "numeric": list(columns.numeric),
        "categorical": list(columns.categorical),
        "categorical_sizes": list(
            categorical_sizes(paths.category_codes_file, columns.categorical)
        ),
        "numeric_input": config.model.numeric_input,
        "n_bins": config.n_bins,
        "clip_percentiles": [config.clip_low_percentile, config.clip_high_percentile],
    }


def cache_dir(inputs: TrialInputs, target_fold: str, source: dict[str, Any]) -> Path:
    """The cache folder of a source: artifacts/deepfm/cache/{target}_{mode}_{key}/."""
    return inputs.paths.deepfm_cache_dir(
        target_fold, inputs.config.model.numeric_input, source_key(source)
    )


def write_cache(
    inputs: TrialInputs, target_fold: str, folds: tuple[Fold, ...], directory: Path
) -> None:
    """Encode the training rows and all candidates of a target week and write them as a cache.

    The sample files and the wide table are fully verified (sha256) while they
    are read, exactly as in a training without a cache.
    """
    training = encode_window(inputs, target_fold, folds)
    # The folder is created only now: a sample file that fails its check leaves nothing behind.
    writer = CacheWriter(directory, cache_source(inputs, target_fold, folds))
    writer.write_training(training)
    transform = training.transform
    # The training rows are not needed for the candidates: release them first.
    del training
    for part in encoded_parts(inputs, target_fold, transform):
        writer.write_part(part)
        LOGGER.info("cached %s (%d rows)", part.name, part.rows.n_rows)
    writer.finish()
    LOGGER.info("input cache written -> %s", directory)


def open_input_cache(
    inputs: TrialInputs, target_fold: str, folds: tuple[Fold, ...]
) -> tuple[CacheReader | None, str]:
    """The verified input cache of a target week, written first if it does not exist.

    Returns:
        (reader, "read" or "written"), or (None, "off") if the configuration's
        input mode is not cached.
    """
    if inputs.config.model.numeric_input not in inputs.config.cached_numeric_inputs:
        return None, "off"
    source = cache_source(inputs, target_fold, folds)
    directory = cache_dir(inputs, target_fold, source)
    reader = open_cache(directory, source)
    if reader is not None:
        LOGGER.info("input cache verified and read: %s", directory)
        return reader, "read"
    write_cache(inputs, target_fold, folds, directory)
    reader = open_cache(directory, source)
    if reader is None:
        raise ValueError(f"{directory} does not exist right after it was written")
    return reader, "written"


def prepare_training(
    inputs: TrialInputs,
    target_fold: str,
    folds: tuple[Fold, ...],
    device: torch.device,
    reader: CacheReader | None = None,
) -> PreparedTraining:
    """The encoded training rows of the window on the device, from the cache if there is one."""
    if reader is None:
        training = encode_window(inputs, target_fold, folds)
    else:
        training = reader.read_training()
    data = training_set(training.rows, training.labels, training.query_ids, device)
    return PreparedTraining(training.transform, data, data.boundaries.size - 1)


# ---------------------------------------------------------------- scoring and evaluation


def load_final_model(
    directory: Path, transform: InputTransform, inputs: TrialInputs, epochs: int
) -> DeepFM:
    """The saved model of the last epoch, loaded from its file, in eval mode on the device."""
    device = torch.device(inputs.config.training.device)
    model = DeepFM(transform.layout, inputs.config.model).to(device)
    state = torch.load(directory / f"epoch{epochs}.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model.eval()


def score_target_fold(
    inputs: TrialInputs,
    target_fold: str,
    transform: InputTransform,
    model: DeepFM,
    reader: CacheReader | None = None,
) -> tuple[pl.DataFrame, int]:
    """Score ALL candidates of the target week with the final model.

    One part is in memory at a time. Without a cache the wide table is verified
    and each part is read and encoded here; with a cache the encoded parts are
    read (they were verified when the cache was opened).

    Returns:
        (the Top-K table of the whole fold, number of candidate rows).
    """
    config = inputs.config
    device = torch.device(config.training.device)
    if reader is None:
        parts = encoded_parts(inputs, target_fold, transform)
    else:
        parts = (reader.read_part(name) for name in reader.parts)
    tops: list[pl.DataFrame] = []
    n_rows = 0
    for part in parts:
        scores = score_rows(model, part.rows, config.scoring_rows_per_chunk, device)
        tops.append(top_k_per_query(part.query_ids, part.article_idx, scores, config.top_k))
        n_rows += part.rows.n_rows
        LOGGER.info("scored %s (%d rows)", part.name, part.rows.n_rows)
    return pl.concat(tops), n_rows


def evaluate_and_save(
    inputs: TrialInputs,
    target: Fold,
    top: pl.DataFrame,
    last_epoch: EpochResult,
    directory: Path,
) -> tuple[EpochMetrics, ...]:
    """Evaluate the final model's Top-K on the target V fold and write the trial's two tables.

    The labels are read through load_m7_evaluator, which checks the fold's phase
    before it opens the file.
    """
    evaluator = load_m7_evaluator(target, inputs.paths.labels_file(target.name))
    scores = ranking_scores(evaluator, top, inputs.config.metrics)
    epoch_column = pl.lit(last_epoch.epoch, dtype=pl.Int32).alias(EPOCH)
    top.select(epoch_column, QUERY_ID, RANK, ARTICLE_IDX, SCORE).write_parquet(
        directory / PREDICTIONS_FILE
    )
    scores.per_query.with_columns(epoch_column).write_parquet(directory / PER_QUERY_FILE)
    metrics = EpochMetrics(last_epoch.epoch, scores.n_queries, scores.means, last_epoch.mean_loss)
    return (metrics,)


# ---------------------------------------------------------------- one trial


def train_and_save(
    inputs: TrialInputs, spec: TrialSpec, prepared: PreparedTraining, directory: Path
) -> tuple[EpochResult, ...]:
    """Train `spec.epochs` epochs with the run's schedule and save the final model."""
    training = replace(inputs.config.training, seed=spec.seed)
    device = torch.device(training.device)
    model = build_model(prepared.transform.layout, inputs.config.model, spec.seed, device)
    optimizer = build_optimizer(model, training)
    total_steps = count_steps(prepared.data, training, spec.seed, spec.epochs)
    scheduler = build_scheduler(optimizer, training, total_steps)
    LOGGER.info(
        "%d epochs, %d optimizer steps, lr schedule %s",
        spec.epochs,
        total_steps,
        training.lr_schedule,
    )
    results = []
    for epoch in range(1, spec.epochs + 1):
        results.append(
            train_epoch(model, optimizer, prepared.data, training, spec.seed, epoch, scheduler)
        )
    if sum(result.steps for result in results) != total_steps:
        raise ValueError("the run took another number of steps than its schedule was built for")
    torch.save(model.state_dict(), directory / f"epoch{spec.epochs}.pt")
    return tuple(results)


def run_trial(inputs: TrialInputs, spec: TrialSpec) -> TrialResult:
    """Train one configuration for one V fold and evaluate its final model.

    Args:
        inputs: Configuration.
        spec: The trial.

    Returns:
        The result; the caller writes the result rows.

    Raises:
        ValueError: If the fold check fails (e.g. a T fold), a sample file is not
            the one LightGBM trained on, or the input cache fails its check.
        FileExistsError: If the trial's folder exists.
    """
    folds = check_spec(inputs, spec)  # before anything is read
    directory = trial_dir(inputs, spec)
    device = torch.device(inputs.config.training.device)
    torch.set_num_threads(inputs.base.num_threads)
    started = time.perf_counter()
    reader, cache_state = open_input_cache(inputs, spec.target_fold, folds)
    prepared = prepare_training(inputs, spec.target_fold, folds, device, reader)
    directory.mkdir(parents=True)
    save_transform(prepared.transform, directory / TRANSFORM_FILE)
    prepare_seconds = time.perf_counter() - started
    n_train_rows = prepared.data.n_rows
    epochs = train_and_save(inputs, spec, prepared, directory)
    transform, n_train_queries = prepared.transform, prepared.n_queries
    # The training rows are not needed for scoring: release them first.
    del prepared
    started = time.perf_counter()
    model = load_final_model(directory, transform, inputs, spec.epochs)
    top, n_scored_rows = score_target_fold(inputs, spec.target_fold, transform, model, reader)
    target = inputs.manifest.get(spec.target_fold)
    metrics = evaluate_and_save(inputs, target, top, epochs[-1], directory)
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
        input_cache=cache_state,
    )
    summary = {
        "config_id": result.config_id,
        "seed": spec.seed,
        "epochs": spec.epochs,
        "input_cache": cache_state,
        "epoch_mean_loss": {str(epoch.epoch): epoch.mean_loss for epoch in epochs},
        "epoch_steps": {str(epoch.epoch): epoch.steps for epoch in epochs},
    }
    (directory / DONE_FILE).write_text(json.dumps(summary), encoding="utf-8")
    return result


def check_scores_are_finite(result: TrialResult) -> None:
    """Fail if a training loss is NaN or infinite (the model diverged)."""
    bad = [epoch.epoch for epoch in result.epochs if not np.isfinite(epoch.mean_loss)]
    if bad:
        raise ValueError(f"the training loss is not finite in epochs {bad}")
