"""The result tables of the DeepFM study and the recorded reference scores (proposal §16.4).

    results/deepfm_benchmark.csv   one row per device run of the device benchmark
    results/deepfm_trials.csv      one row per training (time, true memory peak, device)
    results/deepfm_metrics.csv     one row per (configuration, seed, fold, epoch); from
                                   2026-11-20 a run is evaluated after its last epoch only

Rows are only appended (hmrec.csv_records). Floats are written with repr().

The reference scores (the RRF baseline and the locked LightGBM ranker on a V
fold) are READ from results/v_phase_metrics.csv; they are never recomputed
here. That table is read as plain CSV, because the module that owns it
(hmrec.hpo.v_records) imports LightGBM.
"""

import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from hmrec.config import DEEPFM_LR_CONSTANT, DeepFMConfig
from hmrec.locks import load_ranker_lock
from hmrec.paths import ProjectPaths
from hmrec.run_records import RunStamp

CONFIG_ID_LENGTH = 12
RRF_BASELINE = "rrf_baseline"
FULL_EXPERIMENT = "full"

BENCHMARK_COLUMNS: tuple[str, ...] = (
    "device",
    "config_id",
    "target_fold",
    "batch_rows",
    "steps",
    "discarded_steps",
    "median_step_ms",
    "mean_step_ms",
    "same_seed_losses_identical",
    "same_seed_max_loss_difference",
    "first_loss",
    "last_loss",
    "scoring_rows",
    "scoring_rows_per_second",
    "num_threads",
    "torch_version",
    "process_max_rss_gb",
    "git_commit",
    "git_dirty",
    "finished_at",
)
TRIAL_COLUMNS: tuple[str, ...] = (
    "area",
    "config_id",
    "config",
    "seed",
    "fold",
    "train_folds",
    "device",
    "epochs",
    "n_train_rows",
    "n_train_queries",
    "n_scored_rows",
    "n_queries",
    "prepare_seconds",
    "train_seconds",
    "seconds_per_epoch",
    "score_seconds",
    "score_seconds_per_model",
    "total_seconds",
    "process_max_rss_gb",
    "num_threads",
    "torch_version",
    "git_commit",
    "git_dirty",
    "finished_at",
)
METRIC_KEY_COLUMNS: tuple[str, ...] = ("area", "config_id", "seed", "fold", "epoch", "n_queries")
METRIC_TAIL_COLUMNS: tuple[str, ...] = ("train_loss", "git_commit", "finished_at")


def metric_columns(metrics: tuple[str, ...]) -> tuple[str, ...]:
    """The header of results/deepfm_metrics.csv for the given metrics."""
    return (*METRIC_KEY_COLUMNS, *metrics, *METRIC_TAIL_COLUMNS)


def model_settings(config: DeepFMConfig) -> dict[str, Any]:
    """Everything that defines a DeepFM configuration's results, as plain values.

    Left out on purpose: the seed (a configuration is run with several seeds)
    and the device (recorded with every training instead).

    The number of epochs depends on the schedule. With the cosine schedule it IS
    part of the configuration: the rate of every step depends on the length of
    the run, so a run of T epochs is its own model. With a constant rate, epoch
    k of a longer run is the model of a k-epoch run, so the number of epochs is
    left out; the two schedule settings are left out too, so that a
    constant-rate configuration keeps the id it had before the schedule existed
    (the step-0 rows of 2026-11-19).
    """
    training = asdict(config.training)
    for name in ("seed", "device"):
        del training[name]
    if config.training.lr_schedule == DEEPFM_LR_CONSTANT:
        for name in ("epochs", "lr_schedule", "min_lr_ratio"):
            del training[name]
    return {
        "experiment": config.experiment,
        "negatives_per_positive": config.negatives_per_positive,
        "window_weeks": config.window_weeks,
        "model": {**asdict(config.model), "hidden_units": list(config.model.hidden_units)},
        "n_bins": config.n_bins,
        "clip_percentiles": [config.clip_low_percentile, config.clip_high_percentile],
        "training": training,
    }


def deepfm_config_id(config: DeepFMConfig) -> str:
    """A 12-character id of a configuration: the sha256 of its settings as sorted JSON."""
    text = json.dumps(model_settings(config), sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:CONFIG_ID_LENGTH]


def now() -> str:
    """The current local time as text, to the second."""
    return datetime.now().isoformat(timespec="seconds")


def stamp_columns(stamp: RunStamp) -> dict[str, str]:
    """The code state and the time, as the last three columns of a row."""
    return {"git_commit": stamp.git_commit, "git_dirty": str(stamp.git_dirty), "finished_at": now()}


# ---------------------------------------------------------------- recorded reference scores


@dataclass(frozen=True)
class ReferenceScores:
    """The recorded scores of the two reference systems on one V fold.

    Attributes:
        fold: The fold.
        metric: The metric, e.g. "map@12".
        rrf_baseline: The RRF baseline's value.
        lightgbm: The locked LightGBM ranker's value (locked configuration and number of trees).
    """

    fold: str
    metric: str
    rrf_baseline: float
    lightgbm: float


def _one_value(rows: list[dict[str, str]], metric: str, what: str) -> float:
    """The metric of the single matching row; anything else is an error."""
    if len(rows) != 1:
        raise ValueError(f"expected exactly one recorded row of {what}, found {len(rows)}")
    return float(rows[0][metric])


def read_reference_scores(paths: ProjectPaths, fold: str, metric: str) -> ReferenceScores:
    """Read the RRF baseline and the locked LightGBM score of a fold from the M7 records.

    Args:
        paths: Project paths (results/v_phase_metrics.csv, the ranker lock).
        fold: A V fold.
        metric: A metric column of that table.

    Raises:
        ValueError: If a reference row is missing or repeated.
    """
    lock = load_ranker_lock(paths.ranker_lock_file)
    with paths.v_phase_metrics_file.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["fold"] == fold]
    baseline = [row for row in rows if row["experiment"] == RRF_BASELINE]
    locked = [
        row
        for row in rows
        if row["experiment"] == FULL_EXPERIMENT
        and row["config_id"] == lock.selection.config_id
        and int(row["num_trees"]) == lock.model.num_trees
    ]
    return ReferenceScores(
        fold=fold,
        metric=metric,
        rrf_baseline=_one_value(baseline, metric, f"the RRF baseline on {fold}"),
        lightgbm=_one_value(locked, metric, f"the locked LightGBM ranker on {fold}"),
    )


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """All rows of a result table as dicts; empty if the file does not exist."""
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@dataclass(frozen=True)
class PooledReference:
    """The recorded reference scores pooled over several V folds (weighted by queries).

    Attributes:
        metric: The metric.
        folds: The folds that were pooled.
        rrf_baseline: Pooled value of the RRF baseline.
        lightgbm: Pooled value of the locked LightGBM ranker.
        lightgbm_by_fold: Fold -> the locked LightGBM ranker's value.
        rrf_by_fold: Fold -> the RRF baseline's value.
        n_queries: Fold -> its number of queries.
    """

    metric: str
    folds: tuple[str, ...]
    rrf_baseline: float
    lightgbm: float
    lightgbm_by_fold: dict[str, float]
    rrf_by_fold: dict[str, float]
    n_queries: dict[str, int]


def read_pooled_reference(
    paths: ProjectPaths, folds: tuple[str, ...], metric: str
) -> PooledReference:
    """Read the reference scores of several folds and pool them by their numbers of queries.

    pooled = sum_f n_f * value_f / sum_f n_f, with n_f from the same recorded rows.
    """
    with paths.v_phase_metrics_file.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    is_baseline = [row for row in rows if row["experiment"] == RRF_BASELINE]
    n_queries, rrf, lightgbm = {}, {}, {}
    for fold in folds:
        reference = read_reference_scores(paths, fold, metric)
        rrf[fold], lightgbm[fold] = reference.rrf_baseline, reference.lightgbm
        n_queries[fold] = int(next(row for row in is_baseline if row["fold"] == fold)["n_queries"])
    total = sum(n_queries.values())
    return PooledReference(
        metric=metric,
        folds=folds,
        rrf_baseline=math.fsum(n_queries[fold] * rrf[fold] for fold in folds) / total,
        lightgbm=math.fsum(n_queries[fold] * lightgbm[fold] for fold in folds) / total,
        lightgbm_by_fold=lightgbm,
        rrf_by_fold=rrf,
        n_queries=n_queries,
    )


def locked_lightgbm(paths: ProjectPaths) -> tuple[str, int]:
    """(config_id, number of trees) of the locked LightGBM ranker, from its lock file."""
    lock = load_ranker_lock(paths.ranker_lock_file)
    return lock.selection.config_id, lock.model.num_trees


# ---------------------------------------------------------------- run time and memory of an arm


def _run_cost(rows: list[dict[str, str]], folds: tuple[str, ...], what: str) -> tuple[float, float]:
    """(summed total_seconds, highest process_max_rss_gb) of exactly one row per fold.

    Raises:
        ValueError: If the rows are not one per fold.
    """
    if sorted(row["fold"] for row in rows) != sorted(folds):
        raise ValueError(
            f"{what}: expected one trial row per fold {list(folds)}, found {len(rows)}"
        )
    seconds = math.fsum(float(row["total_seconds"]) for row in rows)
    return seconds, max(float(row["process_max_rss_gb"]) for row in rows)


def deepfm_run_cost(
    paths: ProjectPaths, area: str, config_id: str, seed: int, folds: tuple[str, ...]
) -> tuple[float, float]:
    """Run time (seconds, all folds) and highest true peak (GB) of a DeepFM configuration, seed."""
    rows = [
        row
        for row in read_csv_rows(paths.deepfm_trials_file)
        if (row["area"], row["config_id"], int(row["seed"])) == (area, config_id, seed)
        and row["fold"] in folds
    ]
    return _run_cost(rows, folds, f"DeepFM {config_id} seed {seed}")


def lightgbm_arm_run_cost(
    paths: ProjectPaths, arm: str, seed: int, folds: tuple[str, ...]
) -> tuple[float, float]:
    """Run time and highest true peak of a LightGBM arm and seed (results/m9_lgb_trials.csv)."""
    rows = [
        row
        for row in read_csv_rows(paths.m9_lgb_trials_file)
        if (row["arm"], int(row["seed"])) == (arm, seed) and row["fold"] in folds
    ]
    return _run_cost(rows, folds, f"LightGBM {arm} seed {seed}")


def locked_lightgbm_run_cost(paths: ProjectPaths, folds: tuple[str, ...]) -> tuple[float, float]:
    """Run time and highest true peak of the locked Ranker's M7 trials (results/v_phase.csv).

    These trials trained to the grid's full number of trees and evaluated every
    checkpoint, so their time is not the time of training the locked number of trees.
    """
    config_id, _ = locked_lightgbm(paths)
    rows = [
        row
        for row in read_csv_rows(paths.v_phase_results_file)
        if (row["experiment"], row["config_id"]) == (FULL_EXPERIMENT, config_id)
        and row["fold"] in folds
    ]
    return _run_cost(rows, folds, f"the locked LightGBM ranker {config_id}")
