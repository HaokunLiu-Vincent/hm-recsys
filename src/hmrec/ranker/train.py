"""LambdaMART training on a 4-week window (proposal §9.2-9.4).

Input: a binned Dataset of the 4 training weeks (hmrec.ranker.dataset) and the
parameters of configs/ranker.yaml. Output: a model trained FROM SCRATCH to a
fixed number of trees (no early stopping, no init_model), saved as LightGBM text.

Reproducibility: `deterministic` is on, the seed and the number of threads are
fixed, and the histogram mode (col_wise / row_wise) is fixed. With these, the
first k trees of a model trained to K > k trees are exactly the model trained to
k trees, so one model gives the whole curve over the number of trees.
"""

import hashlib
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb

from hmrec.config import (
    HISTOGRAM_COL_WISE,
    HISTOGRAM_ROW_WISE,
    RankerConfig,
    RankerTunedParams,
)
from hmrec.locks import file_sha256
from hmrec.logging_utils import get_logger

LOGGER = get_logger("ranker.train")

HISTOGRAM_PARAMS = {HISTOGRAM_COL_WISE: "force_col_wise", HISTOGRAM_ROW_WISE: "force_row_wise"}
FIRST_TREE_MARKER = "Tree=0"
END_OF_TREES_MARKER = "end of trees"


# ---------------------------------------------------------------- parameters


def dataset_params(config: RankerConfig, num_threads: int) -> dict[str, Any]:
    """The LightGBM parameters a binary Dataset is built with (they define the bins).

    The seed is part of them: it draws the rows that are sampled to find the bin
    boundaries (bin_construct_sample_cnt).
    """
    return {
        **config.dataset_params,
        "seed": config.seed,
        "deterministic": config.train_params["deterministic"],
        "num_threads": num_threads,
        "verbosity": config.train_params["verbosity"],
    }


def training_params(
    config: RankerConfig, tuned: RankerTunedParams, histogram_mode: str, num_threads: int
) -> dict[str, Any]:
    """All LightGBM parameters of one training run.

    Args:
        config: configs/ranker.yaml.
        tuned: num_leaves and min_data_in_leaf of this run (negatives_per_positive
            only chooses the Dataset and is not a LightGBM parameter).
        histogram_mode: "col_wise" or "row_wise".
        num_threads: CPU threads (configs/base.yaml).

    Raises:
        ValueError: If the histogram mode is not decided.
    """
    if histogram_mode not in HISTOGRAM_PARAMS:
        raise ValueError(
            f"histogram mode must be one of {sorted(HISTOGRAM_PARAMS)}, got {histogram_mode!r}. "
            "It is decided by the speed test and then written to configs/ranker.yaml."
        )
    return {
        **config.dataset_params,
        **config.train_params,
        "num_leaves": tuned.num_leaves,
        "min_data_in_leaf": tuned.min_data_in_leaf,
        "seed": config.seed,
        "num_threads": num_threads,
        HISTOGRAM_PARAMS[histogram_mode]: True,
    }


def decide_histogram_mode(
    col_wise_seconds_per_tree: float,
    row_wise_seconds_per_tree: float,
    row_wise_peak_gb: float,
    memory_limit_gb: float,
    min_row_wise_speedup: float,
) -> str:
    """The rule of configs/ranker.yaml: row_wise only if clearly faster and within memory.

    row_wise is chosen if BOTH hold:
        (col_wise - row_wise) / col_wise >= min_row_wise_speedup   (seconds per tree)
        row_wise_peak_gb <= memory_limit_gb                       (true peak of its process)
    Otherwise col_wise. Only speed and memory are used: no score is involved.
    """
    if col_wise_seconds_per_tree <= 0 or row_wise_seconds_per_tree <= 0:
        raise ValueError("seconds per tree must be > 0 for both modes")
    speedup = (col_wise_seconds_per_tree - row_wise_seconds_per_tree) / col_wise_seconds_per_tree
    if speedup >= min_row_wise_speedup and row_wise_peak_gb <= memory_limit_gb:
        return HISTOGRAM_ROW_WISE
    return HISTOGRAM_COL_WISE


# ---------------------------------------------------------------- training


@dataclass(frozen=True)
class TrainLog:
    """What was measured while a model trained.

    Attributes:
        elapsed_seconds: Seconds from the start of training to the end of each
            tree; elapsed_seconds[i] belongs to tree i + 1.
        train_metric: number of trees -> training-set NDCG@12 at that point
            (every train_metric_every trees and after the last tree).
    """

    elapsed_seconds: tuple[float, ...]
    train_metric: dict[int, float]

    @property
    def total_seconds(self) -> float:
        """Seconds of the whole training."""
        return self.elapsed_seconds[-1]

    def seconds_per_tree(self, warmup_trees: int) -> float:
        """Mean seconds per tree over the trees after the first `warmup_trees`."""
        n_trees = len(self.elapsed_seconds)
        if not 0 <= warmup_trees < n_trees:
            raise ValueError(f"warmup_trees must be in [0, {n_trees}), got {warmup_trees}")
        start = self.elapsed_seconds[warmup_trees - 1] if warmup_trees > 0 else 0.0
        return (self.elapsed_seconds[-1] - start) / (n_trees - warmup_trees)


def train_ranker(
    dataset: lgb.Dataset, params: Mapping[str, Any], num_trees: int, metric_every: int
) -> tuple[lgb.Booster, TrainLog]:
    """Train a LambdaMART model from scratch to exactly `num_trees` trees.

    Args:
        dataset: The binned training data with labels and groups.
        params: Output of training_params.
        num_trees: Trees to train. There is no early stopping.
        metric_every: The training-set metric is computed and logged every this
            many trees (it is not needed for training, only for the log).

    Returns:
        The model and its training log.
    """
    if num_trees <= 0 or metric_every <= 0:
        raise ValueError("num_trees and metric_every must be > 0")
    elapsed: list[float] = []
    train_metric: dict[int, float] = {}
    started = time.perf_counter()

    def record_progress(env: lgb.callback.CallbackEnv) -> None:
        """Called by LightGBM after every tree."""
        n_done = env.iteration + 1
        elapsed.append(time.perf_counter() - started)
        if (n_done % metric_every == 0 or n_done == num_trees) and isinstance(
            env.model, lgb.Booster
        ):
            result = env.model.eval_train()[0]
            train_metric[n_done] = float(result.metric_value)
            LOGGER.info(
                "trees=%d | train %s=%.6f | elapsed=%.1f s",
                n_done,
                result.metric_name,
                train_metric[n_done],
                elapsed[-1],
            )

    booster = lgb.train(
        dict(params), dataset, num_boost_round=num_trees, callbacks=[record_progress]
    )
    if booster.num_trees() != num_trees:
        raise ValueError(f"trained {booster.num_trees()} trees, expected {num_trees}")
    return booster, TrainLog(tuple(elapsed), train_metric)


# ---------------------------------------------------------------- model files


def save_model(booster: lgb.Booster, path: Path) -> str:
    """Save the model as LightGBM text and return the file's sha256. Never replaces a file."""
    if path.exists():
        raise FileExistsError(f"{path} already exists; models are never replaced")
    path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(path)
    return file_sha256(path)


def load_model(path: Path) -> lgb.Booster:
    """Load a model saved by save_model."""
    if not path.is_file():
        raise FileNotFoundError(f"model file not found: {path}")
    return lgb.Booster(model_file=path)


def first_trees_sha256(booster: lgb.Booster, num_trees: int) -> str:
    """sha256 of the text of the model's first `num_trees` trees (not of its header).

    Two models whose first k trees are the same give the same value, even if one
    of them has more trees. Used to check that training is reproducible.
    """
    if not 0 < num_trees <= booster.num_trees():
        raise ValueError(f"num_trees must be in 1..{booster.num_trees()}, got {num_trees}")
    text = booster.model_to_string(num_iteration=num_trees)
    trees = text[text.index(FIRST_TREE_MARKER) : text.index(END_OF_TREES_MARKER)]
    return hashlib.sha256(trees.encode("utf-8")).hexdigest()


def gain_importance(booster: lgb.Booster, num_trees: int) -> dict[str, float]:
    """Total split gain of every feature over the first `num_trees` trees (analysis only, §8.7)."""
    gains = booster.feature_importance(importance_type="gain", iteration=num_trees)
    return dict(zip(booster.feature_name(), (float(gain) for gain in gains), strict=True))
