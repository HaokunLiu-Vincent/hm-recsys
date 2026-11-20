"""Result tables of the two LightGBM arms of the M9 comparison (proposal §16.8).

    results/m9_lgb_trials.csv    one row per training (arm, seed, fold): what ran,
                                 how long, the true memory peak, the runtime
    results/m9_lgb_metrics.csv   one row per (arm, seed, fold, number of trees): the
                                 fold's metric values with the first k trees

Rows are only appended (hmrec.csv_records); floats are written with repr().
Nothing here touches results/v_phase*.csv: the M7 records stay as they are, and
the lambdarank arm's first seed is READ from them, never written again.

Also here: the number-of-trees rule of the binary arm. Its first seed is trained
to the configured number of trees and evaluated at every checkpoint; with P(k)
the pooled MAP@12 (weighted by queries) of the folds with k trees,

    k* = the smallest k with (max P - P(k)) / max P < tolerance

(hmrec.hpo.selection.choose_num_trees, the rule of proposal §9.4).

This module imports neither LightGBM nor torch: the LightGBM side writes these
tables (hmrec.hpo.lgb_arms) and the torch side reads them for the comparison
(scripts/m9_deepfm_compare.py).
"""

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

from hmrec.csv_records import append_row, read_rows
from hmrec.hpo.selection import NumTreesChoice, choose_num_trees

ARM_LAMBDARANK = "lambdarank"
ARM_BINARY = "binary"
ARMS = (ARM_LAMBDARANK, ARM_BINARY)

TRIAL_COLUMNS: tuple[str, ...] = (
    "arm",
    "objective",
    "seed",
    "config_id",
    "config_json",
    "fold",
    "train_folds",
    "num_trees",
    "evaluated_trees",
    "n_train_rows",
    "n_train_queries",
    "n_scored_rows",
    "n_queries",
    "load_seconds",
    "train_seconds",
    "seconds_per_tree",
    "score_seconds",
    "total_seconds",
    "process_max_rss_gb",
    "model_sha256",
    "num_threads",
    "lightgbm_version",
    "libomp_path",
    "libomp_version",
    "git_commit",
    "git_dirty",
    "finished_at",
)
METRIC_KEY_COLUMNS: tuple[str, ...] = (
    "arm",
    "seed",
    "config_id",
    "fold",
    "num_trees",
    "n_queries",
)
METRIC_TAIL_COLUMNS: tuple[str, ...] = ("train_metric", "git_commit", "finished_at")


def metric_columns(metrics: Sequence[str]) -> tuple[str, ...]:
    """The header of results/m9_lgb_metrics.csv for the given metrics."""
    return (*METRIC_KEY_COLUMNS, *metrics, *METRIC_TAIL_COLUMNS)


def append_trial_row(path: Path, row: Mapping[str, str]) -> None:
    """Append one row to results/m9_lgb_trials.csv."""
    append_row(path, TRIAL_COLUMNS, row)


def append_metric_row(path: Path, metrics: Sequence[str], row: Mapping[str, str]) -> None:
    """Append one row to results/m9_lgb_metrics.csv."""
    append_row(path, metric_columns(metrics), row)


def read_trial_rows(path: Path) -> list[dict[str, str]]:
    """All trial rows as text dicts; empty if the file does not exist yet."""
    return read_rows(path, TRIAL_COLUMNS)


def read_metric_rows(path: Path, metrics: Sequence[str]) -> list[dict[str, str]]:
    """All metric rows as text dicts; empty if the file does not exist yet."""
    return read_rows(path, metric_columns(metrics))


def is_recorded(rows: Sequence[Mapping[str, str]], arm: str, seed: int, fold: str) -> bool:
    """True if the training of (arm, seed, fold) has its trial row.

    Raises:
        ValueError: If it is recorded more than once.
    """
    matches = [
        row for row in rows if (row["arm"], int(row["seed"]), row["fold"]) == (arm, seed, fold)
    ]
    if len(matches) > 1:
        raise ValueError(f"the training ({arm}, seed {seed}, {fold}) is recorded twice")
    return bool(matches)


def pooled_curve(
    rows: Sequence[Mapping[str, str]], arm: str, seed: int, folds: Sequence[str], metric: str
) -> dict[int, float]:
    """Number of trees -> pooled score of one arm and seed, for every k all folds have.

    pooled(k) = sum_f n_f * value_f(k) / sum_f n_f   (math.fsum; n_f = queries of fold f)

    Args:
        rows: Rows of results/m9_lgb_metrics.csv.
        arm: "lambdarank" or "binary".
        seed: The LightGBM seed.
        folds: The folds to pool (V1 ... V4).
        metric: e.g. "map@12".

    Raises:
        ValueError: If a (fold, k) is recorded twice.
    """
    values: dict[int, dict[str, tuple[int, float]]] = {}
    for row in rows:
        if row["arm"] != arm or int(row["seed"]) != seed or row["fold"] not in folds:
            continue
        by_fold = values.setdefault(int(row["num_trees"]), {})
        if row["fold"] in by_fold:
            raise ValueError(
                f"{arm} seed {seed}: {row['fold']} with {row['num_trees']} trees is recorded twice"
            )
        by_fold[row["fold"]] = (int(row["n_queries"]), float(row[metric]))
    curve: dict[int, float] = {}
    for num_trees in sorted(values):
        by_fold = values[num_trees]
        if sorted(by_fold) != sorted(folds):
            continue
        total_queries = sum(n_queries for n_queries, _ in by_fold.values())
        weighted = math.fsum(n_queries * value for n_queries, value in by_fold.values())
        curve[num_trees] = weighted / total_queries
    return curve


def choose_binary_num_trees(
    rows: Sequence[Mapping[str, str]],
    seed: int,
    folds: Sequence[str],
    metric: str,
    tolerance: float,
    expected_checkpoints: Sequence[int],
) -> NumTreesChoice:
    """Apply the number-of-trees rule to the binary arm's first seed.

    Args:
        rows: Rows of results/m9_lgb_metrics.csv.
        seed: The seed that was trained to the full number of trees.
        folds: The folds to pool.
        metric: The selection metric.
        tolerance: Relative tolerance of the rule (0.01).
        expected_checkpoints: The numbers of trees that must all be recorded for
            every fold, e.g. 100, 200, ..., 1000.

    Raises:
        ValueError: If a checkpoint is missing for a fold.
    """
    curve = pooled_curve(rows, ARM_BINARY, seed, folds, metric)
    missing = sorted(set(expected_checkpoints) - set(curve))
    if missing:
        raise ValueError(
            f"binary seed {seed}: no complete metric rows at {missing} trees for folds "
            f"{list(folds)}; the arm's first seed is not finished"
        )
    return choose_num_trees({k: curve[k] for k in expected_checkpoints}, tolerance)
