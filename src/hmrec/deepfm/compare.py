"""DeepFM against LightGBM on the V folds: a paired bootstrap and three query groups (report only).

Both models saved the AP@12 of every query of a V fold when they were
evaluated. This module only reads those saved values; no label is read and no
model is loaded.

    1. Interval of the difference. The cluster bootstrap over customers of the
       diagnostics after M8 (proposal §10.2), with its settings
       (configs/m8_diagnostics.yaml): the queries of the 4 folds are pooled; a
       replicate draws n_customers customers with replacement, a drawn customer
       brings all of their queries; pooled MAP* = sum_c w_c S_c / sum_c w_c n_c.
       DeepFM and LightGBM use the SAME draws, so "DeepFM minus LightGBM" is a
       paired comparison. The interval is the alpha/2 and 1 - alpha/2 quantiles.
       It reflects the sampling of customers only, not differences between weeks
       or between training runs.
    2. Query groups, by the customer's history before the anchor (the saved
       feature user_days_since_last_purchase): no_history (NaN), recent_purchase
       (at most `recent_purchase_days` days), other. MAP@12 of both models per
       group, per fold and pooled.

The bootstrap is written again here because no other code may import
hmrec.diagnostics (AGENTS.md §3.2); tests/unit/test_deepfm_compare.py shows that
it gives exactly the intervals of the diagnostics' function on the same input.

This module imports neither torch nor LightGBM.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from hmrec.feature_files import verify_feature_files

QUERY_ID = "query_id"
FOLD = "fold"
CUSTOMER_IDX = "customer_idx"
DEEPFM_AP = "deepfm_ap"
LIGHTGBM_AP = "lightgbm_ap"
QUERY_GROUP = "query_group"
DAYS = "user_days_since_last_purchase"
NO_HISTORY = "no_history"
RECENT_PURCHASE = "recent_purchase"
OTHER = "other"
GROUPS = (NO_HISTORY, RECENT_PURCHASE, OTHER)
POOLED = "pooled"
DEEPFM = "deepfm"
LIGHTGBM = "lightgbm"
DIFFERENCE = "deepfm_minus_lightgbm"
QUANTITIES = (DEEPFM, LIGHTGBM, DIFFERENCE)
QUANTILE_DECIMALS = 12  # removes float noise from (1 - level) / 2, as in the diagnostics


# ---------------------------------------------------------------- reading saved values


def read_deepfm_ap(per_query_file: Path, epoch: int, metric: str) -> pl.DataFrame:
    """The saved per-query values of one DeepFM trial at one epoch: query_id, deepfm_ap."""
    table = pl.read_parquet(per_query_file).filter(pl.col("epoch") == epoch)
    if table.height == 0:
        raise ValueError(f"{per_query_file} has no rows for epoch {epoch}")
    return table.select(QUERY_ID, pl.col(metric).alias(DEEPFM_AP))


def read_lightgbm_ap(per_query_file: Path, num_trees: int, metric: str) -> pl.DataFrame:
    """The saved per-query values of the LightGBM trial at `num_trees`: query_id, lightgbm_ap."""
    table = pl.read_parquet(per_query_file).filter(pl.col("num_trees") == num_trees)
    if table.height == 0:
        raise ValueError(f"{per_query_file} has no rows for {num_trees} trees")
    return table.select(QUERY_ID, pl.col(metric).alias(LIGHTGBM_AP))


def read_query_days(features_dir: Path, wide_columns: Sequence[str]) -> pl.DataFrame:
    """One row per query of a fold with its user_days_since_last_purchase (NaN = no history).

    The wide table is verified first; only two columns are read.

    Raises:
        ValueError: If the value differs between the candidate rows of a query.
    """
    feature_files = verify_feature_files(features_dir, wide_columns)
    parts = [
        pl.read_parquet(features_dir / part.name, columns=[QUERY_ID, DAYS]).unique()
        for part in feature_files.parts
    ]
    per_query = pl.concat(parts)
    if per_query[QUERY_ID].is_duplicated().any():
        raise ValueError(f"{DAYS} differs between the candidate rows of one query")
    return per_query.sort(QUERY_ID)


def assign_groups(days: pl.DataFrame, recent_purchase_days: int) -> pl.DataFrame:
    """Put every query into one of the three groups: query_id, query_group."""
    if recent_purchase_days <= 0:
        raise ValueError(f"recent_purchase_days must be > 0, got {recent_purchase_days}")
    group = (
        pl.when(pl.col(DAYS).is_nan())
        .then(pl.lit(NO_HISTORY))
        .when(pl.col(DAYS) <= recent_purchase_days)
        .then(pl.lit(RECENT_PURCHASE))
        .otherwise(pl.lit(OTHER))
        .alias(QUERY_GROUP)
    )
    return days.select(QUERY_ID, group).sort(QUERY_ID)


def fold_table(
    fold: str,
    queries: pl.DataFrame,
    groups: pl.DataFrame,
    deepfm_ap: pl.DataFrame,
    lightgbm_ap: pl.DataFrame,
) -> pl.DataFrame:
    """One row per LABELLED query of a fold with both models' AP, its customer and its group.

    Args:
        fold: Fold name.
        queries: The fold's query list: query_id, customer_idx.
        groups: query_id, query_group.
        deepfm_ap: query_id, deepfm_ap (one row per labelled query).
        lightgbm_ap: query_id, lightgbm_ap (one row per labelled query).

    Returns:
        Columns query_id, fold, customer_idx, query_group, deepfm_ap, lightgbm_ap;
        sorted by query_id.

    Raises:
        ValueError: If the two models were not evaluated on exactly the same queries,
            or a query has no customer or group.
    """
    if deepfm_ap[QUERY_ID].sort().to_list() != lightgbm_ap[QUERY_ID].sort().to_list():
        raise ValueError(f"{fold}: DeepFM and LightGBM were not evaluated on the same queries")
    table = (
        deepfm_ap.join(lightgbm_ap, on=QUERY_ID)
        .join(queries.select(QUERY_ID, CUSTOMER_IDX), on=QUERY_ID, how="left")
        .join(groups, on=QUERY_ID, how="left")
        .with_columns(pl.lit(fold).alias(FOLD))
    )
    if table[CUSTOMER_IDX].null_count() > 0 or table[QUERY_GROUP].null_count() > 0:
        raise ValueError(f"{fold}: a query has no customer or no group")
    columns = [QUERY_ID, FOLD, CUSTOMER_IDX, QUERY_GROUP, DEEPFM_AP, LIGHTGBM_AP]
    return table.select(columns).sort(QUERY_ID)


# ---------------------------------------------------------------- the bootstrap


@dataclass(frozen=True)
class Interval:
    """A point estimate with its bootstrap interval.

    Attributes:
        estimate: The value on the real data (no resampling).
        low: Lower end of the interval.
        high: Upper end of the interval.
    """

    estimate: float
    low: float
    high: float


def _interval(estimate: float, replicates: np.ndarray, confidence_level: float) -> Interval:
    """The percentile interval of the replicates around the real-data estimate."""
    low_quantile = round((1.0 - confidence_level) / 2.0, QUANTILE_DECIMALS)
    high_quantile = round((1.0 + confidence_level) / 2.0, QUANTILE_DECIMALS)
    low, high = np.quantile(replicates, [low_quantile, high_quantile])
    return Interval(estimate=estimate, low=float(low), high=float(high))


def paired_bootstrap(
    per_query: pl.DataFrame, n_replicates: int, seed: int, confidence_level: float
) -> tuple[dict[str, Interval], int]:
    """Intervals of the pooled MAP of both models and of DeepFM minus LightGBM.

    Args:
        per_query: One row per query of the pooled folds: customer_idx,
            deepfm_ap, lightgbm_ap (float64).
        n_replicates: Number of bootstrap replicates.
        seed: Seed of the draws (numpy default_rng).
        confidence_level: e.g. 0.95.

    Returns:
        (quantity -> interval for "deepfm", "lightgbm", "deepfm_minus_lightgbm",
        the number of customers).
    """
    if not 0 < confidence_level < 1:
        raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}")
    # Per customer: number of queries and summed AP; sorted, so the same seed
    # always draws the same customers.
    totals = (
        per_query.group_by(CUSTOMER_IDX)
        .agg(
            pl.len().cast(pl.Int64).alias("n"),
            pl.col(DEEPFM_AP).sum().alias(DEEPFM),
            pl.col(LIGHTGBM_AP).sum().alias(LIGHTGBM),
        )
        .sort(CUSTOMER_IDX)
    )
    n_customers = totals.height
    if n_customers == 0 or n_replicates <= 0:
        raise ValueError("the bootstrap needs customers and at least one replicate")
    n_queries = totals["n"].to_numpy()
    sums = {name: totals[name].to_numpy().astype(np.float64) for name in (DEEPFM, LIGHTGBM)}
    total_queries = int(n_queries.sum())
    estimates = {name: math.fsum(sums[name].tolist()) / total_queries for name in sums}
    replicates = {name: np.empty(n_replicates, dtype=np.float64) for name in sums}
    generator = np.random.default_rng(seed)
    for index in range(n_replicates):
        drawn = generator.integers(0, n_customers, size=n_customers)
        weights = np.bincount(drawn, minlength=n_customers).astype(np.int64)
        drawn_queries = np.dot(weights, n_queries)
        for name in sums:
            # The SAME weights for both models: the difference is a paired comparison.
            replicates[name][index] = float(np.dot(weights, sums[name]) / drawn_queries)
    difference = replicates[DEEPFM] - replicates[LIGHTGBM]
    intervals = {
        DEEPFM: _interval(estimates[DEEPFM], replicates[DEEPFM], confidence_level),
        LIGHTGBM: _interval(estimates[LIGHTGBM], replicates[LIGHTGBM], confidence_level),
        DIFFERENCE: _interval(
            estimates[DEEPFM] - estimates[LIGHTGBM], difference, confidence_level
        ),
    }
    return intervals, n_customers


# ---------------------------------------------------------------- the group table


def _mean(table: pl.DataFrame, column: str) -> float:
    """Mean of a column over the rows, summed with math.fsum in query_id order."""
    values = table.sort(QUERY_ID)[column].to_list()
    return math.fsum(values) / len(values)


def group_rows(per_query: pl.DataFrame) -> list[dict[str, str]]:
    """MAP of both models in every query group, per fold and pooled over all folds.

    Args:
        per_query: The fold tables of all folds, concatenated.

    Returns:
        One dict per (fold or "pooled", group or "all"): fold, query_group,
        n_queries, query_share, deepfm_map, lightgbm_map, difference (as text;
        floats with repr()). A group without queries has no row.
    """
    rows: list[dict[str, str]] = []
    folds = per_query[FOLD].unique(maintain_order=True).to_list()
    for scope in (*folds, POOLED):
        in_scope = per_query if scope == POOLED else per_query.filter(pl.col(FOLD) == scope)
        for group in (*GROUPS, "all"):
            table = in_scope if group == "all" else in_scope.filter(pl.col(QUERY_GROUP) == group)
            if table.height == 0:
                continue
            deepfm, lightgbm = _mean(table, DEEPFM_AP), _mean(table, LIGHTGBM_AP)
            rows.append(
                {
                    "fold": scope,
                    "query_group": group,
                    "n_queries": str(table.height),
                    "query_share": repr(table.height / in_scope.height),
                    "deepfm_map": repr(deepfm),
                    "lightgbm_map": repr(lightgbm),
                    "difference": repr(deepfm - lightgbm),
                }
            )
    return rows


# ---------------------------------------------------------------- one comparison, from files

COMPARE_COLUMNS: tuple[str, ...] = (
    "folds",
    "metric",
    "quantity",
    "estimate",
    "ci_low",
    "ci_high",
    "confidence_level",
    "n_replicates",
    "seed",
    "n_customers",
    "n_queries",
    "deepfm_config_id",
    "deepfm_seed",
    "deepfm_epoch",
    "lightgbm_config_id",
    "lightgbm_num_trees",
)
GROUP_COLUMNS: tuple[str, ...] = (
    "fold",
    "query_group",
    "n_queries",
    "query_share",
    "deepfm_map",
    "lightgbm_map",
    "difference",
)


@dataclass(frozen=True)
class FoldFiles:
    """The saved files of one fold that the comparison reads.

    Attributes:
        fold: Fold name (a V fold).
        deepfm_per_query: The DeepFM trial's per_query.parquet.
        lightgbm_per_query: The LightGBM trial's per-query file.
        queries: The fold's query list (query_id, customer_idx).
        features_dir: The fold's frozen wide table (for the query groups).
    """

    fold: str
    deepfm_per_query: Path
    lightgbm_per_query: Path
    queries: Path
    features_dir: Path


@dataclass(frozen=True)
class CompareSettings:
    """How the comparison is made.

    Attributes:
        metric: "map@12".
        deepfm_epoch: The epoch of the DeepFM trials that is compared.
        lightgbm_num_trees: The number of trees of the LightGBM trials.
        recent_purchase_days: Largest number of days that still counts as recent.
        n_replicates: Bootstrap replicates.
        seed: Seed of the bootstrap draws.
        confidence_level: e.g. 0.95.
        wide_columns: The columns the wide tables must have.
    """

    metric: str
    deepfm_epoch: int
    lightgbm_num_trees: int
    recent_purchase_days: int
    n_replicates: int
    seed: int
    confidence_level: float
    wide_columns: tuple[str, ...]


def read_fold_table(files: FoldFiles, settings: CompareSettings) -> pl.DataFrame:
    """The per-query table of one fold (fold_table), read from the saved files."""
    queries = pl.read_parquet(files.queries, columns=[QUERY_ID, CUSTOMER_IDX])
    days = read_query_days(files.features_dir, settings.wide_columns)
    return fold_table(
        files.fold,
        queries,
        assign_groups(days, settings.recent_purchase_days),
        read_deepfm_ap(files.deepfm_per_query, settings.deepfm_epoch, settings.metric),
        read_lightgbm_ap(files.lightgbm_per_query, settings.lightgbm_num_trees, settings.metric),
    )


def compare_folds(
    folds: Sequence[FoldFiles], settings: CompareSettings
) -> tuple[dict[str, Interval], int, int, list[dict[str, str]]]:
    """Compare DeepFM with LightGBM on the given folds.

    Returns:
        (quantity -> interval, number of customers, number of queries, the rows
        of the group table).
    """
    per_query = pl.concat([read_fold_table(files, settings) for files in folds])
    intervals, n_customers = paired_bootstrap(
        per_query, settings.n_replicates, settings.seed, settings.confidence_level
    )
    return intervals, n_customers, per_query.height, group_rows(per_query)
