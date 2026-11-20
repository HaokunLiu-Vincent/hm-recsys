"""The arms of the M9 comparison on the V folds, from saved values (report only; §16.8).

Arms of the 2x2 study: lgb_lambdarank, lgb_binary, deepfm_listwise,
deepfm_pointwise, each with several seeds. The input-representation extension
(proposal §16.10) adds four DeepFM arms and its own contrasts: the caller gives
the arms and the contrasts, the method is the same. Every training saved the
metric values of every query of its V fold when it was evaluated. This module
only reads those saved values; no label is read and no model is loaded.

    1. Arm table. For each arm and seed: the pooled metric = the mean over ALL
       queries of the folds (which is the query-weighted mean of the folds).
       Over the seeds: mean, sample standard deviation (n - 1) and
       range = (max - min) / min.
    2. Contrasts. The cluster bootstrap over customers of the diagnostics after
       M8 (proposal §10.2), with its settings (configs/m8_diagnostics.yaml): the
       queries of the folds are pooled; a replicate draws n_customers customers
       with replacement, a drawn customer brings all of their queries;
       pooled MAP* = sum_c w_c S_c / sum_c w_c n_c. Both arms of a contrast use
       the SAME draws, so "A minus B" is a paired comparison. The interval is
       the alpha/2 and 1 - alpha/2 quantiles. Two bases: one seed, and the mean
       over the seeds of every query's AP ("seed_mean"). The interval reflects
       the sampling of customers only, not differences between weeks or runs.
    3. Query groups, by the customer's history before the anchor (the saved
       feature user_days_since_last_purchase): no_history (NaN), recent_purchase
       (at most `recent_purchase_days` days), other. MAP of every arm per group,
       per fold and pooled.

The bootstrap is written again here because no other code may import
hmrec.diagnostics (AGENTS.md §3.2); tests/unit/test_deepfm_compare.py shows that
it gives exactly the intervals of the diagnostics' function on the same input.

This module imports neither torch nor LightGBM.
"""

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from hmrec.feature_files import verify_feature_files

QUERY_ID = "query_id"
FOLD = "fold"
CUSTOMER_IDX = "customer_idx"
QUERY_GROUP = "query_group"
DAYS = "user_days_since_last_purchase"
NO_HISTORY = "no_history"
RECENT_PURCHASE = "recent_purchase"
OTHER = "other"
GROUPS = (NO_HISTORY, RECENT_PURCHASE, OTHER)
ALL_QUERIES = "all"
POOLED = "pooled"
SEED_MEAN = "seed_mean"
FIRST = "first"
SECOND = "second"
DIFFERENCE = "difference"
QUANTILE_DECIMALS = 12  # removes float noise from (1 - level) / 2, as in the diagnostics

ARM_LGB_LAMBDARANK = "lgb_lambdarank"
ARM_LGB_BINARY = "lgb_binary"
ARM_DEEPFM_LISTWISE = "deepfm_listwise"
ARM_DEEPFM_POINTWISE = "deepfm_pointwise"
ARMS = (ARM_LGB_LAMBDARANK, ARM_LGB_BINARY, ARM_DEEPFM_LISTWISE, ARM_DEEPFM_POINTWISE)
# (first, second): the contrast is "first minus second".
CONTRASTS: tuple[tuple[str, str], ...] = (
    (ARM_DEEPFM_POINTWISE, ARM_LGB_LAMBDARANK),
    (ARM_DEEPFM_LISTWISE, ARM_LGB_LAMBDARANK),
    (ARM_LGB_LAMBDARANK, ARM_LGB_BINARY),
    (ARM_DEEPFM_LISTWISE, ARM_DEEPFM_POINTWISE),
)


def seed_basis(seed: int) -> str:
    """The name of the basis "one seed", e.g. "seed_42"."""
    return f"seed_{seed}"


def value_column(arm: str, basis: str, metric: str) -> str:
    """The per-query table's column of one arm's values, e.g. "lgb_binary|seed_42|map@12"."""
    return f"{arm}|{basis}|{metric}"


# ---------------------------------------------------------------- reading saved values


@dataclass(frozen=True)
class ArmRuns:
    """Where the saved per-query values of one arm are.

    Attributes:
        arm: Name of the arm.
        step_column: The column that says which model of the file is meant:
            "epoch" (DeepFM) or "num_trees" (LightGBM).
        steps: Seed -> the value of that column to keep (the selected T or k).
        files: Seed -> fold -> the per-query file of that training.
    """

    arm: str
    step_column: str
    steps: dict[int, int]
    files: dict[int, dict[str, Path]]

    @property
    def seeds(self) -> tuple[int, ...]:
        """The arm's seeds in order."""
        return tuple(self.files)


def read_saved_values(
    per_query_file: Path, step_column: str, step: int, metrics: Sequence[str]
) -> pl.DataFrame:
    """The saved per-query values of one model: query_id and one float64 column per metric.

    Raises:
        ValueError: If the file has no rows for `step`.
    """
    table = pl.read_parquet(per_query_file).filter(pl.col(step_column) == step)
    if table.height == 0:
        raise ValueError(f"{per_query_file} has no rows with {step_column} = {step}")
    return table.select(QUERY_ID, *metrics).sort(QUERY_ID)


def arm_fold_values(arm: ArmRuns, fold: str, metrics: Sequence[str]) -> pl.DataFrame:
    """One row per labelled query of a fold with the arm's values for every seed and their mean.

    Returns:
        query_id, then for every metric one column per seed
        (value_column(arm, seed_basis(seed), metric)) and one seed-mean column
        (value_column(arm, SEED_MEAN, metric)); sorted by query_id.

    Raises:
        ValueError: If two seeds were not evaluated on exactly the same queries.
    """
    table: pl.DataFrame | None = None
    for seed in arm.seeds:
        values = read_saved_values(arm.files[seed][fold], arm.step_column, arm.steps[seed], metrics)
        renamed = values.rename(
            {metric: value_column(arm.arm, seed_basis(seed), metric) for metric in metrics}
        )
        if table is None:
            table = renamed
            continue
        if table[QUERY_ID].to_list() != renamed[QUERY_ID].to_list():
            raise ValueError(f"{arm.arm} {fold}: seed {seed} was evaluated on other queries")
        table = table.join(renamed, on=QUERY_ID)
    if table is None:
        raise ValueError(f"{arm.arm}: the arm has no seed")
    for metric in metrics:
        seed_columns = [value_column(arm.arm, seed_basis(seed), metric) for seed in arm.seeds]
        # Mean over the seeds of one query's value; the same order of terms for every query.
        mean = pl.sum_horizontal(seed_columns) / len(seed_columns)
        table = table.with_columns(mean.alias(value_column(arm.arm, SEED_MEAN, metric)))
    return table.sort(QUERY_ID)


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
    fold: str, queries: pl.DataFrame, groups: pl.DataFrame, arm_values: Sequence[pl.DataFrame]
) -> pl.DataFrame:
    """One row per LABELLED query of a fold with every arm's values, its customer and its group.

    Args:
        fold: Fold name.
        queries: The fold's query list: query_id, customer_idx.
        groups: query_id, query_group.
        arm_values: One table per arm (arm_fold_values): query_id and value columns.

    Returns:
        Columns query_id, fold, customer_idx, query_group, then the value
        columns of every arm; sorted by query_id.

    Raises:
        ValueError: If the arms were not evaluated on exactly the same queries,
            or a query has no customer or group.
    """
    table = arm_values[0]
    for values in arm_values[1:]:
        if table[QUERY_ID].to_list() != values[QUERY_ID].to_list():
            raise ValueError(f"{fold}: the arms were not evaluated on the same queries")
        table = table.join(values, on=QUERY_ID)
    value_columns = [name for name in table.columns if name != QUERY_ID]
    table = (
        table.join(queries.select(QUERY_ID, CUSTOMER_IDX), on=QUERY_ID, how="left")
        .join(groups, on=QUERY_ID, how="left")
        .with_columns(pl.lit(fold).alias(FOLD))
    )
    if table[CUSTOMER_IDX].null_count() > 0 or table[QUERY_GROUP].null_count() > 0:
        raise ValueError(f"{fold}: a query has no customer or no group")
    return table.select(QUERY_ID, FOLD, CUSTOMER_IDX, QUERY_GROUP, *value_columns).sort(QUERY_ID)


# ---------------------------------------------------------------- the arm table


def _mean(table: pl.DataFrame, column: str) -> float:
    """Mean of a column over the rows, summed with math.fsum in (fold, query_id) order."""
    values = table.sort([FOLD, QUERY_ID])[column].to_list()
    return math.fsum(values) / len(values)


def seed_summary(values: Sequence[float]) -> dict[str, float]:
    """Mean, sample standard deviation (n - 1) and range (max - min) / min of the seeds' values.

    Raises:
        ValueError: With fewer than two values or a non-positive minimum.
    """
    if len(values) < 2 or min(values) <= 0:
        raise ValueError("the summary needs at least two positive values")
    return {
        "mean": math.fsum(values) / len(values),
        "std": statistics.stdev(values),
        "range": (max(values) - min(values)) / min(values),
    }


def arm_rows(
    per_query: pl.DataFrame, arms: Sequence[ArmRuns], metrics: Sequence[str]
) -> list[dict[str, str]]:
    """The arm table: per arm one row per seed, then its mean, std and range rows.

    Args:
        per_query: The fold tables of all folds, concatenated.
        arms: The arms.
        metrics: The metrics, the main one first.

    Returns:
        One dict per (arm, row): arm, row ("seed_42", ..., "mean", "std",
        "range"), step_column, step, n_queries, one column per metric (pooled
        over all folds) and one "{metric}|{fold}" column per fold for the MAIN
        metric (empty in the std and range rows). Floats with repr().
    """
    folds = per_query[FOLD].unique(maintain_order=True).to_list()
    main = metrics[0]
    rows: list[dict[str, str]] = []
    for arm in arms:
        pooled: dict[str, list[float]] = {metric: [] for metric in metrics}
        by_fold: dict[str, list[float]] = {fold: [] for fold in folds}
        for seed in arm.seeds:
            basis = seed_basis(seed)
            row = {
                "arm": arm.arm,
                "row": basis,
                "step_column": arm.step_column,
                "step": str(arm.steps[seed]),
                "n_queries": str(per_query.height),
            }
            for metric in metrics:
                value = _mean(per_query, value_column(arm.arm, basis, metric))
                pooled[metric].append(value)
                row[metric] = repr(value)
            for fold in folds:
                in_fold = per_query.filter(pl.col(FOLD) == fold)
                value = _mean(in_fold, value_column(arm.arm, basis, main))
                by_fold[fold].append(value)
                row[f"{main}|{fold}"] = repr(value)
            rows.append(row)
        summaries = {metric: seed_summary(pooled[metric]) for metric in metrics}
        for name in ("mean", "std", "range"):
            row = {"arm": arm.arm, "row": name, "step_column": arm.step_column, "step": ""}
            row["n_queries"] = str(per_query.height)
            for metric in metrics:
                row[metric] = repr(summaries[metric][name])
            for fold in folds:
                is_mean = name == "mean"
                row[f"{main}|{fold}"] = (
                    repr(math.fsum(by_fold[fold]) / len(by_fold[fold])) if is_mean else ""
                )
            rows.append(row)
    return rows


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
    per_query: pl.DataFrame,
    first_column: str,
    second_column: str,
    n_replicates: int,
    seed: int,
    confidence_level: float,
) -> tuple[dict[str, Interval], int]:
    """Intervals of the pooled MAP of two arms and of "first minus second".

    Args:
        per_query: One row per query of the pooled folds: customer_idx and the
            two value columns (float64 AP of the query).
        first_column: The first arm's column.
        second_column: The second arm's column.
        n_replicates: Number of bootstrap replicates.
        seed: Seed of the draws (numpy default_rng).
        confidence_level: e.g. 0.95.

    Returns:
        ({"first", "second", "difference"} -> interval, the number of customers).
    """
    if not 0 < confidence_level < 1:
        raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}")
    # Per customer: number of queries and summed AP; sorted, so the same seed
    # always draws the same customers.
    totals = (
        per_query.group_by(CUSTOMER_IDX)
        .agg(
            pl.len().cast(pl.Int64).alias("n"),
            pl.col(first_column).sum().alias(FIRST),
            pl.col(second_column).sum().alias(SECOND),
        )
        .sort(CUSTOMER_IDX)
    )
    n_customers = totals.height
    if n_customers == 0 or n_replicates <= 0:
        raise ValueError("the bootstrap needs customers and at least one replicate")
    n_queries = totals["n"].to_numpy()
    sums = {name: totals[name].to_numpy().astype(np.float64) for name in (FIRST, SECOND)}
    total_queries = int(n_queries.sum())
    estimates = {name: math.fsum(sums[name].tolist()) / total_queries for name in sums}
    replicates = {name: np.empty(n_replicates, dtype=np.float64) for name in sums}
    generator = np.random.default_rng(seed)
    for index in range(n_replicates):
        drawn = generator.integers(0, n_customers, size=n_customers)
        weights = np.bincount(drawn, minlength=n_customers).astype(np.int64)
        drawn_queries = np.dot(weights, n_queries)
        for name in sums:
            # The SAME weights for both arms: the difference is a paired comparison.
            replicates[name][index] = float(np.dot(weights, sums[name]) / drawn_queries)
    difference = replicates[FIRST] - replicates[SECOND]
    intervals = {
        FIRST: _interval(estimates[FIRST], replicates[FIRST], confidence_level),
        SECOND: _interval(estimates[SECOND], replicates[SECOND], confidence_level),
        DIFFERENCE: _interval(estimates[FIRST] - estimates[SECOND], difference, confidence_level),
    }
    return intervals, n_customers


@dataclass(frozen=True)
class CompareSettings:
    """How the comparison is made.

    Attributes:
        metrics: The metrics of the arm table, the main one ("map@12") first.
        seed: The single seed of the "one seed" basis (the selection seed).
        recent_purchase_days: Largest number of days that still counts as recent.
        n_replicates: Bootstrap replicates.
        bootstrap_seed: Seed of the bootstrap draws.
        confidence_level: e.g. 0.95.
        wide_columns: The columns the wide tables must have.
    """

    metrics: tuple[str, ...]
    seed: int
    recent_purchase_days: int
    n_replicates: int
    bootstrap_seed: int
    confidence_level: float
    wide_columns: tuple[str, ...]

    @property
    def bases(self) -> tuple[str, ...]:
        """The two bases of the contrasts and the group table."""
        return (seed_basis(self.seed), SEED_MEAN)


def contrast_rows(
    per_query: pl.DataFrame,
    settings: CompareSettings,
    contrasts: Sequence[tuple[str, str]] = CONTRASTS,
) -> list[dict[str, str]]:
    """The bootstrap intervals of every contrast on both bases.

    Args:
        per_query: The fold tables of all folds, concatenated.
        settings: The bootstrap settings and the bases.
        contrasts: (first arm, second arm) pairs, "first minus second"; by
            default the four contrasts of the 2x2 study.

    Returns:
        One dict per (contrast, basis, quantity): first_arm, second_arm, basis,
        quantity ("first", "second", "difference"), estimate, ci_low, ci_high,
        confidence_level, n_replicates, bootstrap_seed, n_customers, n_queries.
    """
    metric = settings.metrics[0]
    rows: list[dict[str, str]] = []
    for first_arm, second_arm in contrasts:
        for basis in settings.bases:
            intervals, n_customers = paired_bootstrap(
                per_query,
                value_column(first_arm, basis, metric),
                value_column(second_arm, basis, metric),
                settings.n_replicates,
                settings.bootstrap_seed,
                settings.confidence_level,
            )
            for quantity in (FIRST, SECOND, DIFFERENCE):
                interval = intervals[quantity]
                rows.append(
                    {
                        "first_arm": first_arm,
                        "second_arm": second_arm,
                        "basis": basis,
                        "metric": metric,
                        "quantity": quantity,
                        "estimate": repr(interval.estimate),
                        "ci_low": repr(interval.low),
                        "ci_high": repr(interval.high),
                        "confidence_level": repr(settings.confidence_level),
                        "n_replicates": str(settings.n_replicates),
                        "bootstrap_seed": str(settings.bootstrap_seed),
                        "n_customers": str(n_customers),
                        "n_queries": str(per_query.height),
                    }
                )
    return rows


# ---------------------------------------------------------------- the group table


def group_rows(
    per_query: pl.DataFrame, arms: Sequence[str], bases: Sequence[str], metric: str
) -> list[dict[str, str]]:
    """MAP of every arm in every query group, per fold and pooled over all folds.

    Args:
        per_query: The fold tables of all folds, concatenated.
        arms: The arm names.
        bases: The bases, e.g. ("seed_42", "seed_mean").
        metric: The metric, "map@12".

    Returns:
        One dict per (basis, fold or "pooled", group or "all"): basis, fold,
        query_group, n_queries, query_share and one column per arm (as text;
        floats with repr()). A group without queries has no row.
    """
    rows: list[dict[str, str]] = []
    folds = per_query[FOLD].unique(maintain_order=True).to_list()
    for basis in bases:
        for scope in (*folds, POOLED):
            in_scope = per_query if scope == POOLED else per_query.filter(pl.col(FOLD) == scope)
            for group in (*GROUPS, ALL_QUERIES):
                table = (
                    in_scope
                    if group == ALL_QUERIES
                    else in_scope.filter(pl.col(QUERY_GROUP) == group)
                )
                if table.height == 0:
                    continue
                row = {
                    "basis": basis,
                    "fold": scope,
                    "query_group": group,
                    "n_queries": str(table.height),
                    "query_share": repr(table.height / in_scope.height),
                }
                for arm in arms:
                    row[arm] = repr(_mean(table, value_column(arm, basis, metric)))
                rows.append(row)
    return rows


# ---------------------------------------------------------------- one comparison, from files


@dataclass(frozen=True)
class FoldFiles:
    """The saved files of one fold that the comparison reads besides the arms' values.

    Attributes:
        fold: Fold name (a V fold).
        queries: The fold's query list (query_id, customer_idx).
        features_dir: The fold's frozen wide table (for the query groups).
    """

    fold: str
    queries: Path
    features_dir: Path


def read_fold_table(
    files: FoldFiles, arms: Sequence[ArmRuns], settings: CompareSettings
) -> pl.DataFrame:
    """The per-query table of one fold (fold_table), read from the saved files."""
    queries = pl.read_parquet(files.queries, columns=[QUERY_ID, CUSTOMER_IDX])
    days = read_query_days(files.features_dir, settings.wide_columns)
    values = [arm_fold_values(arm, files.fold, settings.metrics) for arm in arms]
    return fold_table(
        files.fold, queries, assign_groups(days, settings.recent_purchase_days), values
    )


@dataclass(frozen=True)
class Comparison:
    """The three tables of the comparison, as rows of text.

    Attributes:
        arms: The arm table (arm_rows).
        contrasts: The contrast table (contrast_rows).
        groups: The group table (group_rows).
        n_queries: Queries of all folds together.
    """

    arms: list[dict[str, str]]
    contrasts: list[dict[str, str]]
    groups: list[dict[str, str]]
    n_queries: int


def compare_arms(
    folds: Sequence[FoldFiles],
    arms: Sequence[ArmRuns],
    settings: CompareSettings,
    contrasts: Sequence[tuple[str, str]] = CONTRASTS,
) -> Comparison:
    """Compare the arms on the given folds.

    Args:
        folds: The saved files of every fold.
        arms: The arms, in the order of the tables' rows and columns.
        settings: Metrics, bases and bootstrap settings.
        contrasts: (first arm, second arm) pairs, "first minus second"; by
            default the four contrasts of the 2x2 study.

    Raises:
        ValueError: If an arm name repeats, a contrast names an arm that is not
            given, or an arm lacks the single seed.
    """
    names = tuple(arm.arm for arm in arms)
    if len(set(names)) != len(names):
        raise ValueError(f"an arm name repeats: {names}")
    unknown = [arm for pair in contrasts for arm in pair if arm not in names]
    if unknown:
        raise ValueError(f"contrasts name arms {unknown} that are not given; the arms are {names}")
    lacking = [arm.arm for arm in arms if settings.seed not in arm.seeds]
    if lacking:
        raise ValueError(f"arms {lacking} have no seed {settings.seed}")
    per_query = pl.concat([read_fold_table(files, arms, settings) for files in folds])
    return Comparison(
        arms=arm_rows(per_query, arms, settings.metrics),
        contrasts=contrast_rows(per_query, settings, contrasts),
        groups=group_rows(per_query, names, settings.bases, settings.metrics[0]),
        n_queries=per_query.height,
    )


def table_columns(rows: Sequence[Mapping[str, str]]) -> tuple[str, ...]:
    """The columns of a table of rows that all have the same keys (in the first row's order).

    Raises:
        ValueError: If there is no row or the rows have different keys.
    """
    if not rows:
        raise ValueError("the table has no rows")
    columns = tuple(rows[0])
    if any(tuple(row) != columns for row in rows):
        raise ValueError("the rows of a table must all have the same columns")
    return columns
