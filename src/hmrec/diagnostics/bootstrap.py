"""Table D: confidence intervals of the pooled MAP@12 by a cluster bootstrap over customers.

The queries of the 4 folds of a phase are pooled. A customer who bought in
several of these weeks has several queries, and they are not independent. So
the unit that is drawn is the CUSTOMER:

    1. Per customer c: n_c = number of queries, S_c = sum of the AP of these queries.
    2. One replicate: draw n_customers customers WITH replacement. A drawn customer
       brings all of their queries, as often as they were drawn (weight w_c).
           pooled MAP* = sum_c w_c * S_c / sum_c w_c * n_c
       (the AP of the drawn queries, summed, divided by the number of drawn queries).
    3. Repeat n_replicates times; the interval is the alpha/2 and 1 - alpha/2
       quantiles of the replicates (alpha = 1 - confidence_level; numpy's default
       linear interpolation between order statistics).

"LambdaMART minus RRF baseline" uses the SAME draws for both systems (a paired
comparison): its replicate is the difference of the two systems' replicates.

What the interval means: the uncertainty from which customers happen to be in
the data. It does NOT show how much a result changes from week to week: the
same 4 weeks are in every replicate.

The draws come from numpy.random.default_rng(seed), so a fixed seed gives the
same intervals. Values are float64.
"""

import math
from dataclasses import dataclass

import numpy as np
import polars as pl

from hmrec.data.id_maps import CUSTOMER_IDX
from hmrec.diagnostics.query_groups import BASELINE_AP, RANKER_AP
from hmrec.diagnostics.records import float_text

N_QUERIES = "n_queries"
RANKER_SUM = "ranker_ap_sum"
BASELINE_SUM = "baseline_ap_sum"

LAMBDAMART_MAP = "lambdamart"
BASELINE_MAP = "rrf_baseline"
DIFFERENCE = "lambdamart_minus_rrf_baseline"
QUANTITIES = (LAMBDAMART_MAP, BASELINE_MAP, DIFFERENCE)
QUANTILE_DECIMALS = 12  # see _interval: removes float noise from (1 - level) / 2

BOOTSTRAP_COLUMNS: tuple[str, ...] = (
    "phase",
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
)


@dataclass(frozen=True)
class Interval:
    """A point estimate with its bootstrap interval.

    Attributes:
        estimate: The value on the real data (no resampling).
        low: Lower end of the interval (the alpha/2 quantile of the replicates).
        high: Upper end (the 1 - alpha/2 quantile).
    """

    estimate: float
    low: float
    high: float


def customer_totals(per_query: pl.DataFrame) -> pl.DataFrame:
    """Per customer: the number of queries and the summed AP of both systems.

    Args:
        per_query: One row per query of the pooled folds. Columns: customer_idx
            (int32), ranker_ap, baseline_ap (float64).

    Returns:
        One row per customer, sorted by customer_idx (a fixed order, so the same
        seed always draws the same customers). Columns: customer_idx, n_queries
        (int64), ranker_ap_sum, baseline_ap_sum (float64).
    """
    totals = per_query.group_by(CUSTOMER_IDX).agg(
        pl.len().cast(pl.Int64).alias(N_QUERIES),
        pl.col(RANKER_AP).sum().alias(RANKER_SUM),
        pl.col(BASELINE_AP).sum().alias(BASELINE_SUM),
    )
    return totals.sort(CUSTOMER_IDX)


def draw_customer_weights(n_customers: int, n_replicates: int, seed: int) -> list[np.ndarray]:
    """How often each customer is drawn in each replicate.

    Returns:
        n_replicates arrays of shape [n_customers], int64; each sums to
        n_customers (that many customers are drawn with replacement).
    """
    if n_customers <= 0 or n_replicates <= 0:
        raise ValueError("n_customers and n_replicates must be > 0")
    generator = np.random.default_rng(seed)
    weights = []
    for _ in range(n_replicates):
        drawn = generator.integers(0, n_customers, size=n_customers)
        weights.append(np.bincount(drawn, minlength=n_customers).astype(np.int64))
    return weights


def pooled_mean(weights: np.ndarray, sums: np.ndarray, n_queries: np.ndarray) -> float:
    """Pooled MAP of one replicate: sum_c w_c * S_c / sum_c w_c * n_c.

    Args:
        weights: Shape [n_customers], how often each customer was drawn.
        sums: Shape [n_customers], float64, S_c.
        n_queries: Shape [n_customers], int64, n_c (>= 1).
    """
    return float(np.dot(weights, sums) / np.dot(weights, n_queries))


def _interval(estimate: float, replicates: np.ndarray, confidence_level: float) -> Interval:
    """The percentile interval of the replicates around the real-data estimate."""
    # Rounded so that 0.95 gives exactly the quantiles 0.025 and 0.975: in floating
    # point 1 - 0.95 is 0.050000000000000044, not 0.05.
    low_quantile = round((1.0 - confidence_level) / 2.0, QUANTILE_DECIMALS)
    high_quantile = round((1.0 + confidence_level) / 2.0, QUANTILE_DECIMALS)
    low, high = np.quantile(replicates, [low_quantile, high_quantile])
    return Interval(estimate=estimate, low=float(low), high=float(high))


def cluster_bootstrap(
    totals: pl.DataFrame, n_replicates: int, seed: int, confidence_level: float
) -> dict[str, Interval]:
    """Intervals of the pooled MAP of both systems and of their difference.

    Args:
        totals: One row per customer (customer_totals).
        n_replicates: Number of bootstrap replicates (1000).
        seed: Seed of the draws.
        confidence_level: e.g. 0.95.

    Returns:
        Quantity ("lambdamart", "rrf_baseline", "lambdamart_minus_rrf_baseline")
        -> its estimate and interval.
    """
    if not 0 < confidence_level < 1:
        raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}")
    n_queries = totals[N_QUERIES].to_numpy()
    ranker_sums = totals[RANKER_SUM].to_numpy().astype(np.float64)
    baseline_sums = totals[BASELINE_SUM].to_numpy().astype(np.float64)
    total_queries = int(n_queries.sum())
    ranker_estimate = math.fsum(ranker_sums.tolist()) / total_queries
    baseline_estimate = math.fsum(baseline_sums.tolist()) / total_queries
    ranker_replicates = np.empty(n_replicates, dtype=np.float64)
    baseline_replicates = np.empty(n_replicates, dtype=np.float64)
    weights_per_replicate = draw_customer_weights(totals.height, n_replicates, seed)
    for index, weights in enumerate(weights_per_replicate):
        # The SAME weights for both systems: the difference below is a paired comparison.
        ranker_replicates[index] = pooled_mean(weights, ranker_sums, n_queries)
        baseline_replicates[index] = pooled_mean(weights, baseline_sums, n_queries)
    difference = ranker_replicates - baseline_replicates
    return {
        LAMBDAMART_MAP: _interval(ranker_estimate, ranker_replicates, confidence_level),
        BASELINE_MAP: _interval(baseline_estimate, baseline_replicates, confidence_level),
        DIFFERENCE: _interval(ranker_estimate - baseline_estimate, difference, confidence_level),
    }


def bootstrap_rows(
    phase: str,
    folds: tuple[str, ...],
    metric: str,
    totals: pl.DataFrame,
    intervals: dict[str, Interval],
    n_replicates: int,
    seed: int,
    confidence_level: float,
) -> list[dict[str, str]]:
    """The three rows of one phase for table D, in the order of QUANTITIES."""
    return [
        {
            "phase": phase,
            "folds": " ".join(folds),
            "metric": metric,
            "quantity": quantity,
            "estimate": float_text(intervals[quantity].estimate),
            "ci_low": float_text(intervals[quantity].low),
            "ci_high": float_text(intervals[quantity].high),
            "confidence_level": float_text(confidence_level),
            "n_replicates": str(n_replicates),
            "seed": str(seed),
            "n_customers": str(totals.height),
            "n_queries": str(int(totals[N_QUERIES].sum())),
        }
        for quantity in QUANTITIES
    ]
