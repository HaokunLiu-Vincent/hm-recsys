"""Experiments: which candidate rows and which features a Ranker sees (proposal §9.5).

    full      every candidate row and all 71 locked features
    minus_r6  the system WITHOUT R6 Two-Tower

"minus_r6" is computed from the frozen wide table (the candidates are not
generated again):
    1. rows returned ONLY by R6 are removed (their other five rank columns are all NaN);
    2. the four R6 features are removed: tt_rank, tt_dot, tt_sim_max_hist, tt_sim_top5_hist;
    3. n_channels, best_rank and rrf_score are computed again from the other five rank
       columns, with the same expressions as the wide table (hmrec.fusion.rrf) and the
       locked rrf_k. Without this step they would still carry R6's rank.
The same function is used when sampling training rows and when scoring a fold.
"""

from dataclasses import dataclass

import polars as pl

from hmrec.fusion.rrf import N_CHANNELS, best_rank_expr, n_channels_expr, rrf_score_expr
from hmrec.fusion.union import RANK_COLUMNS, rank_column
from hmrec.locks import FeaturesLock

QUERY_ID = "query_id"
ARTICLE_IDX = "article_idx"
KEYS = (QUERY_ID, ARTICLE_IDX)

FULL = "full"
MINUS_R6 = "minus_r6"
EXPERIMENTS = (FULL, MINUS_R6)

R6_CHANNEL = "r6_twotower"
# Every feature that exists only because of R6 (proposal §8.2, §8.3).
R6_FEATURES = (rank_column(R6_CHANNEL), "tt_dot", "tt_sim_max_hist", "tt_sim_top5_hist")


def drop_r6(wide: pl.DataFrame, rrf_k: int) -> pl.DataFrame:
    """Turn wide-table rows into the rows of a system without R6 (module docstring 1-3).

    Args:
        wide: One row per (query_id, article_idx) with the locked feature columns.
        rrf_k: The constant of rrf_score (the locked value, 60).

    Returns:
        The rows that at least one of the other five channels returned, in the
        same order, without the R6 features. n_channels, best_rank and rrf_score
        keep their column positions and dtype (float32) but hold the new values.
    """
    remaining_ranks = [name for name in RANK_COLUMNS if name != rank_column(R6_CHANNEL)]
    # with_columns keeps the column order: the three aggregates are replaced in place.
    recomputed = wide.with_columns(
        n_channels_expr(remaining_ranks),
        best_rank_expr(remaining_ranks),
        rrf_score_expr(remaining_ranks, rrf_k),
    )
    # n_channels = 0 means only R6 returned the article: the row would not exist without R6.
    return recomputed.filter(pl.col(N_CHANNELS) > 0).drop(R6_FEATURES)


@dataclass(frozen=True)
class Experiment:
    """One experiment's view of the wide table.

    Attributes:
        name: "full" or "minus_r6".
        feature_names: The model's features in column order (71, or 67 without R6).
        categorical_names: The categorical features among them, in column order.
        rrf_k: The locked rrf_k (needed to recompute rrf_score for minus_r6).
    """

    name: str
    feature_names: tuple[str, ...]
    categorical_names: tuple[str, ...]
    rrf_k: int

    def apply(self, wide: pl.DataFrame) -> pl.DataFrame:
        """The experiment's rows and columns of a wide-table chunk (keys + features)."""
        rows = drop_r6(wide, self.rrf_k) if self.name == MINUS_R6 else wide
        return rows.select(*KEYS, *self.feature_names)


def wide_table_columns(lock: FeaturesLock) -> tuple[str, ...]:
    """The columns of the frozen wide table: query_id, article_idx, then the locked features."""
    return (*KEYS, *(feature.name for feature in lock.features))


def build_experiment(name: str, lock: FeaturesLock) -> Experiment:
    """The experiment `name` with the feature list of configs/locks/features.lock.yaml.

    Raises:
        ValueError: For an unknown experiment name.
    """
    if name not in EXPERIMENTS:
        raise ValueError(f"experiment must be one of {EXPERIMENTS}, got {name!r}")
    removed = set(R6_FEATURES) if name == MINUS_R6 else set()
    feature_names = tuple(feature.name for feature in lock.features if feature.name not in removed)
    categorical_names = tuple(
        feature for feature in lock.categorical_features if feature not in removed
    )
    return Experiment(name, feature_names, categorical_names, lock.rrf_k)
