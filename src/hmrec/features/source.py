"""Group A: recall-source features, 14 columns (proposal §8.2).

    {pop, rep, sib, icf, wals, tt}_rank   each channel's rank (1 = best); NaN = not returned
    {pop, rep, sib, icf}_score            the channel's raw score; NaN = not returned
    pop_is_fallback                       R1: 1 = global fill-in row, 0 = own age bucket;
                                          NaN if R1 did not return the article
    n_channels                            how many channels returned the article (1 ... 6)
    best_rank                             the best (smallest) rank over those channels
    rrf_score                             sum of 1 / (60 + rank) over those channels

The rank / score / fallback columns come from hmrec.fusion.union; the three
aggregates are added here with hmrec.fusion.rrf.
"""

import polars as pl

from hmrec.fusion.rrf import best_rank_expr, n_channels_expr, rrf_score_expr
from hmrec.fusion.union import RANK_COLUMNS


def add_source_features(fused: pl.DataFrame, rrf_k: int) -> pl.DataFrame:
    """Add n_channels, best_rank and rrf_score to a fused candidate table.

    Args:
        fused: Output of fusion.union.fuse_candidates (one row per
            (query_id, article_idx) with the six float32 rank columns).
        rrf_k: The constant of rrf_score (60).

    Returns:
        `fused` with the three float32 columns added; row order unchanged.
    """
    return fused.with_columns(
        n_channels_expr(RANK_COLUMNS),
        best_rank_expr(RANK_COLUMNS),
        rrf_score_expr(RANK_COLUMNS, rrf_k),
    )
