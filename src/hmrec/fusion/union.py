"""Union + dedup of the six channel outputs (proposal §7.3).

Input: each channel's candidates of the same queries (one row per
(query_id, article_idx) within a channel, with rank and score).
Output: ONE row per (query_id, article_idx) over all channels. An article
returned by three channels is one row, not three (proposal §7.1, option C).
Each channel contributes columns, NaN where that channel did not return the article:

    {prefix}_rank   for all six channels: pop, rep, sib, icf, wals, tt   (float32)
    {prefix}_score  for pop, rep, sib, icf only                          (float32)
    pop_is_fallback R1's fill-in flag (0 / 1)                            (float32)

The WALS and Two-Tower scores are NOT carried here: they are the B-group
features wals_dot / tt_dot, computed for every candidate (proposal §8.2, §8.3).

This module reads candidate tables only. It does not import recall/ (the
channel names and prefixes are repeated here on purpose) and never sees labels.
"""

from collections.abc import Mapping

import polars as pl

QUERY_ID = "query_id"
ARTICLE_IDX = "article_idx"
RANK = "rank"
SCORE = "score"
KEYS = [QUERY_ID, ARTICLE_IDX]
POP_IS_FALLBACK = "pop_is_fallback"

# Channel (the candidate file name) -> column prefix of proposal §8.2, in the fixed
# order R1 ... R6. The order is also the summation order of rrf_score.
CHANNEL_PREFIXES: dict[str, str] = {
    "r1_popularity": "pop",
    "r2_repurchase": "rep",
    "r3_sibling": "sib",
    "r4_itemcf": "icf",
    "r5_wals": "wals",
    "r6_twotower": "tt",
}
# Channels whose raw score is an A-group feature (proposal §8.2).
SCORE_CHANNELS = ("r1_popularity", "r2_repurchase", "r3_sibling", "r4_itemcf")
FALLBACK_CHANNEL = "r1_popularity"
NAN = float("nan")


def rank_column(channel: str) -> str:
    """Name of a channel's rank column, e.g. "icf_rank"."""
    return f"{CHANNEL_PREFIXES[channel]}_rank"


def score_column(channel: str) -> str:
    """Name of a channel's score column, e.g. "icf_score"."""
    return f"{CHANNEL_PREFIXES[channel]}_score"


RANK_COLUMNS = tuple(rank_column(channel) for channel in CHANNEL_PREFIXES)
SCORE_COLUMNS = tuple(score_column(channel) for channel in SCORE_CHANNELS)


def channel_columns(channel: str, candidates: pl.DataFrame) -> pl.DataFrame:
    """One channel's candidates with the column names it has in the fused table.

    Args:
        channel: A key of CHANNEL_PREFIXES.
        candidates: The channel's output: query_id, article_idx, rank, score
            (R1 also pop_is_fallback).

    Returns:
        query_id, article_idx, {prefix}_rank (float32), and for the score
        channels {prefix}_score (float32); R1 also pop_is_fallback (float32).

    Raises:
        ValueError: If a (query_id, article_idx) pair repeats in the channel.
    """
    if candidates.select(KEYS).is_duplicated().any():
        raise ValueError(f"{channel}: a (query_id, article_idx) pair repeats in the candidates")
    columns = [
        pl.col(QUERY_ID),
        pl.col(ARTICLE_IDX),
        pl.col(RANK).cast(pl.Float32).alias(rank_column(channel)),
    ]
    if channel in SCORE_CHANNELS:
        columns.append(pl.col(SCORE).cast(pl.Float32).alias(score_column(channel)))
    if channel == FALLBACK_CHANNEL:
        columns.append(pl.col(POP_IS_FALLBACK).cast(pl.Float32))
    return candidates.select(columns)


def fuse_candidates(candidates_by_channel: Mapping[str, pl.DataFrame]) -> pl.DataFrame:
    """Merge the six channels into one row per (query_id, article_idx).

    Args:
        candidates_by_channel: Exactly the six channels of CHANNEL_PREFIXES ->
            their candidates for the same set of queries.

    Returns:
        One row per distinct (query_id, article_idx), sorted by
        (query_id, article_idx). Columns: query_id, article_idx (int32), the
        six rank columns, the four score columns and pop_is_fallback, all
        float32, NaN where the channel did not return the article.

    Raises:
        ValueError: If a channel is missing or unknown.
    """
    if list(candidates_by_channel) != list(CHANNEL_PREFIXES):
        raise ValueError(
            f"expected the channels {list(CHANNEL_PREFIXES)} in this order, "
            f"got {list(candidates_by_channel)}"
        )
    # Dedup at (query_id, article_idx): the same article from several channels is one row.
    pairs = pl.concat([c.select(KEYS) for c in candidates_by_channel.values()]).unique()
    fused = pairs
    for channel, candidates in candidates_by_channel.items():
        fused = fused.join(channel_columns(channel, candidates), on=KEYS, how="left")
    feature_columns = [*RANK_COLUMNS, *SCORE_COLUMNS, POP_IS_FALLBACK]
    # A left join leaves null where the channel did not return the pair; the wide
    # table uses NaN for "missing" (proposal §8.1).
    filled = [pl.col(name).fill_null(NAN) for name in feature_columns]
    return fused.select(KEYS + filled).sort(KEYS)
