"""Tests for fusion.union, fusion.rrf and features.source (group A).

Candidates of three queries (rank; score for pop / rep / sib / icf; pop_is_fallback):
    pop   q1: art 0 (rank 1, 2.0, own), art 2 (rank 2, 1.0, fill-in); q2: art 2 (1, 3.0, own);
          q3: art 0 (1, 1.0, fill-in)
    rep   q1: art 0 (1, 0.5)
    sib   q1: art 1 (1, 0.7)
    icf   q1: art 2 (1, 0.4), art 0 (2, 0.1)
    wals  q1: art 0 (rank 3)
    tt    q1: art 0 (rank 1), art 2 (rank 5); q2: art 2 (rank 1); q3: art 0 (rank 1), art 3 (rank 2)

Fused rows (one per distinct (query, article)), with k = 60:
    (q1, 0): pop 1, rep 1, icf 2, wals 3, tt 1 -> n 5, best 1, rrf 3/61 + 1/62 + 1/63 = 0.081183
    (q1, 1): sib 1                             -> n 1, best 1, rrf 1/61           = 0.016393
    (q1, 2): pop 2, icf 1, tt 5                -> n 3, best 1, rrf 1/62 + 1/61 + 1/65 = 0.047907
    (q2, 2): pop 1, tt 1                       -> n 2, best 1, rrf 2/61           = 0.032787
    (q3, 0): pop 1, tt 1                       -> n 2, best 1, rrf 2/61
    (q3, 3): tt 2                              -> n 1, best 2, rrf 1/62           = 0.016129
RRF baseline of q1: art 0 (0.0812), art 2 (0.0479), art 1 (0.0164).
"""

import math

import polars as pl
import pytest

from hmrec.features.source import add_source_features
from hmrec.fusion.rrf import rrf_baseline, rrf_score_expr
from hmrec.fusion.union import (
    CHANNEL_PREFIXES,
    RANK_COLUMNS,
    SCORE_COLUMNS,
    channel_columns,
    fuse_candidates,
)

Row = tuple[int, int, int, float]  # (query_id, article_idx, rank, score)


def _candidates(rows: list[Row], fallback: list[int] | None = None) -> pl.DataFrame:
    table = pl.DataFrame(
        {
            "query_id": [row[0] for row in rows],
            "article_idx": [row[1] for row in rows],
            "channel": ["x"] * len(rows),
            "rank": [row[2] for row in rows],
            "score": [row[3] for row in rows],
        },
        schema={
            "query_id": pl.Int32,
            "article_idx": pl.Int32,
            "channel": pl.String,
            "rank": pl.Int32,
            "score": pl.Float32,
        },
    )
    if fallback is not None:
        table = table.with_columns(pl.Series("pop_is_fallback", fallback, dtype=pl.Int8))
    return table


CANDIDATES = {
    "r1_popularity": _candidates(
        [(1, 0, 1, 2.0), (1, 2, 2, 1.0), (2, 2, 1, 3.0), (3, 0, 1, 1.0)], fallback=[0, 1, 0, 1]
    ),
    "r2_repurchase": _candidates([(1, 0, 1, 0.5)]),
    "r3_sibling": _candidates([(1, 1, 1, 0.7)]),
    "r4_itemcf": _candidates([(1, 2, 1, 0.4), (1, 0, 2, 0.1)]),
    "r5_wals": _candidates([(1, 0, 3, 9.0)]),
    "r6_twotower": _candidates(
        [(1, 0, 1, 8.0), (1, 2, 5, 7.0), (2, 2, 1, 6.0), (3, 0, 1, 5.0), (3, 3, 2, 4.0)]
    ),
}
NAN = float("nan")


def _row(table: pl.DataFrame, query_id: int, article_idx: int) -> dict:
    rows = table.filter((pl.col("query_id") == query_id) & (pl.col("article_idx") == article_idx))
    assert rows.height == 1
    return rows.to_dicts()[0]


def test_an_article_from_several_channels_is_one_row() -> None:
    fused = fuse_candidates(CANDIDATES)
    # 14 candidate rows over the six channels, 6 distinct (query, article) pairs.
    assert sum(table.height for table in CANDIDATES.values()) == 14
    assert fused.select("query_id", "article_idx").rows() == [
        (1, 0),
        (1, 1),
        (1, 2),
        (2, 2),
        (3, 0),
        (3, 3),
    ]
    assert not fused.select("query_id", "article_idx").is_duplicated().any()


def test_each_channels_columns_and_nan_where_it_did_not_return_the_article() -> None:
    fused = fuse_candidates(CANDIDATES)
    assert fused.columns == [
        "query_id",
        "article_idx",
        *RANK_COLUMNS,
        *SCORE_COLUMNS,
        "pop_is_fallback",
    ]
    for name in fused.columns[2:]:
        assert str(fused.schema[name]) == "Float32" and fused[name].null_count() == 0, name
    first = _row(fused, 1, 0)  # returned by pop, rep, icf, wals, tt; not by sib
    ranks = [first[name] for name in RANK_COLUMNS]
    assert ranks[:2] == [1.0, 1.0] and math.isnan(ranks[2]) and ranks[3:] == [2.0, 3.0, 1.0]
    assert first["pop_score"] == 2.0 and first["rep_score"] == 0.5
    assert math.isnan(first["sib_score"]) and first["icf_score"] == pytest.approx(0.1)
    assert first["pop_is_fallback"] == 0.0
    # R1's fill-in row keeps its flag; a row R1 did not return has NaN, not 0.
    assert _row(fused, 1, 2)["pop_is_fallback"] == 1.0
    only_tt = _row(fused, 3, 3)
    assert math.isnan(only_tt["pop_is_fallback"]) and math.isnan(only_tt["pop_rank"])
    assert only_tt["tt_rank"] == 2.0
    # WALS and Two-Tower scores are not A-group columns (they are wals_dot / tt_dot).
    assert "wals_score" not in fused.columns and "tt_score" not in fused.columns


def test_n_channels_best_rank_and_rrf_score_hand_computed() -> None:
    wide = add_source_features(fuse_candidates(CANDIDATES), rrf_k=60)
    expected = {
        (1, 0): (5, 1, 3 / 61 + 1 / 62 + 1 / 63),
        (1, 1): (1, 1, 1 / 61),
        (1, 2): (3, 1, 1 / 62 + 1 / 61 + 1 / 65),
        (2, 2): (2, 1, 2 / 61),
        (3, 0): (2, 1, 2 / 61),
        (3, 3): (1, 2, 1 / 62),
    }
    for (query_id, article_idx), (n_channels, best_rank, rrf_score) in expected.items():
        row = _row(wide, query_id, article_idx)
        assert (row["n_channels"], row["best_rank"]) == (n_channels, best_rank)
        assert row["rrf_score"] == pytest.approx(rrf_score, rel=1e-6)
    for name in ("n_channels", "best_rank", "rrf_score"):
        assert str(wide.schema[name]) == "Float32"


def test_rrf_baseline_order_and_tie_breaks() -> None:
    wide = add_source_features(fuse_candidates(CANDIDATES), rrf_k=60)
    top = rrf_baseline(wide, top_k=2)
    assert top.columns == ["query_id", "article_idx", "rank", "score"]
    # q1: art 0 (0.0812) > art 2 (0.0479) > art 1 (0.0164, cut by top_k = 2).
    assert top.filter(pl.col("query_id") == 1).select("article_idx", "rank").rows() == [
        (0, 1),
        (2, 2),
    ]
    # q3: art 0 (2/61) before art 3 (1/62).
    assert top.filter(pl.col("query_id") == 3)["article_idx"].to_list() == [0, 3]
    # Equal rrf_score: the better best_rank first; still equal: the smaller article_idx.
    tied = pl.DataFrame(
        {
            "query_id": [7, 7, 7],
            "article_idx": [30, 20, 10],
            "rrf_score": [0.5, 0.5, 0.5],
            "best_rank": [2.0, 2.0, 1.0],
        },
        schema={
            "query_id": pl.Int32,
            "article_idx": pl.Int32,
            "rrf_score": pl.Float32,
            "best_rank": pl.Float32,
        },
    )
    assert rrf_baseline(tied, top_k=3)["article_idx"].to_list() == [10, 20, 30]
    with pytest.raises(ValueError, match="top_k must be > 0"):
        rrf_baseline(tied, top_k=0)


def test_rrf_uses_the_configured_k() -> None:
    fused = fuse_candidates(CANDIDATES)
    with_k_10 = fused.with_columns(rrf_score_expr(RANK_COLUMNS, 10))
    # (q3, 3): only tt rank 2 -> 1 / (10 + 2).
    assert _row(with_k_10, 3, 3)["rrf_score"] == pytest.approx(1 / 12, rel=1e-6)
    with pytest.raises(ValueError, match="rrf_k must be > 0"):
        rrf_score_expr(RANK_COLUMNS, 0)


def test_wrong_channels_and_repeated_pairs_are_refused() -> None:
    without_r6 = {name: table for name, table in CANDIDATES.items() if name != "r6_twotower"}
    with pytest.raises(ValueError, match="expected the channels"):
        fuse_candidates(without_r6)
    assert list(CHANNEL_PREFIXES.values()) == ["pop", "rep", "sib", "icf", "wals", "tt"]
    repeated = _candidates([(1, 0, 1, 0.5), (1, 0, 2, 0.4)])
    with pytest.raises(ValueError, match="repeats in the candidates"):
        channel_columns("r2_repurchase", repeated)
