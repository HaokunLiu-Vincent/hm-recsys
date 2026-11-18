"""Tests for hmrec.ranker.ablation: the system without R6 (proposal §9.5).

Hand example, rrf_k = 60 (ranks: pop, rep, sib, icf, wals, tt):
    row A  only R6 (tt_rank 3)            -> removed: without R6 the row would not exist
    row B  R2 rank 2 and R6 rank 1        before: n_channels 2, best_rank 1, rrf 1/62 + 1/61
                                          after:  n_channels 1, best_rank 2, rrf 1/62
    row C  R1 rank 5 and R4 rank 10       unchanged: n_channels 2, best_rank 5, rrf 1/65 + 1/70
"""

import numpy as np
import polars as pl
import pytest
from ranker_example import FEATURE_NAMES, LOCK, wide_table, with_aggregates

from hmrec.ranker.ablation import (
    FULL,
    MINUS_R6,
    R6_FEATURES,
    build_experiment,
    drop_r6,
    wide_table_columns,
)

WIDE = with_aggregates(
    wide_table(
        [
            {"query_id": 1, "article_idx": 1, "tt_rank": 3.0, "tt_dot": 0.9, "wals_dot": 0.1},
            {"query_id": 1, "article_idx": 2, "rep_rank": 2.0, "tt_rank": 1.0, "tt_dot": 0.8},
            {"query_id": 1, "article_idx": 3, "pop_rank": 5.0, "icf_rank": 10.0, "tt_dot": 0.2},
        ]
    )
)


def test_the_wide_example_has_the_expected_aggregates_before() -> None:
    assert WIDE["n_channels"].to_list() == [1.0, 2.0, 2.0]
    assert WIDE["best_rank"].to_list() == [3.0, 1.0, 5.0]
    expected = np.array([1 / 63, 1 / 62 + 1 / 61, 1 / 65 + 1 / 70], dtype=np.float32)
    assert WIDE["rrf_score"].to_list() == expected.tolist()


def test_drop_r6_hand_example() -> None:
    result = drop_r6(WIDE, rrf_k=60)
    # Row A (only R6) is gone; B and C stay in order.
    assert result["article_idx"].to_list() == [2, 3]
    assert result["n_channels"].to_list() == [1.0, 2.0]
    assert result["best_rank"].to_list() == [2.0, 5.0]
    expected = np.array([1 / 62, 1 / 65 + 1 / 70], dtype=np.float32)
    assert result["rrf_score"].to_list() == expected.tolist()
    for name in ("n_channels", "best_rank", "rrf_score"):
        assert result[name].dtype == pl.Float32


def test_drop_r6_removes_the_four_r6_features_and_keeps_the_order() -> None:
    assert R6_FEATURES == ("tt_rank", "tt_dot", "tt_sim_max_hist", "tt_sim_top5_hist")
    result = drop_r6(WIDE, rrf_k=60)
    remaining = [name for name in WIDE.columns if name not in R6_FEATURES]
    assert result.columns == remaining
    # Every other column is untouched (wals_dot of row A went with the row).
    assert result["pop_rank"].to_list()[1] == 5.0
    assert result["rep_rank"].to_list()[0] == 2.0


def test_experiments_feature_lists() -> None:
    full = build_experiment(FULL, LOCK)
    assert full.feature_names == FEATURE_NAMES and len(full.feature_names) == 71
    assert full.categorical_names == LOCK.categorical_features
    minus = build_experiment(MINUS_R6, LOCK)
    assert len(minus.feature_names) == 67
    assert not set(minus.feature_names) & set(R6_FEATURES)
    assert minus.feature_names == tuple(n for n in FEATURE_NAMES if n not in R6_FEATURES)
    assert minus.categorical_names == LOCK.categorical_features  # none of them is an R6 feature
    assert minus.rrf_k == 60
    assert wide_table_columns(LOCK) == ("query_id", "article_idx", *FEATURE_NAMES)


def test_apply_selects_keys_and_features() -> None:
    full_rows = build_experiment(FULL, LOCK).apply(WIDE)
    assert full_rows.equals(WIDE)  # nothing changes for the full system
    minus_rows = build_experiment(MINUS_R6, LOCK).apply(WIDE)
    assert minus_rows.height == 2 and len(minus_rows.columns) == 2 + 67


def test_unknown_experiment_is_refused() -> None:
    with pytest.raises(ValueError, match="experiment must be one of"):
        build_experiment("minus_r5", LOCK)
