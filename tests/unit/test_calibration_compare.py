"""Tests for hmrec.calibration.compare: how a container result is compared with stage 1.

Every expected value is computed by hand in the comments.
"""

import numpy as np
import polars as pl
import pytest

from hmrec.calibration.compare import (
    arrays_bit_identical,
    compare_candidates,
    compare_top_k,
    leading_identical_trees,
    max_abs_matrix_difference,
    relative_difference,
    split_model_trees,
    tables_bit_identical,
    within_relative,
)

# The float32 right above 1.0: 1 + 2**-23. The difference to 1.0 is 2**-23 exactly.
ONE_PLUS_ONE_BIT = float(np.nextafter(np.float32(1.0), np.float32(2.0)))
FLOAT32_STEP = 2.0**-23


def _candidates(rows: list[tuple[int, int, int, float]]) -> pl.DataFrame:
    """(query_id, article_idx, rank, score) rows as a candidates table."""
    return pl.DataFrame(
        rows,
        schema={
            "query_id": pl.Int32,
            "article_idx": pl.Int32,
            "rank": pl.Int32,
            "score": pl.Float32,
        },
        orient="row",
    )


RECORDED = _candidates([(1, 10, 1, 0.5), (1, 11, 2, 0.25), (1, 12, 3, 0.125), (2, 10, 1, 1.0)])


def test_relative_difference_and_its_limit() -> None:
    # (0.099 - 0.1) / 0.1 = -0.01 (up to rounding of the decimal literals).
    assert relative_difference(0.099, 0.1) == pytest.approx(-0.01)
    assert relative_difference(3.0, 2.0) == 0.5
    assert within_relative(101.0, 100.0, 0.01)  # exactly 1% is inside
    assert not within_relative(98.0, 100.0, 0.01)  # 2% below is outside
    with pytest.raises(ValueError, match="not defined for a recorded value of 0"):
        relative_difference(1.0, 0.0)


def test_identical_candidates() -> None:
    # The same rows in another row order: the comparison is by key, not by position.
    result = compare_candidates(RECORDED, RECORDED.reverse())
    assert (result.recorded_rows, result.measured_rows, result.common_pairs) == (4, 4, 4)
    assert result.rank_rows_identical and result.differing_rank_rows == 0
    assert result.recorded_pairs_present_share == 1.0
    assert result.new_pairs_share == 0.0
    assert result.common_pairs_same_rank_share == 1.0
    assert result.scores_bit_identical
    assert (result.max_abs_score_difference, result.max_rel_score_difference) == (0.0, 0.0)


def test_candidates_that_differ() -> None:
    # Against RECORDED: query 1 lost article 11, article 12 moved from rank 3 to rank 2,
    # article 13 is new; query 2's score differs in its last bit.
    measured = _candidates(
        [(1, 10, 1, 0.5), (1, 12, 2, 0.125), (1, 13, 3, 0.0625), (2, 10, 1, ONE_PLUS_ONE_BIT)]
    )
    result = compare_candidates(RECORDED, measured)
    # Common pairs: (1,10), (1,12), (2,10) = 3. With the same rank: (1,10), (2,10) = 2.
    assert result.common_pairs == 3
    assert not result.rank_rows_identical
    # Rows in exactly one table: recorded (1,11,2), (1,12,3); measured (1,12,2), (1,13,3).
    assert result.differing_rank_rows == 4
    assert result.recorded_pairs_present_share == 3 / 4
    assert result.new_pairs_share == 1 / 4
    assert result.common_pairs_same_rank_share == 2 / 3
    assert not result.scores_bit_identical
    # The only score difference: 2**-23 at a recorded score of 1.0.
    assert result.max_abs_score_difference == FLOAT32_STEP
    assert result.max_rel_score_difference == FLOAT32_STEP


def test_same_ranks_but_one_score_bit_is_not_bit_identical() -> None:
    measured = _candidates(
        [(1, 10, 1, 0.5), (1, 11, 2, 0.25), (1, 12, 3, 0.125), (2, 10, 1, ONE_PLUS_ONE_BIT)]
    )
    result = compare_candidates(RECORDED, measured)
    # The (customer, article, rank) rows are the same; only a score differs.
    assert result.rank_rows_identical and result.differing_rank_rows == 0
    assert not result.scores_bit_identical
    assert result.max_abs_score_difference == FLOAT32_STEP


def test_a_recorded_score_of_zero_counts_only_in_the_absolute_difference() -> None:
    recorded = _candidates([(1, 10, 1, 0.0), (1, 11, 2, 2.0)])
    measured = _candidates([(1, 10, 1, 0.5), (1, 11, 2, 2.5)])
    result = compare_candidates(recorded, measured)
    assert result.max_abs_score_difference == 0.5
    # Only the pair with a recorded score of 2.0 has a relative difference: 0.5 / 2.0.
    assert result.max_rel_score_difference == 0.25


def test_no_common_pair() -> None:
    result = compare_candidates(RECORDED, _candidates([(3, 99, 1, 1.0)]))
    assert (result.common_pairs, result.recorded_pairs_present_share) == (0, 0.0)
    assert result.new_pairs_share == 1.0
    assert result.differing_rank_rows == 5  # all 4 recorded rows and the 1 measured row
    assert result.common_pairs_same_rank_share == 1.0  # defined as 1 when nothing is common
    assert not result.scores_bit_identical


def test_candidates_must_not_be_empty_or_repeat_a_pair() -> None:
    with pytest.raises(ValueError, match="empty candidates table"):
        compare_candidates(RECORDED, RECORDED.head(0))
    repeated = _candidates([(1, 10, 1, 0.5), (1, 10, 2, 0.25)])
    with pytest.raises(ValueError, match="the measured candidates repeat"):
        compare_candidates(RECORDED, repeated)
    with pytest.raises(ValueError, match="the recorded candidates repeat"):
        compare_candidates(repeated, RECORDED)


def test_max_abs_matrix_difference() -> None:
    index = np.array([3, 7], dtype=np.int32)
    recorded = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    measured = np.array([[1.0, 2.5], [2.0, 4.0]], dtype=np.float32)
    # Differences: 0, 0.5, 1.0, 0 -> the largest is 1.0.
    assert max_abs_matrix_difference(index, recorded, index, measured) == 1.0
    assert max_abs_matrix_difference(index, recorded, index, recorded) == 0.0
    with pytest.raises(ValueError, match="different row ids"):
        max_abs_matrix_difference(index, recorded, np.array([3, 8], dtype=np.int32), measured)
    with pytest.raises(ValueError, match="have shapes"):
        max_abs_matrix_difference(index, recorded, index, measured[:, :1])


MODEL_TEXT = "\n".join(
    [
        "tree",
        "version=v4",
        "",
        "Tree=0",
        "split_feature=1 2",
        "leaf_value=0.1 0.2 0.3",
        "",
        "Tree=1",
        "split_feature=3",
        "leaf_value=0.4 0.5",
        "",
        "Tree=2",
        "split_feature=4",
        "leaf_value=0.6 0.7",
        "",
        "end of trees",
        "",
        "feature_importances:",
        "a=3",
    ]
)


def test_split_model_trees_and_the_first_tree_that_differs() -> None:
    trees = split_model_trees(MODEL_TEXT)
    assert len(trees) == 3
    assert trees[0].splitlines()[:2] == ["Tree=0", "split_feature=1 2"]
    assert trees[2].splitlines()[:3] == ["Tree=2", "split_feature=4", "leaf_value=0.6 0.7"]
    assert leading_identical_trees(trees, trees) == 3
    # Another leaf value in tree 1: trees 0 is the same, tree 1 is the first that differs.
    changed = split_model_trees(MODEL_TEXT.replace("leaf_value=0.4 0.5", "leaf_value=0.4 0.6"))
    assert leading_identical_trees(trees, changed) == 1
    # A header that differs (another version line) does not count: only trees are compared.
    assert leading_identical_trees(trees, split_model_trees(MODEL_TEXT.replace("v4", "v5"))) == 3
    # A shorter model that is a prefix of the other: its length.
    assert leading_identical_trees(trees, trees[:2]) == 2
    with pytest.raises(ValueError, match="not a LightGBM model"):
        split_model_trees("no trees here")


def _top(rows: list[tuple[int, int, int]]) -> pl.DataFrame:
    """(query_id, rank, article_idx) rows as a Top-K table."""
    return pl.DataFrame(
        rows,
        schema={"query_id": pl.Int32, "rank": pl.Int32, "article_idx": pl.Int32},
        orient="row",
    )


def test_compare_top_k() -> None:
    recorded = _top(
        [
            (1, 1, 1),
            (1, 2, 2),
            (1, 3, 3),
            (2, 1, 4),
            (2, 2, 5),
            (2, 3, 6),
            (3, 1, 7),
            (3, 2, 8),
            (3, 3, 9),
        ]
    )
    # Query 1: the same list. Query 2: the same articles, two of them swapped.
    # Query 3: article 9 replaced by article 10 (2 of 3 in common).
    measured = _top(
        [
            (1, 1, 1),
            (1, 2, 2),
            (1, 3, 3),
            (2, 1, 5),
            (2, 2, 4),
            (2, 3, 6),
            (3, 1, 7),
            (3, 2, 8),
            (3, 3, 10),
        ]
    )
    result = compare_top_k(recorded, measured.sample(fraction=1.0, shuffle=True, seed=1))
    assert result.n_queries == 3
    assert result.identical_list_share == 1 / 3
    assert result.identical_set_share == 2 / 3
    # (3/3 + 3/3 + 2/3) / 3 = 8/9
    assert result.mean_overlap == pytest.approx(8 / 9)
    with pytest.raises(ValueError, match="do not hold the same queries"):
        compare_top_k(recorded, measured.filter(pl.col("query_id") != 3))


def test_tables_bit_identical() -> None:
    table = _candidates([(1, 10, 1, 0.5), (2, 10, 1, 1.0)]).with_columns(
        pl.lit("pop").alias("channel")
    )
    assert tables_bit_identical(table, table.clone()) == (True, "2 rows, 5 columns")
    one_bit = table.with_columns(
        pl.when(pl.col("query_id") == 2)
        .then(pl.lit(ONE_PLUS_ONE_BIT, dtype=pl.Float32))
        .otherwise(pl.col("score"))
        .alias("score")
    )
    assert tables_bit_identical(table, one_bit) == (
        False,
        "columns with different values: ['score']",
    )
    other_text = table.with_columns(pl.lit("rep").alias("channel"))
    assert (
        tables_bit_identical(table, other_text)[1] == "columns with different values: ['channel']"
    )
    # The same rows in another order are NOT the same table: the files must repeat exactly.
    assert not tables_bit_identical(table, table.reverse())[0]
    assert tables_bit_identical(table, table.head(1)) == (False, "row counts differ: 2 and 1")
    assert not tables_bit_identical(table, table.cast({"rank": pl.Int64}))[0]


def test_bit_identity_tells_zero_from_minus_zero() -> None:
    plus = np.array([0.0, 1.0], dtype=np.float32)
    minus = np.array([-0.0, 1.0], dtype=np.float32)
    assert arrays_bit_identical(plus, plus.copy()) == (True, "shape (2,), dtype float32")
    # 0.0 == -0.0 as numbers, but the bits differ: the two runs did not compute the same.
    assert not arrays_bit_identical(plus, minus)[0]
    assert not arrays_bit_identical(plus, plus.astype(np.float64))[0]
    assert arrays_bit_identical(plus, np.array([0.0, 2.0], dtype=np.float32)) == (
        False,
        "1 of 2 values differ",
    )
