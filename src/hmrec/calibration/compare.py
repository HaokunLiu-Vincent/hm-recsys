"""How a container result is compared with a stage-1 record (docs/m10_calibration_plan.md §5).

Pure functions: tables, arrays and text in, numbers out. Nothing here reads a
file, a label or a configuration, and nothing here decides anything: the
numbers are written to results/m10_calibration/comparisons.csv and the user
decides the tolerance.

Definitions used by every item:

    relative difference   (measured - recorded) / recorded
    pair                  one (query_id, article_idx) of a candidates table. A
                          week has one query per customer, so this is a
                          (customer, article) pair.
    row                   one (query_id, article_idx, rank) of a candidates table

This module imports neither torch nor lightgbm.
"""

from dataclasses import dataclass

import numpy as np
import polars as pl

QUERY_ID = "query_id"
ARTICLE_IDX = "article_idx"
RANK = "rank"
SCORE = "score"
PAIR = [QUERY_ID, ARTICLE_IDX]
TREE_PREFIX = "Tree="
END_OF_TREES = "end of trees"


def relative_difference(measured: float, recorded: float) -> float:
    """(measured - recorded) / recorded.

    Raises:
        ValueError: If recorded is 0 (the ratio is not defined).
    """
    if recorded == 0:
        raise ValueError("the relative difference is not defined for a recorded value of 0")
    return (measured - recorded) / recorded


def within_relative(measured: float, recorded: float, limit: float) -> bool:
    """True if |relative difference| <= limit."""
    return abs(relative_difference(measured, recorded)) <= limit


# ---------------------------------------------------------------- candidates


@dataclass(frozen=True)
class CandidateComparison:
    """Two candidates tables of the same channel and week, compared.

    Attributes:
        recorded_rows: Rows of the stage-1 table.
        measured_rows: Rows of the container's table.
        rank_rows_identical: True if both hold exactly the same
            (query_id, article_idx, rank) rows.
        differing_rank_rows: Rows that are in exactly one of the two tables:
            recorded rows missing in the container plus container rows that are
            not in the record. 0 exactly when rank_rows_identical.
        recorded_pairs_present_share: Recorded pairs that are also in the
            container's table, divided by the recorded pairs.
        new_pairs_share: Container pairs that are NOT in the record, divided by
            the container's pairs.
        common_pairs: Pairs in both tables.
        common_pairs_same_rank_share: Among the common pairs, the share with the
            same rank in both tables (1.0 if there is no common pair).
        scores_bit_identical: True if both tables hold the same pairs and every
            pair has the same float32 score, bit for bit.
        max_abs_score_difference: Largest |measured - recorded| score over the
            common pairs, computed in float64 (0.0 if there is none).
        max_rel_score_difference: Largest |measured - recorded| / |recorded|
            over the common pairs whose recorded score is not 0 (0.0 if there
            is none). A pair with a recorded score of 0 counts only in the
            absolute difference.
    """

    recorded_rows: int
    measured_rows: int
    rank_rows_identical: bool
    differing_rank_rows: int
    recorded_pairs_present_share: float
    new_pairs_share: float
    common_pairs: int
    common_pairs_same_rank_share: float
    scores_bit_identical: bool
    max_abs_score_difference: float
    max_rel_score_difference: float


def _require_unique_pairs(table: pl.DataFrame, name: str) -> None:
    """Fail if a (query_id, article_idx) pair repeats: the counts below assume it does not."""
    if table.select(PAIR).is_duplicated().any():
        raise ValueError(f"the {name} candidates repeat a (query_id, article_idx) pair")


def compare_candidates(recorded: pl.DataFrame, measured: pl.DataFrame) -> CandidateComparison:
    """Compare the container's candidates of a channel with the stage-1 candidates.

    Args:
        recorded: The stage-1 table. One row per (query_id, article_idx);
            columns query_id (int32), article_idx (int32), rank (int32), score (float32).
        measured: The container's table, same grain and columns.

    Returns:
        The comparison (see CandidateComparison for every definition).

    Raises:
        ValueError: If a table is empty or repeats a pair.
    """
    if recorded.height == 0 or measured.height == 0:
        raise ValueError("cannot compare an empty candidates table")
    _require_unique_pairs(recorded, "recorded")
    _require_unique_pairs(measured, "measured")
    left = recorded.select(
        *PAIR, pl.col(RANK).alias("recorded_rank"), pl.col(SCORE).alias("recorded_score")
    )
    right = measured.select(
        *PAIR, pl.col(RANK).alias("measured_rank"), pl.col(SCORE).alias("measured_score")
    )
    # Inner join: one row per pair that both tables hold.
    common = left.join(right, on=PAIR, how="inner")
    same_rank = common.filter(pl.col("recorded_rank") == pl.col("measured_rank")).height
    # A (pair, rank) row is in both tables exactly when the pair is common and the rank
    # is the same. Every other row of either table is a differing row.
    differing = (recorded.height - same_rank) + (measured.height - same_rank)
    max_abs, max_rel = _score_differences(common)
    same_pairs = common.height == recorded.height == measured.height
    return CandidateComparison(
        recorded_rows=recorded.height,
        measured_rows=measured.height,
        rank_rows_identical=differing == 0,
        differing_rank_rows=differing,
        recorded_pairs_present_share=common.height / recorded.height,
        new_pairs_share=(measured.height - common.height) / measured.height,
        common_pairs=common.height,
        common_pairs_same_rank_share=same_rank / common.height if common.height else 1.0,
        scores_bit_identical=same_pairs and _same_bits(common, "recorded_score", "measured_score"),
        max_abs_score_difference=max_abs,
        max_rel_score_difference=max_rel,
    )


def _same_bits(table: pl.DataFrame, first: str, second: str) -> bool:
    """True if two numeric columns of a table hold the same bytes (so 0.0 != -0.0)."""
    return table[first].to_numpy().tobytes() == table[second].to_numpy().tobytes()


def _score_differences(common: pl.DataFrame) -> tuple[float, float]:
    """(largest absolute, largest relative) score difference over the common pairs."""
    if common.height == 0:
        return 0.0, 0.0
    # float64 for the difference: the scores are float32, and their difference is exact
    # in float64, so a difference in the last bit of a float32 is not rounded away.
    recorded = common["recorded_score"].to_numpy().astype(np.float64)
    measured = common["measured_score"].to_numpy().astype(np.float64)
    absolute = np.abs(measured - recorded)
    not_zero = recorded != 0
    relative = absolute[not_zero] / np.abs(recorded[not_zero])
    max_relative = float(relative.max()) if relative.size else 0.0
    return float(absolute.max()), max_relative


# ---------------------------------------------------------------- vectors and parameters


def max_abs_matrix_difference(
    recorded_index: np.ndarray,
    recorded_matrix: np.ndarray,
    measured_index: np.ndarray,
    measured_matrix: np.ndarray,
) -> float:
    """Largest |measured - recorded| entry of two matrices with the same row ids.

    Args:
        recorded_index: int32 [n_rows]: the id (customer_idx or article_idx) of each row.
        recorded_matrix: float32 [n_rows, n_factors].
        measured_index: int32 [n_rows], must equal recorded_index.
        measured_matrix: float32 [n_rows, n_factors].

    Returns:
        The largest absolute difference, computed in float64.

    Raises:
        ValueError: If the row ids or the shapes differ: then the two matrices
            do not describe the same users or articles.
    """
    if not np.array_equal(recorded_index, measured_index):
        raise ValueError("the two matrices have different row ids")
    if recorded_matrix.shape != measured_matrix.shape:
        raise ValueError(
            f"the two matrices have shapes {recorded_matrix.shape} and {measured_matrix.shape}"
        )
    difference = measured_matrix.astype(np.float64) - recorded_matrix.astype(np.float64)
    return float(np.abs(difference).max()) if difference.size else 0.0


# ---------------------------------------------------------------- LightGBM model text


def split_model_trees(model_text: str) -> list[str]:
    """The trees of a LightGBM model file, as text, in order.

    A model file has a header, then one block per tree that starts with a line
    "Tree=<k>", then the line "end of trees" and the feature importances.

    Args:
        model_text: The content of a model.txt written by LightGBM.

    Returns:
        One string per tree: its lines from "Tree=<k>" up to the next tree.

    Raises:
        ValueError: If the text has no tree or no "end of trees" line.
    """
    lines = model_text.splitlines()
    starts = [index for index, line in enumerate(lines) if line.startswith(TREE_PREFIX)]
    if not starts or END_OF_TREES not in lines:
        raise ValueError("the text is not a LightGBM model: no tree or no 'end of trees' line")
    ends = [*starts[1:], lines.index(END_OF_TREES)]
    return ["\n".join(lines[start:end]) for start, end in zip(starts, ends, strict=True)]


def leading_identical_trees(recorded_trees: list[str], measured_trees: list[str]) -> int:
    """How many trees at the start of two models are the same text.

    Returns:
        The index of the first tree that differs (trees are numbered from 0),
        which is also the number of identical leading trees. If one model is a
        prefix of the other, the length of the shorter one.
    """
    count = 0
    for recorded, measured in zip(recorded_trees, measured_trees, strict=False):
        if recorded != measured:
            break
        count += 1
    return count


# ---------------------------------------------------------------- Top-K lists


@dataclass(frozen=True)
class TopKComparison:
    """The Top-K lists of two models for the same queries, compared.

    Attributes:
        n_queries: Queries compared.
        identical_list_share: Share of queries whose two lists hold the same
            articles in the same order.
        identical_set_share: Share of queries whose two lists hold the same
            articles in any order.
        mean_overlap: Mean over queries of |recorded ∩ measured| / |recorded|.
    """

    n_queries: int
    identical_list_share: float
    identical_set_share: float
    mean_overlap: float


def _lists_per_query(top: pl.DataFrame, name: str) -> pl.DataFrame:
    """One row per query: its articles as a list, ordered by rank."""
    return (
        top.sort(QUERY_ID, RANK)
        .group_by(QUERY_ID, maintain_order=True)
        .agg(pl.col(ARTICLE_IDX).alias(name))
    )


def compare_top_k(recorded: pl.DataFrame, measured: pl.DataFrame) -> TopKComparison:
    """Compare two Top-K tables query by query.

    Args:
        recorded: The stage-1 Top-K. One row per (query_id, rank); columns
            query_id, rank (1 = best), article_idx.
        measured: The container's Top-K, same grain and columns.

    Raises:
        ValueError: If the two tables do not hold the same queries.
    """
    lists = _lists_per_query(recorded, "recorded").join(
        _lists_per_query(measured, "measured"), on=QUERY_ID, how="full", coalesce=True
    )
    if lists["recorded"].null_count() > 0 or lists["measured"].null_count() > 0:
        raise ValueError("the two Top-K tables do not hold the same queries")
    overlap = pl.col("recorded").list.set_intersection(pl.col("measured")).list.len()
    size = pl.col("recorded").list.len()
    measured_size = pl.col("measured").list.len()
    per_query = lists.select(
        (pl.col("recorded") == pl.col("measured")).alias("same_list"),
        ((overlap == size) & (size == measured_size)).alias("same_set"),
        (overlap / size).alias("overlap"),
    )
    n_queries = per_query.height
    return TopKComparison(
        n_queries=n_queries,
        identical_list_share=int(per_query["same_list"].sum()) / n_queries,
        identical_set_share=int(per_query["same_set"].sum()) / n_queries,
        mean_overlap=float(per_query["overlap"].sum()) / n_queries,
    )


# ---------------------------------------------------------------- bit identity of two tables


def tables_bit_identical(first: pl.DataFrame, second: pl.DataFrame) -> tuple[bool, str]:
    """Are two tables the same: schema, row order and every value, bit for bit.

    Used for the repeatability check of two container runs. Numeric columns are
    compared by their bytes (so 0.0 and -0.0 differ, and NaN equals NaN with the
    same bits); other columns by value.

    Returns:
        (identical, what differs or how much was compared).
    """
    if first.schema != second.schema:
        return False, f"schemas differ: {dict(first.schema)} and {dict(second.schema)}"
    if first.height != second.height:
        return False, f"row counts differ: {first.height} and {second.height}"
    different = []
    for name, dtype in first.schema.items():
        if dtype.is_numeric():
            same = first[name].to_numpy().tobytes() == second[name].to_numpy().tobytes()
        else:
            same = first[name].equals(second[name])
        if not same:
            different.append(name)
    if different:
        return False, f"columns with different values: {different}"
    return True, f"{first.height} rows, {len(first.columns)} columns"


def arrays_bit_identical(first: np.ndarray, second: np.ndarray) -> tuple[bool, str]:
    """Are two arrays the same: dtype, shape and every value, bit for bit.

    Returns:
        (identical, what differs or how much was compared).
    """
    if first.dtype != second.dtype or first.shape != second.shape:
        return False, (
            f"dtype / shape differ: {first.dtype} {first.shape} and {second.dtype} {second.shape}"
        )
    if first.tobytes() != second.tobytes():
        return False, f"{int(np.count_nonzero(first != second))} of {first.size} values differ"
    return True, f"shape {first.shape}, dtype {first.dtype}"
