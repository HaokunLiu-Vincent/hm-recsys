"""Tests for the shared recall output spec, ranking and schema check."""

from pathlib import Path

import polars as pl
import pytest

from hmrec.recall.base import (
    BASE_SCHEMA,
    OutputSpec,
    SortKey,
    rank_candidates,
    validate_candidates,
    write_candidates,
)
from hmrec.recall.r1_popularity import POP_SPEC

PLAIN = OutputSpec(channel="rep")


def _plain(rows: list[tuple[int, int, int, float]], channel: str = "rep") -> pl.DataFrame:
    """Rows of (query_id, article_idx, rank, score) in the base schema."""
    return pl.DataFrame(
        {
            "query_id": [row[0] for row in rows],
            "article_idx": [row[1] for row in rows],
            "channel": [channel] * len(rows),
            "rank": [row[2] for row in rows],
            "score": [row[3] for row in rows],
        },
        schema=BASE_SCHEMA,
    )


VALID = _plain([(1, 5, 1, 0.9), (1, 3, 2, 0.5), (1, 4, 3, 0.5), (2, 7, 1, 0.1)])


def _pop(rows: list[tuple[int, int, int, float, int]]) -> pl.DataFrame:
    """Rows of (query_id, article_idx, rank, score, pop_is_fallback)."""
    base = _plain([row[:4] for row in rows], channel="pop")
    return base.with_columns(pl.Series("pop_is_fallback", [row[4] for row in rows], pl.Int8))


# ---------------------------------------------------------------- valid tables


def test_valid_table_passes() -> None:
    validate_candidates(VALID, PLAIN, quota=3)


def test_rank_candidates_breaks_ties_by_article_and_cuts_quota() -> None:
    scored = pl.DataFrame(
        {"query_id": [1, 1, 1, 1], "article_idx": [9, 4, 6, 2], "score": [0.5, 0.5, 0.9, 0.1]}
    )
    ranked = rank_candidates(scored, PLAIN, quota=3)
    # 6 (0.9), then 4 and 9 tie at 0.5 -> article_idx ASC; 2 is cut by the quota.
    assert ranked["article_idx"].to_list() == [6, 4, 9]
    assert ranked["rank"].to_list() == [1, 2, 3]
    assert ranked.schema == BASE_SCHEMA


def test_rank_candidates_sorts_on_float32_scores() -> None:
    # 1.0 and 1.0 + 1e-12 differ in float64 but are equal in float32 (the stored
    # precision), so article_idx decides: 3 before 8.
    scored = pl.DataFrame({"query_id": [1, 1], "article_idx": [8, 3], "score": [1.0 + 1e-12, 1.0]})
    assert rank_candidates(scored, PLAIN, quota=5)["article_idx"].to_list() == [3, 8]


def test_pop_order_puts_fallback_last() -> None:
    # Own row with score 0.2 ranks above a fill-in row with score 0.9.
    valid = _pop([(1, 5, 1, 0.2, 0), (1, 3, 2, 0.9, 1)])
    validate_candidates(valid, POP_SPEC)
    wrong = _pop([(1, 3, 1, 0.9, 1), (1, 5, 2, 0.2, 0)])
    with pytest.raises(ValueError, match="pop_is_fallback ASC, score DESC, article_idx ASC"):
        validate_candidates(wrong, POP_SPEC)


# ---------------------------------------------------------------- spec declaration errors


def test_extra_column_needs_channel_prefix() -> None:
    with pytest.raises(ValueError, match="must start with 'pop_'"):
        OutputSpec(channel="pop", extra_columns={"is_fallback": pl.Int8()})


def test_extra_column_cannot_be_a_base_column() -> None:
    with pytest.raises(ValueError, match="is a base column"):
        OutputSpec(channel="pop", extra_columns={"score": pl.Float32()})


def test_sort_key_must_be_an_output_column() -> None:
    with pytest.raises(ValueError, match="sort key 'pop_x'"):
        OutputSpec(
            channel="pop",
            sort_keys=(SortKey("pop_x", False), SortKey("article_idx", False)),
        )


def test_sort_keys_must_end_with_article_ascending() -> None:
    with pytest.raises(ValueError, match="end with article_idx ASC"):
        OutputSpec(channel="rep", sort_keys=(SortKey("score", True),))


# ---------------------------------------------------------------- column errors


def test_undeclared_column_raises() -> None:
    with pytest.raises(ValueError, match=r"undeclared columns \['rep_extra'\]"):
        validate_candidates(VALID.with_columns(pl.lit(1).alias("rep_extra")), PLAIN)


def test_missing_declared_column_raises() -> None:
    with pytest.raises(ValueError, match=r"missing columns \['pop_is_fallback'\]"):
        validate_candidates(_plain([(1, 5, 1, 0.9)], channel="pop"), POP_SPEC)


def test_wrong_dtype_raises() -> None:
    as_float64 = VALID.with_columns(pl.col("score").cast(pl.Float64))
    with pytest.raises(ValueError, match="'score' has dtype Float64"):
        validate_candidates(as_float64, PLAIN)
    fallback_int32 = _pop([(1, 5, 1, 0.2, 0)]).with_columns(
        pl.col("pop_is_fallback").cast(pl.Int32)
    )
    with pytest.raises(ValueError, match="'pop_is_fallback' has dtype Int32"):
        validate_candidates(fallback_int32, POP_SPEC)


# ---------------------------------------------------------------- value errors


def test_null_in_extra_column_raises() -> None:
    with_null = _pop([(1, 5, 1, 0.2, 0)]).with_columns(
        pl.lit(None, dtype=pl.Int8).alias("pop_is_fallback")
    )
    with pytest.raises(ValueError, match=r"null values in columns \['pop_is_fallback'\]"):
        validate_candidates(with_null, POP_SPEC)


def test_nan_score_raises() -> None:
    with pytest.raises(ValueError, match="NaN or infinite"):
        validate_candidates(_plain([(1, 5, 1, float("nan"))]), PLAIN)


def test_wrong_channel_value_raises() -> None:
    with pytest.raises(ValueError, match="channel column must be 'rep'"):
        validate_candidates(_plain([(1, 5, 1, 0.9)], channel="sib"), PLAIN)


def test_duplicate_article_in_query_raises() -> None:
    with pytest.raises(ValueError, match="repeats within a query"):
        validate_candidates(_plain([(1, 5, 1, 0.9), (1, 5, 2, 0.5)]), PLAIN)


def test_more_rows_than_quota_raises() -> None:
    with pytest.raises(ValueError, match="quota is 2"):
        validate_candidates(VALID, PLAIN, quota=2)


# ---------------------------------------------------------------- order errors


def test_rank_gap_raises() -> None:
    with pytest.raises(ValueError, match="rank must be 1..n"):
        validate_candidates(_plain([(1, 5, 1, 0.9), (1, 3, 3, 0.5)]), PLAIN)


def test_rank_not_starting_at_one_raises() -> None:
    with pytest.raises(ValueError, match="rank must be 1..n"):
        validate_candidates(_plain([(1, 5, 2, 0.9), (1, 3, 3, 0.5)]), PLAIN)


def test_rank_against_score_order_raises() -> None:
    with pytest.raises(ValueError, match="score DESC, article_idx ASC"):
        validate_candidates(_plain([(1, 5, 1, 0.1), (1, 3, 2, 0.9)]), PLAIN)


def test_tie_not_broken_by_article_ascending_raises() -> None:
    with pytest.raises(ValueError, match="score DESC, article_idx ASC"):
        validate_candidates(_plain([(1, 5, 1, 0.5), (1, 3, 2, 0.5)]), PLAIN)


# ---------------------------------------------------------------- writing


def test_write_validates_and_refuses_to_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "candidates" / "D1" / "rep.parquet"
    write_candidates(VALID, PLAIN, quota=3, path=path, overwrite=False)
    assert pl.read_parquet(path).equals(VALID)
    with pytest.raises(FileExistsError):
        write_candidates(VALID, PLAIN, quota=3, path=path, overwrite=False)
    bad_path = tmp_path / "bad.parquet"
    with pytest.raises(ValueError):
        write_candidates(VALID, PLAIN, quota=2, path=bad_path, overwrite=False)
    assert not bad_path.exists()
