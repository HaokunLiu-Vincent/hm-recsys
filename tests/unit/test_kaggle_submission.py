"""Tests for kaggle.submission: the submission file, its format checks, candidate statistics.

Tiny example with k = 2 predictions per customer:
    queries   q10 -> customer 0 ("cA"), q11 -> customer 1 ("cB"), q12 -> customer 2 ("cC")
    Top-2     q10: articles 3, 1    q11: 2, 3    q12: 1, 2
    article ids: idx 1 -> 0000000011, 2 -> 0000000022, 3 -> 0000000033
    the sample file lists the customers as cC, cA, cB
"""

import gzip
from pathlib import Path

import polars as pl
import pytest

from hmrec.kaggle.submission import (
    FileCheck,
    candidate_statistics,
    check_complete_top,
    check_submission_file,
    order_like_sample,
    predictions_table,
    write_submission,
)
from hmrec.locks import file_sha256

K = 2
QUERIES = pl.DataFrame(
    {"query_id": [10, 11, 12], "customer_idx": [0, 1, 2]},
    schema={"query_id": pl.Int32, "customer_idx": pl.Int32},
)
CUSTOMER_IDS = pl.DataFrame(
    {"customer_idx": [0, 1, 2], "customer_id": ["cA", "cB", "cC"]},
    schema={"customer_idx": pl.Int32, "customer_id": pl.String},
)
ARTICLE_IDS = pl.DataFrame(
    {"article_idx": [1, 2, 3], "article_id": ["0000000011", "0000000022", "0000000033"]},
    schema={"article_idx": pl.Int32, "article_id": pl.String},
)
SAMPLE = pl.Series("customer_id", ["cC", "cA", "cB"])
KNOWN = ARTICLE_IDS["article_id"]


def _top(rows: list[tuple[int, int, int]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows, schema={"query_id": pl.Int32, "rank": pl.Int32, "article_idx": pl.Int32}, orient="row"
    )


# Deliberately not in (query, rank) order: the file must follow the ranks, not the row order.
TOP = _top([(11, 2, 3), (10, 1, 3), (12, 1, 1), (10, 2, 1), (11, 1, 2), (12, 2, 2)])


def _table() -> pl.DataFrame:
    return order_like_sample(predictions_table(TOP, QUERIES, CUSTOMER_IDS, ARTICLE_IDS, K), SAMPLE)


def _write(tmp_path: Path, lines: list[str] | None = None) -> tuple[Path, Path]:
    """Write the example file, or the given raw lines with a matching gzip copy."""
    csv_file, gzip_file = tmp_path / "submission.csv", tmp_path / "submission.csv.gz"
    if lines is None:
        write_submission(_table(), csv_file, gzip_file)
    else:
        csv_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        gzip_file.write_bytes(gzip.compress(csv_file.read_bytes()))
    return csv_file, gzip_file


def _check(tmp_path: Path, lines: list[str] | None = None) -> FileCheck:
    return check_submission_file(*_write(tmp_path, lines), SAMPLE, KNOWN, K)


def test_rows_follow_the_sample_order_and_ids_follow_the_ranks(tmp_path: Path) -> None:
    csv_file, _ = _write(tmp_path)
    assert csv_file.read_text(encoding="utf-8").splitlines() == [
        "customer_id,prediction",
        "cC,0000000011 0000000022",
        "cA,0000000033 0000000011",
        "cB,0000000022 0000000033",
    ]


def test_written_file_passes_every_check_and_the_gzip_copy_is_reproducible(tmp_path: Path) -> None:
    csv_file, gzip_file = _write(tmp_path / "a")
    check = check_submission_file(csv_file, gzip_file, SAMPLE, KNOWN, K)
    assert check.passed and check.failed == [] and len(check.checks) == 9
    assert (check.n_rows, check.csv_bytes) == (3, csv_file.stat().st_size)
    assert check.csv_sha256 == file_sha256(csv_file)
    assert gzip.decompress(gzip_file.read_bytes()) == csv_file.read_bytes()
    # No file name and no time in the gzip header: the same CSV gives the same bytes.
    _, again = _write(tmp_path / "b")
    assert file_sha256(again) == check.gzip_sha256
    with pytest.raises(FileExistsError, match="never replaced"):
        write_submission(_table(), csv_file, gzip_file)


@pytest.mark.parametrize(
    ("lines", "failed"),
    [
        # another header
        (["customer,prediction", "cC,0000000011 0000000022"], "header is customer_id,prediction"),
        # a customer is missing
        (
            ["customer_id,prediction", "cC,0000000011 0000000022", "cA,0000000033 0000000011"],
            "same number of rows as the sample",
        ),
        # the right customers in another order
        (
            [
                "customer_id,prediction",
                "cA,0000000033 0000000011",
                "cC,0000000011 0000000022",
                "cB,0000000022 0000000033",
            ],
            "same customers in the same row order",
        ),
        # only one id in a row
        (
            [
                "customer_id,prediction",
                "cC,0000000011",
                "cA,0000000033 0000000011",
                "cB,0000000022 0000000033",
            ],
            "exactly 2 ids per row",
        ),
        # the same id twice
        (
            [
                "customer_id,prediction",
                "cC,0000000011 0000000011",
                "cA,0000000033 0000000011",
                "cB,0000000022 0000000033",
            ],
            "2 distinct ids per row",
        ),
        # an id that lost its leading zero
        (
            [
                "customer_id,prediction",
                "cC,11 0000000022",
                "cA,0000000033 0000000011",
                "cB,0000000022 0000000033",
            ],
            "every id is 10 digits with a leading zero",
        ),
        # a well-formed id that is not an article
        (
            [
                "customer_id,prediction",
                "cC,0000000099 0000000022",
                "cA,0000000033 0000000011",
                "cB,0000000022 0000000033",
            ],
            "every id is an article of the catalog",
        ),
    ],
)
def test_each_broken_rule_is_caught(tmp_path: Path, lines: list[str], failed: str) -> None:
    check = _check(tmp_path, lines)
    assert not check.passed and failed in check.failed


def test_a_gzip_copy_of_another_file_is_caught(tmp_path: Path) -> None:
    csv_file, gzip_file = _write(tmp_path)
    gzip_file.write_bytes(gzip.compress(csv_file.read_bytes() + b"cD,0000000011 0000000022\n"))
    check = check_submission_file(csv_file, gzip_file, SAMPLE, KNOWN, K)
    assert check.failed == ["the gzip copy unpacks to the same bytes"]


def test_top_k_must_be_complete_for_every_query() -> None:
    check_complete_top(TOP, QUERIES, K)
    without_q12 = TOP.filter(pl.col("query_id") != 12)
    with pytest.raises(ValueError, match="holds 2 queries, the run has 3"):
        predictions_table(without_q12, QUERIES, CUSTOMER_IDS, ARTICLE_IDS, K)
    one_row_short = TOP.filter(~((pl.col("query_id") == 12) & (pl.col("rank") == 2)))
    with pytest.raises(ValueError, match=r"1 queries do not have exactly 2 distinct.*\[12\]"):
        check_complete_top(one_row_short, QUERIES, K)
    repeated_article = _top(
        [(10, 1, 3), (10, 2, 3), (11, 1, 2), (11, 2, 3), (12, 1, 1), (12, 2, 2)]
    )
    with pytest.raises(ValueError, match=r"\[10\]"):
        check_complete_top(repeated_article, QUERIES, K)


def test_unknown_ids_and_other_customers_are_refused() -> None:
    with pytest.raises(ValueError, match="not in the article id map"):
        predictions_table(TOP, QUERIES, CUSTOMER_IDS, ARTICLE_IDS.head(2), K)
    with pytest.raises(ValueError, match="not in the customer id map"):
        predictions_table(TOP, QUERIES, CUSTOMER_IDS.head(2), ARTICLE_IDS, K)
    predictions = predictions_table(TOP, QUERIES, CUSTOMER_IDS, ARTICLE_IDS, K)
    with pytest.raises(ValueError, match="1 sample customers have no prediction"):
        order_like_sample(predictions, pl.Series("customer_id", ["cC", "cA", "cX"]))
    with pytest.raises(ValueError, match="the run predicts 3 customers, the sample holds 2"):
        order_like_sample(predictions, pl.Series("customer_id", ["cC", "cA"]))
    with pytest.raises(ValueError, match="repeat"):
        order_like_sample(predictions, pl.Series("customer_id", ["cC", "cC", "cA"]))


def test_candidate_statistics_hand_computed() -> None:
    stats = pl.DataFrame(
        {"n_candidates": [100, 200, 300, 400], "has_history": [True, False, True, False]},
        schema={"n_candidates": pl.Int32, "has_history": pl.Boolean},
    )
    result = candidate_statistics(stats)
    # mean (100 + 200 + 300 + 400) / 4 = 250; median 250; P90 = 300 + 0.7 x 100 = 370.
    assert (result.n_customers, result.n_candidate_rows) == (4, 1000)
    assert result.per_customer_mean == pytest.approx(250.0)
    assert result.per_customer_p50 == pytest.approx(250.0)
    assert result.per_customer_p90 == pytest.approx(370.0)
    assert result.no_history_share == pytest.approx(0.5)
    with pytest.raises(ValueError, match="no query statistics"):
        candidate_statistics(stats.head(0))
