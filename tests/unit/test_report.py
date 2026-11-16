"""Tests for eval.report: metric names, FoldEvaluator, multi-fold summaries.

Labels (fold D1): q1 -> {10, 11}, q2 -> {12}. Candidates: q1 ranks 10 (1), 20 (2);
q2 has nothing. Recall@2: q1 = 1/2, q2 = 0 -> fold mean 0.25.

Summary example: D1 queries 0.5, 0.0 (mean 0.25); D2 queries 1.0, 1.0, 0.0 (mean 2/3).
    pooled     = (0.5 + 0 + 1 + 1 + 0) / 5 = 0.5
    equal-fold = (0.25 + 2/3) / 2 = 0.458333...
"""

import polars as pl
import pytest

from hmrec.eval.report import FoldEvaluator, parse_metric, summarize_folds

LABELS = pl.DataFrame(
    {"query_id": [1, 1, 2], "fold": ["D1"] * 3, "article_idx": [10, 11, 12]},
    schema={"query_id": pl.Int32, "fold": pl.String, "article_idx": pl.Int32},
)
CANDIDATES = pl.DataFrame(
    {"query_id": [1, 1], "article_idx": [10, 20], "rank": [1, 2]},
    schema={"query_id": pl.Int32, "article_idx": pl.Int32, "rank": pl.Int32},
)


def test_parse_metric() -> None:
    assert parse_metric("recall@100") == ("recall", 100)
    assert parse_metric("map@12") == ("map", 12)
    for bad in ("recall", "precision@5", "ndcg@0", "recall@x"):
        with pytest.raises(ValueError, match="metric must look like"):
            parse_metric(bad)


def test_fold_evaluator_recall_and_hits() -> None:
    evaluator = FoldEvaluator("D1", LABELS)
    per_query = evaluator.per_query(CANDIDATES, "recall@2")
    assert per_query["value"].to_list() == [0.5, 0.0]
    assert evaluator.n_queries == 2
    marked = evaluator.mark_hits(CANDIDATES).sort("article_idx")
    assert marked["is_hit"].to_list() == [True, False]


def test_from_file_checks_fold(tmp_path) -> None:
    path = tmp_path / "D1.parquet"
    LABELS.write_parquet(path)
    assert FoldEvaluator.from_file("D1", path).n_queries == 2
    with pytest.raises(ValueError, match="expected \\['D2'\\]"):
        FoldEvaluator.from_file("D2", path)


def test_summarize_folds_hand_computed() -> None:
    per_query = pl.DataFrame(
        {
            "query_id": [1, 2, 10, 11, 12],
            "fold": ["D1", "D1", "D2", "D2", "D2"],
            "value": [0.5, 0.0, 1.0, 1.0, 0.0],
        },
        schema={"query_id": pl.Int32, "fold": pl.String, "value": pl.Float64},
    )
    summary = summarize_folds(per_query, {"only_D2": ("D2",), "both": ("D1", "D2")})
    assert summary.fold_scores == {"D1": 0.25, "D2": pytest.approx(2 / 3)}
    assert summary.fold_queries == {"D1": 2, "D2": 3}
    assert summary.pooled == 0.5
    assert summary.equal_fold == pytest.approx((0.25 + 2 / 3) / 2)
    assert summary.group_scores == {"only_D2": pytest.approx(2 / 3), "both": 0.5}
