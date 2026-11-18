"""Tests for hmrec.eval.ranker_eval: the V-fold label guard and the metrics of a Top-K table.

Hand example (two queries, Top-K tables with 2 and 3 rows):
    q1  G = {7}       ranked 5, 7       hit at rank 2
        AP@12 = (1/2) / min(12, 1) = 0.5        Recall@12 = 1/1 = 1
        NDCG@12 = (1 / log2(3)) / 1 = 0.630930
    q2  G = {1, 2}    ranked 1, 8, 9    hit at rank 1
        AP@12 = (1/1) / min(12, 2) = 0.5        Recall@12 = 1/2 = 0.5
        NDCG@12 = 1 / (1 + 1 / log2(3)) = 0.613147
    q3  G = {4}       no prediction at all: every metric is 0, and q3 stays in the mean
Fold means: MAP 1/3, Recall 0.5, NDCG (0.630930 + 0.613147) / 3 = 0.414692.
"""

import math
from pathlib import Path

import polars as pl
import pytest
from ranker_example import MANIFEST, write_labels

from hmrec.eval.ranker_eval import load_m7_evaluator, ranking_scores

METRICS = ("map@12", "recall@12", "ndcg@12")


@pytest.mark.parametrize("fold", ["T1", "T4", "F1", "D1"])
def test_only_v_labels_are_read(fold: str, tmp_path: Path) -> None:
    # The file does not exist: a ValueError (not FileNotFoundError) proves it is not opened.
    with pytest.raises(ValueError, match="M7 evaluates only V folds"):
        load_m7_evaluator(MANIFEST.get(fold), tmp_path / "missing.parquet")


def test_hand_example(tmp_path: Path) -> None:
    labels_file = tmp_path / "labels" / "V1.parquet"
    write_labels(labels_file, "V1", [(1, 7), (2, 1), (2, 2), (3, 4)])
    evaluator = load_m7_evaluator(MANIFEST.get("V1"), labels_file)
    ranked = pl.DataFrame(
        {
            "query_id": [1, 1, 2, 2, 2],
            "article_idx": [5, 7, 1, 8, 9],
            "rank": [1, 2, 1, 2, 3],
        },
        schema={"query_id": pl.Int32, "article_idx": pl.Int32, "rank": pl.Int32},
    )
    scores = ranking_scores(evaluator, ranked, METRICS)
    assert scores.n_queries == 3
    ndcg_q1 = 1 / math.log2(3)
    ndcg_q2 = 1 / (1 + 1 / math.log2(3))
    assert scores.means["map@12"] == pytest.approx(1 / 3)
    assert scores.means["recall@12"] == pytest.approx(0.5)
    assert scores.means["ndcg@12"] == pytest.approx((ndcg_q1 + ndcg_q2) / 3)
    assert scores.per_query.columns == ["query_id", "fold", *METRICS]
    assert scores.per_query["query_id"].to_list() == [1, 2, 3]
    assert scores.per_query["map@12"].to_list() == [0.5, 0.5, 0.0]
    assert scores.per_query["recall@12"].to_list() == [1.0, 0.5, 0.0]
