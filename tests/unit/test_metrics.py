"""Tests for hmrec.eval.metrics with hand-computed examples.

The main example has 4 queries in fold D1:

    query 1: G = {10, 20, 30}  (|G| = 3)
             ranked list: 1:10 (hit), 2:99, 3:20 (hit), 4:98, 5:30 (hit)
    query 2: G = {40}          (single label)
             ranked list: 1:41, 2:40 (hit)      -> only 2 items: short output
    query 3: G = {50, 60}
             ranked list: 1:51, 2:52            -> no hit at all
    query 4: G = {70}
             no predictions at all              -> still counted, scores 0

log2 values used below: log2(2) = 1, log2(3) = 1.5849625, log2(4) = 2.
"""

import math
import random

import polars as pl
import pytest

from hmrec.eval.metrics import (
    average_precision_at_k,
    equal_fold_mean,
    macro_mean,
    ndcg_at_k,
    per_fold_means,
    query_weighted_mean,
    recall_at_k,
    union_recall,
)

REL = 1e-12  # tolerance for comparing with a fraction written in a different order


def _labels(rows: list[tuple[int, str, int]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={"query_id": pl.Int32, "fold": pl.String, "article_idx": pl.Int32},
        orient="row",
    )


def _predictions(rows: list[tuple[int, int, int]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={"query_id": pl.Int32, "article_idx": pl.Int32, "rank": pl.Int32},
        orient="row",
    )


LABELS = _labels(
    [
        (1, "D1", 10),
        (1, "D1", 20),
        (1, "D1", 30),
        (2, "D1", 40),
        (3, "D1", 50),
        (3, "D1", 60),
        (4, "D1", 70),
    ]
)
PREDICTIONS = _predictions(
    [
        (1, 10, 1),
        (1, 99, 2),
        (1, 20, 3),
        (1, 98, 4),
        (1, 30, 5),
        (2, 41, 1),
        (2, 40, 2),
        (3, 51, 1),
        (3, 52, 2),
    ]
)


def _values(per_query: pl.DataFrame) -> dict[int, float]:
    return dict(zip(per_query["query_id"].to_list(), per_query["value"].to_list(), strict=True))


# ---------------------------------------------------------------- Recall@K


def test_recall_at_3_hand_computed() -> None:
    values = _values(recall_at_k(PREDICTIONS, LABELS, k=3))
    # q1: top 3 = {10, 99, 20}; hits {10, 20} -> 2 / |G| = 2/3.
    assert values[1] == pytest.approx(2 / 3, rel=REL)
    # q2: only 2 items (short output, not padded); 40 at rank 2 is a hit -> 1/1.
    assert values[2] == 1.0
    # q3: no hit; q4: no predictions -> 0, but both stay in the average.
    assert values[3] == 0.0
    assert values[4] == 0.0
    # Macro mean = (2/3 + 1 + 0 + 0) / 4 = 5/12.
    assert macro_mean(recall_at_k(PREDICTIONS, LABELS, k=3)) == pytest.approx(5 / 12, rel=REL)


def test_recall_cut_off_and_single_label() -> None:
    values = _values(recall_at_k(PREDICTIONS, LABELS, k=1))
    # k = 1: q1 rank 1 is 10 -> 1/3; q2 rank 1 is 41, the single label 40 is at rank 2 -> 0.
    assert values[1] == pytest.approx(1 / 3, rel=REL)
    assert values[2] == 0.0
    # k = 5: q1 gets all three hits -> 3/3.
    assert _values(recall_at_k(PREDICTIONS, LABELS, k=5))[1] == 1.0


# ---------------------------------------------------------------- AP@12


def test_ap_at_12_hand_computed() -> None:
    values = _values(average_precision_at_k(PREDICTIONS, LABELS, k=12))
    # q1: hits at ranks 1, 3, 5.
    #     P(1) = 1/1, P(3) = 2/3, P(5) = 3/5.
    #     AP = (1 + 2/3 + 3/5) / min(12, 3) = (15/15 + 10/15 + 9/15) / 3 = 34/45 = 0.75556
    assert values[1] == pytest.approx(34 / 45, rel=REL)
    # q2: single label at rank 2 -> P(2) = 1/2; denominator min(12, 1) = 1 -> 0.5.
    assert values[2] == 0.5
    # q3: no hit -> 0. q4: no predictions -> 0.
    assert values[3] == 0.0
    assert values[4] == 0.0
    # MAP@12 = (34/45 + 1/2 + 0 + 0) / 4 = (68/90 + 45/90) / 4 = 113/360 = 0.31389
    map_at_12 = macro_mean(average_precision_at_k(PREDICTIONS, LABELS, k=12))
    assert map_at_12 == pytest.approx(113 / 360, rel=REL)


def test_ap_denominator_is_min_of_12_and_label_count() -> None:
    # 13 labels, the 12 predictions are all hits. The denominator is min(12, 13) = 12,
    # so AP = (1/1 + 2/2 + ... + 12/12) / 12 = 1.0, although recall@12 = 12/13.
    labels = _labels([(7, "D1", article) for article in range(13)])
    predictions = _predictions([(7, article, article + 1) for article in range(12)])
    assert _values(average_precision_at_k(predictions, labels, k=12))[7] == 1.0
    assert _values(recall_at_k(predictions, labels, k=12))[7] == pytest.approx(12 / 13, rel=REL)


def test_ap_ignores_hits_below_cut_off() -> None:
    # q1 with k = 3: only hits at ranks 1 and 3 count -> (1 + 2/3) / min(3, 3) = 5/9.
    values = _values(average_precision_at_k(PREDICTIONS, LABELS, k=3))
    assert values[1] == pytest.approx(5 / 9, rel=REL)


# ---------------------------------------------------------------- NDCG@K


def test_ndcg_at_3_hand_computed() -> None:
    values = _values(ndcg_at_k(PREDICTIONS, LABELS, k=3))
    # q1: hits at ranks 1 and 3 (rank 5 is cut off).
    #     DCG  = 1/log2(2) + 1/log2(4) = 1 + 0.5 = 1.5
    #     IDCG = min(3, 3) = 3 perfect hits = 1 + 1/log2(3) + 1/log2(4) = 2.1309298
    #     NDCG = 1.5 / 2.1309298 = 0.7039180
    ideal_q1 = 1 + 1 / math.log2(3) + 0.5
    assert values[1] == pytest.approx(1.5 / ideal_q1, rel=REL)
    assert values[1] == pytest.approx(0.7039180, abs=1e-7)
    # q2: single label at rank 2. DCG = 1/log2(3) = 0.6309298; IDCG = 1/log2(2) = 1.
    assert values[2] == pytest.approx(1 / math.log2(3), rel=REL)
    assert values[3] == 0.0
    assert values[4] == 0.0


def test_ndcg_perfect_ranking_is_exactly_one() -> None:
    labels = _labels([(5, "D1", 1), (5, "D1", 2), (5, "D1", 3)])
    predictions = _predictions([(5, 3, 1), (5, 1, 2), (5, 2, 3), (5, 9, 4)])
    assert _values(ndcg_at_k(predictions, labels, k=3))[5] == 1.0
    assert _values(ndcg_at_k(predictions, labels, k=12))[5] == 1.0


# ---------------------------------------------------------------- union recall


def test_union_recall_counts_each_candidate_once() -> None:
    # Candidates from two channels; article 10 appears twice for q1 but counts once.
    candidates = pl.DataFrame(
        {"query_id": [1, 1, 1, 1, 2], "article_idx": [10, 10, 20, 77, 41]},
        schema={"query_id": pl.Int32, "article_idx": pl.Int32},
    )
    values = _values(union_recall(candidates, LABELS))
    assert values[1] == pytest.approx(2 / 3, rel=REL)  # {10, 20} of {10, 20, 30}
    assert values[2] == 0.0  # 41 is not 40
    assert values[3] == 0.0  # no candidates for q3
    assert values[4] == 0.0  # nor for q4


# ---------------------------------------------------------------- aggregation


def test_query_weighted_vs_equal_fold() -> None:
    # D1: 3 queries with values 0, 0, 1 -> fold mean 1/3.
    # D2: 1 query with value 1           -> fold mean 1.
    # query-weighted: (0 + 0 + 1 + 1) / 4 = 0.5
    # equal-fold:     (1/3 + 1) / 2      = 2/3
    per_query = pl.DataFrame(
        {
            "query_id": [0, 1, 2, 1000],
            "fold": ["D1", "D1", "D1", "D2"],
            "value": [0.0, 0.0, 1.0, 1.0],
        }
    )
    assert query_weighted_mean(per_query) == 0.5
    assert equal_fold_mean(per_query) == pytest.approx(2 / 3, rel=REL)
    fold_means = per_fold_means(per_query)
    assert fold_means["fold"].to_list() == ["D1", "D2"]
    assert fold_means["n_queries"].to_list() == [3, 1]


# ---------------------------------------------------------------- determinism


def _random_example(seed: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    rng = random.Random(seed)
    label_rows = []
    prediction_rows = []
    for query_id in range(300):
        fold = "D1" if query_id < 150 else "D2"
        labels = rng.sample(range(40), rng.randint(1, 15))
        label_rows += [(query_id, fold, article) for article in labels]
        ranked = rng.sample(range(40), rng.randint(0, 20))
        prediction_rows += [(query_id, article, rank + 1) for rank, article in enumerate(ranked)]
    return _predictions(prediction_rows), _labels(label_rows)


def test_metrics_are_identical_after_shuffling_rows() -> None:
    predictions, labels = _random_example(seed=7)
    shuffled_predictions = predictions.sample(fraction=1.0, shuffle=True, seed=1)
    shuffled_labels = labels.sample(fraction=1.0, shuffle=True, seed=2)
    for metric in (recall_at_k, average_precision_at_k, ndcg_at_k):
        original = metric(predictions, labels, 12)
        shuffled = metric(shuffled_predictions, shuffled_labels, 12)
        assert original.equals(shuffled)
        # Exact equality on purpose, not approx: results must be bit-identical.
        assert macro_mean(original) == macro_mean(shuffled)
        assert equal_fold_mean(original) == equal_fold_mean(shuffled.sample(fraction=1.0, seed=3))


# ---------------------------------------------------------------- invalid input


def test_prediction_for_query_without_labels_raises() -> None:
    # Query 9 has an empty (or unknown) label set: that is a bug, not a 0.
    predictions = pl.concat([PREDICTIONS, _predictions([(9, 1, 1)])])
    with pytest.raises(ValueError, match="without labels"):
        recall_at_k(predictions, LABELS, k=12)


def test_empty_labels_raise() -> None:
    with pytest.raises(ValueError, match="labels is empty"):
        recall_at_k(PREDICTIONS.head(0), LABELS.head(0), k=12)


def test_bad_predictions_raise() -> None:
    duplicate_article = _predictions([(1, 10, 1), (1, 10, 2)])
    with pytest.raises(ValueError, match="unique within each query"):
        recall_at_k(duplicate_article, LABELS, k=12)
    rank_gap = _predictions([(1, 10, 1), (1, 20, 3)])
    with pytest.raises(ValueError, match="ranks must be 1..n"):
        recall_at_k(rank_gap, LABELS, k=12)
    with pytest.raises(ValueError, match="positive integer"):
        recall_at_k(PREDICTIONS, LABELS, k=0)
