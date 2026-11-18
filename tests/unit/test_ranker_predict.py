"""Tests for hmrec.ranker.predict: staged scores, Top-K and the tie-break."""

import numpy as np
import polars as pl
import pytest
from ranker_example import LEARNABLE_FOLDS, build_workspace

from hmrec.hpo.v_trial import (
    TrialInputs,
    build_and_save_dataset,
    load_planned_dataset,
    plan_dataset,
)
from hmrec.ranker.predict import (
    checkpoints,
    feature_matrix,
    score_rows,
    staged_raw_scores,
    top_k_per_query,
)
from hmrec.ranker.train import train_ranker, training_params


@pytest.fixture(scope="module")
def inputs(tmp_path_factory: pytest.TempPathFactory) -> TrialInputs:
    workspace = build_workspace(tmp_path_factory.mktemp("predict"))
    plan = plan_dataset(workspace, "full", "V1", 2, "test-commit")
    build_and_save_dataset(plan, workspace.ranker.sampling.row_group_rows)
    return workspace


def _model_and_rows(inputs: TrialInputs):  # noqa: ANN202 (a pair of library objects)
    params = training_params(inputs.ranker, inputs.ranker.tuned_defaults, "col_wise", 2)
    dataset, _ = load_planned_dataset(plan_dataset(inputs, "full", "V1", 2, "c"), params)
    booster, _ = train_ranker(dataset, params, 20, 10)
    rows = pl.read_parquet(inputs.paths.features_dir("V1") / "part-00000.parquet")
    return booster, rows


def test_checkpoints() -> None:
    assert checkpoints(2000, 100) == tuple(range(100, 2001, 100))
    assert checkpoints(300, 100) == (100, 200, 300)
    for num_trees, every in ((250, 100), (0, 100), (100, 0)):
        with pytest.raises(ValueError, match="positive multiple"):
            checkpoints(num_trees, every)


def test_feature_matrix_is_float32_in_model_order() -> None:
    rows = pl.DataFrame(
        {"b": [1.5, float("nan")], "code": [3, 0], "a": [2.0, 4.0]},
        schema={"b": pl.Float32, "code": pl.Int16, "a": pl.Float32},
    )
    matrix = feature_matrix(rows, ["a", "code", "b"])
    assert matrix.dtype == np.float32 and matrix.flags["C_CONTIGUOUS"]
    assert matrix[0].tolist() == [2.0, 3.0, 1.5]
    assert np.isnan(matrix[1, 2]) and matrix[1, 1] == 0.0


def test_staged_scores_agree_with_a_direct_prediction(inputs: TrialInputs) -> None:
    booster, rows = _model_and_rows(inputs)
    features = feature_matrix(rows, booster.feature_name())
    staged = dict(staged_raw_scores(booster, features, 20, 10, 2))
    assert sorted(staged) == [10, 20]
    for k, scores in staged.items():
        direct = np.asarray(booster.predict(features, num_iteration=k, raw_score=True))
        assert scores.dtype == np.float64 and scores.shape == direct.shape
        # Not bit-identical in general (another order of additions), but only by rounding.
        assert np.abs(scores - direct).max() < 1e-9
        staged_top = top_k_per_query(
            rows["query_id"].to_numpy(), rows["article_idx"].to_numpy(), scores, 12
        )
        direct_top = top_k_per_query(
            rows["query_id"].to_numpy(), rows["article_idx"].to_numpy(), direct, 12
        )
        assert staged_top.select("query_id", "article_idx", "rank").equals(
            direct_top.select("query_id", "article_idx", "rank")
        )
    # Every yielded array is a new one: a later checkpoint does not change an earlier one.
    assert not np.array_equal(staged[10], staged[20])


def test_staged_scores_check_their_inputs(inputs: TrialInputs) -> None:
    booster, rows = _model_and_rows(inputs)
    features = feature_matrix(rows, booster.feature_name())
    with pytest.raises(ValueError, match="the model has 20 trees"):
        list(staged_raw_scores(booster, features, 30, 10, 2))
    with pytest.raises(ValueError, match="features must be"):
        list(staged_raw_scores(booster, features[:, :5], 20, 10, 2))


def test_top_k_order_and_tie_break() -> None:
    query_ids = np.array([1, 1, 1, 1, 2, 2], dtype=np.int32)
    articles = np.array([40, 10, 30, 20, 7, 5], dtype=np.int32)
    scores = np.array([0.5, 0.9, 0.5, 0.5, 0.1, 0.1])
    top = top_k_per_query(query_ids, articles, scores, 3)
    # q1: 0.9 first; then the three 0.5 ties by article_idx ASC (20, 30, 40), cut at 3.
    # q2: both 0.1 -> article 5 before 7.
    assert top["query_id"].to_list() == [1, 1, 1, 2, 2]
    assert top["article_idx"].to_list() == [10, 20, 30, 5, 7]
    assert top["rank"].to_list() == [1, 2, 3, 1, 2]
    assert top["score"].to_list() == [0.9, 0.5, 0.5, 0.1, 0.1]
    assert top.schema == {
        "query_id": pl.Int32,
        "article_idx": pl.Int32,
        "score": pl.Float64,
        "rank": pl.Int32,
    }


def test_top_k_refuses_nan_and_bad_k() -> None:
    ids = np.array([1], dtype=np.int32)
    with pytest.raises(ValueError, match="NaN"):
        top_k_per_query(ids, ids, np.array([float("nan")]), 12)
    with pytest.raises(ValueError, match="top_k must be > 0"):
        top_k_per_query(ids, ids, np.array([1.0]), 0)


def test_score_rows_ranks_every_query_at_every_checkpoint(inputs: TrialInputs) -> None:
    booster, rows = _model_and_rows(inputs)
    tops = score_rows(booster, rows, booster.feature_name(), 20, 10, 5, 2)
    assert sorted(tops) == [10, 20]
    n_queries = rows["query_id"].n_unique()
    assert n_queries == 30 and rows["query_id"].min() == LEARNABLE_FOLDS["V1"]
    for top in tops.values():
        assert top.height == 5 * n_queries  # 12 candidates per query, Top-5 kept
        assert top.group_by("query_id").agg(pl.col("rank").max())["rank"].unique().to_list() == [5]
