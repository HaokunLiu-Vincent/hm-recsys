"""Tests for hmrec.hpo.v_trial: one trial end to end on tiny data, and the RRF baseline.

RRF baseline by hand (order: rrf_score DESC, best_rank ASC, article_idx ASC):
    q1 candidates   article  rrf_score  best_rank
                       1       0.02        5
                       2       0.02        5       same score and best_rank as 1 -> after 1
                       3       0.03        9       highest score -> first
                       4       0.02        3       same score, better best_rank -> before 1
    ranking: 3, 4, 1, 2. Label {2} is at rank 4:
        AP@12 = (1/4) / 1 = 0.25    Recall@12 = 1    NDCG@12 = 1 / log2(5)
    q2 has one candidate (article 9) and label {8}: every metric is 0.
    Fold means: MAP 0.125, Recall 0.5, NDCG 1 / (2 * log2(5)).
"""

import importlib.util
import math
from dataclasses import replace
from pathlib import Path

import polars as pl
import pytest
from ranker_example import (
    CANDIDATES_PER_QUERY,
    METRICS,
    REPO,
    build_workspace,
    tiny_inputs,
    tiny_ranker_config,
    wide_table,
    write_features,
    write_labels,
)

from hmrec.hpo.v_phase import sample_job
from hmrec.hpo.v_records import AREA_SPEED_TEST, AREA_V_PHASE
from hmrec.hpo.v_trial import (
    TrialInputs,
    TrialSpec,
    build_and_save_dataset,
    check_trial_settings,
    model_config,
    plan_dataset,
    ranker_config_id,
    rrf_baseline_row,
    run_trial,
)
from hmrec.ranker.sampling import sample_fold
from hmrec.ranker.train import load_model

N_QUERIES = 60


@pytest.fixture(scope="module")
def inputs(tmp_path_factory: pytest.TempPathFactory) -> TrialInputs:
    workspace = build_workspace(tmp_path_factory.mktemp("trial"))
    build_and_save_dataset(plan_dataset(workspace, "full", "V1", 2, "c"), 64)
    return workspace


def _spec(inputs: TrialInputs, **changes: object) -> TrialSpec:
    spec = TrialSpec(
        area=AREA_V_PHASE,
        experiment="full",
        step="leaves_x_min_data",
        target_fold="V1",
        tuned=inputs.ranker.tuned_defaults,
        num_trees=20,
        histogram_mode="col_wise",
        evaluate=True,
    )
    return replace(spec, **changes)  # pyright: ignore[reportArgumentType]  test helper


# ---------------------------------------------------------------- identity and settings


def test_config_id_names_the_model_not_the_number_of_trees() -> None:
    ranker = tiny_ranker_config()
    tuned = ranker.tuned_defaults
    config = model_config(ranker, tuned, "full", "col_wise")
    config_id = ranker_config_id(config)
    assert len(config_id) == 12 and config_id == ranker_config_id(dict(config))
    assert "num_trees" not in config  # the first k trees do not depend on the total
    others = [
        model_config(ranker, replace(tuned, num_leaves=8), "full", "col_wise"),
        model_config(ranker, replace(tuned, negatives_per_positive=4), "full", "col_wise"),
        model_config(ranker, tuned, "minus_r6", "col_wise"),
        model_config(ranker, tuned, "full", "row_wise"),
    ]
    assert len({config_id, *(ranker_config_id(other) for other in others)}) == 5


def test_grid_trials_must_use_the_decided_histogram_mode(inputs: TrialInputs) -> None:
    check_trial_settings(inputs.ranker, _spec(inputs))
    with pytest.raises(ValueError, match="must use histogram.mode"):
        check_trial_settings(inputs.ranker, _spec(inputs, histogram_mode="row_wise"))
    # The speed test may choose a mode itself.
    check_trial_settings(
        inputs.ranker, _spec(inputs, area=AREA_SPEED_TEST, histogram_mode="row_wise")
    )
    undecided = tiny_ranker_config(
        histogram={
            "mode": "undecided",
            "comparison_trees": 6,
            "warmup_trees": 2,
            "min_row_wise_speedup": 0.15,
        }
    )
    with pytest.raises(ValueError, match="'undecided'"):
        check_trial_settings(undecided, _spec(inputs))
    with pytest.raises(ValueError, match="multiple of 10"):
        check_trial_settings(inputs.ranker, _spec(inputs, num_trees=15))
    with pytest.raises(ValueError, match="num_trees must be in 1..20"):
        check_trial_settings(inputs.ranker, _spec(inputs, num_trees=30))
    # A timing-only run has no checkpoints: the speed test trains 50 trees (6 here).
    timing = _spec(inputs, area=AREA_SPEED_TEST, num_trees=6, evaluate=False)
    check_trial_settings(inputs.ranker, timing)


# ---------------------------------------------------------------- T folds


@pytest.mark.parametrize("fold", ["T1", "T2", "T4", "F4"])
def test_run_trial_refuses_every_fold_that_is_not_a_v_fold(fold: str, tmp_path: Path) -> None:
    # An empty folder: nothing can be read, so the error must come from the fold check.
    empty = tiny_inputs(tmp_path)
    with pytest.raises(ValueError, match="M7 predicts and evaluates only V folds"):
        run_trial(empty, _spec(empty, target_fold=fold), "c")
    with pytest.raises(ValueError, match="M7 predicts and evaluates only V folds"):
        plan_dataset(empty, "full", fold, 2, "c")
    assert not (tmp_path / "artifacts").exists()


def test_rrf_baseline_refuses_t_folds(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="M7 evaluates only V folds"):
        rrf_baseline_row(tiny_inputs(tmp_path), "T1")
    assert not (tmp_path / "artifacts").exists()


def test_sample_job_refuses_t_folds(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="M7 trains only on folds of phases"):
        sample_job(tiny_inputs(tmp_path), "T1", "full", [2], "c")
    with pytest.raises(ValueError, match="not in sampling.ratios"):
        sample_job(tiny_inputs(tmp_path), "F1", "full", [40], "c")


# ---------------------------------------------------------------- one trial


def test_trial_trains_scores_and_evaluates(inputs: TrialInputs) -> None:
    outcome = run_trial(inputs, _spec(inputs), "c")
    assert outcome.train_folds == ("F1", "F2", "F3", "F4")
    assert (outcome.n_train_rows, outcome.n_train_queries) == (4 * N_QUERIES * 3, 4 * N_QUERIES)
    # Scoring uses ALL candidates of the fold, not a sample.
    assert outcome.n_scored_rows == N_QUERIES * CANDIDATES_PER_QUERY
    assert outcome.n_queries == N_QUERIES
    assert [row.num_trees for row in outcome.metric_rows] == [10, 20]
    last = outcome.metric_rows[-1]
    assert list(last.scores) == list(METRICS) and last.fold == "V1"
    assert last.train_metric is not None
    # The label is the candidate with the largest ui_decay_ptype: a tiny model finds it.
    assert last.scores["map@12"] > 0.8
    # The RRF baseline ranks the same candidates without that signal: about 1/12 at rank 1.
    baseline = rrf_baseline_row(inputs, "V1")
    assert baseline.num_trees == 0 and baseline.n_queries == N_QUERIES
    assert baseline.scores["map@12"] < 0.5 < last.scores["map@12"]

    key = (AREA_V_PHASE, "full", outcome.config_id, "V1")
    predictions = pl.read_parquet(inputs.paths.ranker_predictions_file(*key))
    assert predictions.columns == ["num_trees", "query_id", "rank", "article_idx", "score"]
    per_checkpoint = predictions.group_by("num_trees").agg(pl.len()).sort("num_trees")
    assert per_checkpoint.rows() == [(10, N_QUERIES * 12), (20, N_QUERIES * 12)]
    per_query = pl.read_parquet(inputs.paths.ranker_per_query_file(*key))
    assert per_query.columns == ["query_id", "fold", *METRICS, "num_trees"]
    assert per_query.height == 2 * N_QUERIES
    mean_at_20 = per_query.filter(pl.col("num_trees") == 20)["map@12"].mean()
    assert mean_at_20 == pytest.approx(last.scores["map@12"])
    model = load_model(inputs.paths.ranker_model_file(*key))
    assert model.num_trees() == 20 and model.num_feature() == 71
    assert len(outcome.model_sha256) == 64

    # Outputs are never replaced: the same trial again stops before training.
    with pytest.raises(FileExistsError, match="outputs of this trial already exist"):
        run_trial(inputs, _spec(inputs), "c")


def _check_script():  # noqa: ANN202 (a module loaded from a file)
    """scripts/checks/check_ranker_eval.py, loaded by path (scripts/ is not a package)."""
    path = REPO / "scripts" / "checks" / "check_ranker_eval.py"
    spec = importlib.util.spec_from_file_location("check_ranker_eval", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_independent_check_recomputes_the_trial_metrics(inputs: TrialInputs) -> None:
    # The check script reads the Top-12 file exactly as a trial saves it (5 columns,
    # num_trees first) and must reach the recorded metrics with its own plain loops.
    spec = _spec(inputs, area=AREA_SPEED_TEST, step="check")
    outcome = run_trial(inputs, spec, "c")
    check = _check_script()
    key = (AREA_SPEED_TEST, "full", outcome.config_id, "V1")
    predictions = pl.read_parquet(inputs.paths.ranker_predictions_file(*key))
    labels = pl.read_parquet(inputs.paths.labels_file("V1"))
    for metric_row in outcome.metric_rows:
        top = predictions.filter(pl.col("num_trees") == metric_row.num_trees)
        means, n_queries = check.recompute_fold_means(top, labels)
        assert n_queries == N_QUERIES
        for metric, value in zip(check.METRICS, means, strict=True):
            assert abs(value - metric_row.scores[metric]) < check.TOLERANCE
    # A ranking in the wrong order is caught: swap the scores of ranks 1 and 2 of a query.
    top = predictions.filter(pl.col("num_trees") == 20)
    first_query = top["query_id"].min()
    broken = top.with_columns(
        pl.when((pl.col("query_id") == first_query) & (pl.col("rank") == 1))
        .then(pl.lit(-1e9))
        .otherwise(pl.col("score"))
        .alias("score")
    )
    with pytest.raises(ValueError, match="not ordered by score DESC"):
        check.recompute_fold_means(broken, labels)
    # A query without labels in the predictions is an error, not silently ignored.
    with pytest.raises(ValueError, match="predicted queries have no label"):
        check.recompute_fold_means(top, labels.filter(pl.col("query_id") != first_query))


def test_check_script_query_metrics_by_hand() -> None:
    check = _check_script()
    # Labels {7, 9}; ranked 5, 7, 9: hits at ranks 2 and 3.
    # AP = (1/2 + 2/3) / min(12, 2) = 0.583333; Recall = 2/2; NDCG = (1/log2(3) + 1/2) /
    # (1 + 1/log2(3)).
    average_precision, recall, ndcg = check.query_metrics([5, 7, 9], {7, 9})
    assert average_precision == pytest.approx((1 / 2 + 2 / 3) / 2)
    assert recall == 1.0
    assert ndcg == pytest.approx((1 / math.log2(3) + 0.5) / (1 + 1 / math.log2(3)))
    assert check.query_metrics([], {1}) == (0.0, 0.0, 0.0)


def test_timing_only_trial_reads_no_label_and_saves_nothing(inputs: TrialInputs) -> None:
    spec = _spec(inputs, area=AREA_SPEED_TEST, step="histogram_row_wise", num_trees=10,
                 histogram_mode="row_wise", evaluate=False)  # fmt: skip
    labels_file = inputs.paths.labels_file("V1")
    hidden = labels_file.with_suffix(".hidden")
    labels_file.rename(hidden)  # the V1 labels are not there: a read would fail
    try:
        outcome = run_trial(inputs, spec, "c")
    finally:
        hidden.rename(labels_file)
    assert outcome.metric_rows == () and outcome.model_sha256 == ""
    assert (outcome.n_scored_rows, outcome.n_queries, outcome.score_seconds) == (0, 0, 0.0)
    assert outcome.train_seconds > 0 and outcome.seconds_per_tree > 0
    key = (AREA_SPEED_TEST, "full", outcome.config_id, "V1")
    assert not inputs.paths.ranker_model_file(*key).exists()
    assert not inputs.paths.ranker_predictions_file(*key).exists()


def test_trial_without_r6(inputs: TrialInputs) -> None:
    for fold in ("F1", "F2", "F3", "F4"):
        job = sample_job(inputs, fold, "minus_r6", [2], "c")
        records = sample_fold(job, inputs.ranker.sampling)
        assert len(records[2].columns) == 3 + 67
    plan = plan_dataset(inputs, "minus_r6", "V1", 2, "c")
    build_and_save_dataset(plan, 64)
    outcome = run_trial(inputs, _spec(inputs, experiment="minus_r6", step="minus_r6"), "c")
    wide = pl.concat(
        pl.read_parquet(path) for path in sorted(inputs.paths.features_dir("V1").glob("part-*"))
    )
    other_ranks = ["pop_rank", "rep_rank", "sib_rank", "icf_rank", "wals_rank"]
    only_r6 = wide.filter(pl.all_horizontal(pl.col(name).is_nan() for name in other_ranks))
    assert only_r6.height > 0
    # Rows returned only by R6 are not scored; every query is still evaluated.
    assert outcome.n_scored_rows == wide.height - only_r6.height
    assert outcome.n_queries == N_QUERIES
    key = (AREA_V_PHASE, "minus_r6", outcome.config_id, "V1")
    model = load_model(inputs.paths.ranker_model_file(*key))
    assert model.num_feature() == 67 and "tt_dot" not in model.feature_name()
    full_id = ranker_config_id(
        model_config(inputs.ranker, inputs.ranker.tuned_defaults, "full", "col_wise")
    )
    assert outcome.config_id != full_id


def test_trial_needs_its_dataset(tmp_path: Path) -> None:
    workspace = build_workspace(tmp_path)
    with pytest.raises(FileNotFoundError, match="the Dataset is not built"):
        run_trial(workspace, _spec(workspace), "c")
    with pytest.raises(FileNotFoundError, match="the sample is not finished"):
        plan_dataset(workspace, "full", "V2", 2, "c")  # V2 needs V1's sample, not drawn here


# ---------------------------------------------------------------- RRF baseline by hand


def test_rrf_baseline_hand_example(tmp_path: Path) -> None:
    inputs = tiny_inputs(tmp_path)
    rows = [
        {"query_id": 1, "article_idx": 1, "rrf_score": 0.02, "best_rank": 5.0},
        {"query_id": 1, "article_idx": 2, "rrf_score": 0.02, "best_rank": 5.0},
        {"query_id": 1, "article_idx": 3, "rrf_score": 0.03, "best_rank": 9.0},
        {"query_id": 1, "article_idx": 4, "rrf_score": 0.02, "best_rank": 3.0},
        {"query_id": 2, "article_idx": 9, "rrf_score": 0.01, "best_rank": 1.0},
    ]
    write_features(inputs.paths.features_dir("V1"), "V1", [wide_table(rows)])
    write_labels(inputs.paths.labels_file("V1"), "V1", [(1, 2), (2, 8)])
    row = rrf_baseline_row(inputs, "V1")
    assert (row.experiment, row.config_id, row.num_trees) == ("rrf_baseline", "rrf_baseline", 0)
    assert row.n_queries == 2 and row.train_metric is None
    assert row.scores["map@12"] == pytest.approx(0.125)
    assert row.scores["recall@12"] == pytest.approx(0.5)
    assert row.scores["ndcg@12"] == pytest.approx(1 / (2 * math.log2(5)))
