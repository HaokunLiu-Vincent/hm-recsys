"""Tests for hmrec.ranker.train: parameters, the histogram rule, deterministic training."""

from dataclasses import replace
from pathlib import Path

import lightgbm as lgb
import pytest
from ranker_example import REPO, build_workspace

from hmrec.config import load_ranker_config
from hmrec.hpo.v_trial import (
    TrialInputs,
    build_and_save_dataset,
    load_planned_dataset,
    plan_dataset,
)
from hmrec.ranker.train import (
    TrainLog,
    dataset_params,
    decide_histogram_mode,
    first_trees_sha256,
    gain_importance,
    load_model,
    save_model,
    train_ranker,
    training_params,
)

REAL = load_ranker_config(REPO / "configs" / "ranker.yaml")


@pytest.fixture(scope="module")
def inputs(tmp_path_factory: pytest.TempPathFactory) -> TrialInputs:
    workspace = build_workspace(tmp_path_factory.mktemp("train"))
    plan = plan_dataset(workspace, "full", "V1", 2, "test-commit")
    build_and_save_dataset(plan, workspace.ranker.sampling.row_group_rows)
    return workspace


def _train(
    inputs: TrialInputs, num_trees: int, mode: str = "col_wise"
) -> tuple[lgb.Booster, TrainLog]:
    params = training_params(inputs.ranker, inputs.ranker.tuned_defaults, mode, 2)
    dataset, _ = load_planned_dataset(plan_dataset(inputs, "full", "V1", 2, "c"), params)
    return train_ranker(dataset, params, num_trees, inputs.ranker.train_metric_every)


# ---------------------------------------------------------------- parameters


def test_training_params_of_the_real_config() -> None:
    params = training_params(REAL, REAL.tuned_defaults, "col_wise", 6)
    # proposal §9.3
    assert params["objective"] == "lambdarank"
    assert (params["learning_rate"], params["feature_fraction"]) == (0.05, 0.8)
    assert (params["bagging_fraction"], params["bagging_freq"]) == (0.8, 1)
    assert (params["max_bin"], params["eval_at"], params["num_threads"]) == (255, [12], 6)
    # tuned defaults and reproducibility
    assert (params["num_leaves"], params["min_data_in_leaf"]) == (31, 500)
    assert params["seed"] == 42 and params["deterministic"] is True
    assert params["force_col_wise"] is True and "force_row_wise" not in params
    assert params["feature_pre_filter"] is False
    assert params["lambdarank_truncation_level"] == 30 and params["bagging_by_query"] is False
    # negatives_per_positive chooses the Dataset; it is not a LightGBM parameter.
    assert "negatives_per_positive" not in params
    row_wise = training_params(REAL, REAL.tuned_defaults, "row_wise", 6)
    assert row_wise["force_row_wise"] is True and "force_col_wise" not in row_wise


def test_undecided_histogram_mode_cannot_train() -> None:
    with pytest.raises(ValueError, match="decided by the speed test"):
        training_params(REAL, REAL.tuned_defaults, "undecided", 6)
    # The real config holds the speed test's decision (2026-11-18).
    assert REAL.histogram.mode == "row_wise"
    assert training_params(REAL, REAL.tuned_defaults, REAL.histogram.mode, 6)["force_row_wise"]


def test_dataset_params_define_the_bins() -> None:
    params = dataset_params(REAL, 6)
    assert params["max_bin"] == 255 and params["seed"] == 42 and params["num_threads"] == 6
    assert "num_leaves" not in params and "objective" not in params


@pytest.mark.parametrize(
    ("col_seconds", "row_seconds", "row_peak", "expected"),
    [
        (1.0, 0.8, 5.0, "row_wise"),  # 20% faster, within 8 GB
        (4.0, 3.4, 5.0, "row_wise"),  # exactly 15% faster: enough
        (1.0, 0.9, 5.0, "col_wise"),  # only 10% faster
        (1.0, 1.2, 5.0, "col_wise"),  # slower
        (1.0, 0.5, 8.5, "col_wise"),  # much faster but above the memory limit
        (1.0, 0.5, 8.0, "row_wise"),  # exactly at the limit is allowed
    ],
)
def test_histogram_rule(
    col_seconds: float, row_seconds: float, row_peak: float, expected: str
) -> None:
    assert decide_histogram_mode(col_seconds, row_seconds, row_peak, 8.0, 0.15) == expected


def test_histogram_rule_needs_positive_times() -> None:
    with pytest.raises(ValueError, match="must be > 0"):
        decide_histogram_mode(0.0, 1.0, 1.0, 8.0, 0.15)


# ---------------------------------------------------------------- training


def test_training_is_deterministic_and_exact(inputs: TrialInputs) -> None:
    first, log = _train(inputs, 20)
    second, _ = _train(inputs, 20)
    assert first.num_trees() == 20
    assert first.model_to_string() == second.model_to_string()
    assert len(log.elapsed_seconds) == 20
    assert list(log.elapsed_seconds) == sorted(log.elapsed_seconds)
    # The training metric is taken every 10 trees (train_metric_every) only.
    assert sorted(log.train_metric) == [10, 20]
    assert 0.0 < log.train_metric[10] <= log.train_metric[20] <= 1.0
    assert log.total_seconds == log.elapsed_seconds[-1]
    per_tree = log.seconds_per_tree(2)
    assert per_tree == pytest.approx((log.elapsed_seconds[-1] - log.elapsed_seconds[1]) / 18)
    with pytest.raises(ValueError, match="warmup_trees"):
        log.seconds_per_tree(20)


def test_first_trees_do_not_depend_on_how_many_follow(inputs: TrialInputs) -> None:
    # The curve over the number of trees relies on this: a model trained to 20 trees
    # starts with exactly the model trained to 10 trees.
    long_model, _ = _train(inputs, 20)
    short_model, _ = _train(inputs, 10)
    assert first_trees_sha256(long_model, 10) == first_trees_sha256(short_model, 10)
    assert first_trees_sha256(long_model, 20) != first_trees_sha256(long_model, 10)
    with pytest.raises(ValueError, match="num_trees must be in"):
        first_trees_sha256(short_model, 11)


def test_both_histogram_modes_train(inputs: TrialInputs) -> None:
    row_wise, _ = _train(inputs, 5, "row_wise")
    assert row_wise.num_trees() == 5


def test_model_files_are_never_replaced(inputs: TrialInputs, tmp_path: Path) -> None:
    booster, _ = _train(inputs, 10)
    path = tmp_path / "models" / "model.txt"
    sha256 = save_model(booster, path)
    assert len(sha256) == 64
    assert load_model(path).model_to_string() == booster.model_to_string()
    with pytest.raises(FileExistsError, match="never replaced"):
        save_model(booster, path)
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "missing.txt")


def test_gain_importance_names_every_feature(inputs: TrialInputs) -> None:
    booster, _ = _train(inputs, 10)
    gains = gain_importance(booster, 10)
    assert list(gains) == [feature.name for feature in inputs.lock.features]
    # The label is the candidate with the largest ui_decay_ptype: it must dominate.
    assert max(gains, key=lambda name: gains[name]) == "ui_decay_ptype"
    assert gain_importance(booster, 5)["ui_decay_ptype"] < gains["ui_decay_ptype"]


def test_train_log_is_a_plain_record() -> None:
    log = TrainLog((1.0, 2.0, 4.0), {3: 0.5})
    assert log.seconds_per_tree(0) == pytest.approx(4.0 / 3)
    assert log.seconds_per_tree(1) == pytest.approx(1.5)
    assert replace(log, train_metric={}).train_metric == {}
