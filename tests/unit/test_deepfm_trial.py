"""Tests for deepfm.trial, benchmark, records and process on the tiny workspace of deepfm_example.

The workspace: training folds F1-F4 and target V1, 30 queries x 12 candidates
each, positives = the 2 candidates with the largest value of the first numeric
feature (so a trained model beats a random order).
"""

import csv
import dataclasses
import json
import shutil
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch
from deepfm_example import (
    MANIFEST,
    N_CANDIDATES,
    N_QUERIES,
    RATIO,
    REPO,
    TARGET,
    TRAIN_FOLDS,
    build_workspace,
)

from hmrec.csv_records import read_rows
from hmrec.deepfm.benchmark import choose_device, run_benchmark
from hmrec.deepfm.process import (
    below_baseline,
    finish,
    record_benchmark,
    record_trial,
    require_clean_tree,
    speed_test_spec,
    with_seed,
)
from hmrec.deepfm.records import (
    BENCHMARK_COLUMNS,
    TRIAL_COLUMNS,
    ReferenceScores,
    deepfm_config_id,
    metric_columns,
    read_reference_scores,
)
from hmrec.deepfm.trial import (
    AREA_SPEED_TEST,
    AREA_V_PHASE,
    TrialInputs,
    TrialResult,
    TrialSpec,
    check_spec,
    run_trial,
)
from hmrec.locks import file_sha256, load_ranker_lock
from hmrec.memory import ProcessPeakMemory
from hmrec.run_records import RunStamp

SPEC = TrialSpec(AREA_V_PHASE, TARGET, 42, 2)
STAMP = RunStamp("abc123", False)
METRICS = ("map@12", "recall@12", "ndcg@12")


def _trial_dir(inputs: TrialInputs, spec: TrialSpec = SPEC) -> Path:
    return inputs.paths.deepfm_trial_dir(
        spec.area, deepfm_config_id(inputs.config), spec.seed, spec.target_fold
    )


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> tuple[TrialInputs, TrialResult]:
    """One finished trial on the tiny workspace (shared by the read-only tests)."""
    inputs = build_workspace(tmp_path_factory.mktemp("deepfm"))
    return inputs, run_trial(inputs, SPEC)


# ---------------------------------------------------------------- one trial


def test_trial_trains_on_the_window_and_evaluates_every_epoch(
    trained: tuple[TrialInputs, TrialResult],
) -> None:
    inputs, result = trained
    assert result.train_folds == TRAIN_FOLDS and result.device == "cpu"
    # 4 folds x 30 queries x 12 candidates: with 10 negatives per query every row is kept.
    assert (result.n_train_rows, result.n_train_queries) == (4 * 360, 4 * 30)
    assert result.n_scored_rows == N_QUERIES * N_CANDIDATES
    assert [epoch.epoch for epoch in result.epochs] == [1, 2]
    assert [metric.epoch for metric in result.metrics] == [1, 2]
    for metric in result.metrics:
        assert metric.n_queries == N_QUERIES and set(metric.scores) == set(METRICS)
        assert 0 < metric.scores["map@12"] <= 1
    assert result.metrics[0].train_loss == result.epochs[0].mean_loss
    assert result.config_id == deepfm_config_id(inputs.config)


def test_trial_files_are_under_artifacts_deepfm(trained: tuple[TrialInputs, TrialResult]) -> None:
    inputs, _ = trained
    directory = _trial_dir(inputs)
    assert inputs.paths.deepfm_dir in directory.parents
    assert sorted(path.name for path in directory.iterdir()) == [
        "epoch1.pt",
        "epoch2.pt",
        "per_query.parquet",
        "predictions.parquet",
        "transform.json",
        "trial.json",
    ]
    predictions = pl.read_parquet(directory / "predictions.parquet")
    # The Top-12 of all 30 queries for each of the 2 epochs (a query has 12 candidates).
    assert predictions.height == 2 * N_QUERIES * 12
    assert predictions.columns == ["epoch", "query_id", "rank", "article_idx", "score"]
    one_query = predictions.filter((pl.col("epoch") == 2) & (pl.col("rank") <= 12)).head(12)
    assert one_query["score"].to_list() == sorted(one_query["score"].to_list(), reverse=True)
    per_query = pl.read_parquet(directory / "per_query.parquet")
    assert per_query.height == 2 * N_QUERIES and set(METRICS) <= set(per_query.columns)


def test_a_trained_model_beats_a_random_order(tmp_path: Path) -> None:
    inputs = build_workspace(tmp_path, epochs=40, learning_rate=0.01)
    result = run_trial(inputs, TrialSpec(AREA_V_PHASE, TARGET, 42, 40))
    # 2 positives among 12 candidates: a random order has MAP@12 of about 0.3.
    assert result.metrics[-1].scores["map@12"] > 0.6
    assert result.epochs[-1].mean_loss < result.epochs[0].mean_loss


def test_same_seed_gives_the_same_models_and_metrics(tmp_path: Path) -> None:
    torch.set_num_threads(1)
    first_inputs = build_workspace(tmp_path / "a")
    second_inputs = build_workspace(tmp_path / "b")
    first, second = run_trial(first_inputs, SPEC), run_trial(second_inputs, SPEC)
    assert [m.scores for m in first.metrics] == [m.scores for m in second.metrics]
    for name in ("epoch1.pt", "epoch2.pt"):
        one = torch.load(_trial_dir(first_inputs) / name, weights_only=True)
        two = torch.load(_trial_dir(second_inputs) / name, weights_only=True)
        assert all(torch.equal(one[key], two[key]) for key in one)
    other = run_trial(first_inputs, dataclasses.replace(SPEC, seed=43))
    assert other.metrics[0].train_loss != first.metrics[0].train_loss


def test_bin_edges_and_models_do_not_depend_on_the_target_week(tmp_path: Path) -> None:
    # Two workspaces with the same training folds but DIFFERENT target-week values.
    first_inputs = build_workspace(tmp_path / "a", target_seed=100)
    second_inputs = build_workspace(tmp_path / "b", target_seed=999)
    first, second = run_trial(first_inputs, SPEC), run_trial(second_inputs, SPEC)
    one, two = _trial_dir(first_inputs), _trial_dir(second_inputs)
    # The fitted transform (bin edges) and the trained weights are identical ...
    assert (one / "transform.json").read_text() == (two / "transform.json").read_text()
    for name in ("epoch1.pt", "epoch2.pt"):
        a = torch.load(one / name, weights_only=True)
        b = torch.load(two / name, weights_only=True)
        assert all(torch.equal(a[key], b[key]) for key in a)
    # ... while the target weeks really differ.
    assert first.metrics[-1].scores != second.metrics[-1].scores
    edges = json.loads((one / "transform.json").read_text())["edges"]
    assert len(edges) == 63 and all(len(feature) <= 3 for feature in edges)  # 4 bins


@pytest.mark.parametrize(
    ("changes", "label"),
    [
        ({"loss": "listwise_softmax"}, "listwise"),
        ({}, "standardised"),
    ],
)
def test_variants_run_through_a_whole_trial(tmp_path: Path, changes: dict, label: str) -> None:
    inputs = build_workspace(tmp_path, **changes)
    if label == "standardised":
        model = dataclasses.replace(inputs.config.model, numeric_input="standardised")
        inputs = dataclasses.replace(inputs, config=dataclasses.replace(inputs.config, model=model))
    result = run_trial(inputs, SPEC)
    assert all(np.isfinite(epoch.mean_loss) for epoch in result.epochs)
    assert result.metrics[-1].n_queries == N_QUERIES
    # A variant is another configuration: another id, another folder.
    assert result.config_id != deepfm_config_id(build_workspace(tmp_path / "default").config)


# ---------------------------------------------------------------- what a trial refuses


@pytest.mark.parametrize("fold", ["T1", "T2", "T3", "T4"])
def test_t_folds_are_refused_before_anything_is_read(tmp_path: Path, fold: str) -> None:
    inputs = build_workspace(tmp_path)
    with pytest.raises(ValueError, match="T folds are evaluated in M8"):
        run_trial(inputs, dataclasses.replace(SPEC, target_fold=fold))
    assert not inputs.paths.deepfm_dir.exists()  # nothing was written


def test_other_folds_and_bad_specs_are_refused(tmp_path: Path) -> None:
    inputs = build_workspace(tmp_path)
    with pytest.raises(ValueError, match="M7 predicts and evaluates only V folds"):
        check_spec(inputs, dataclasses.replace(SPEC, target_fold="F4"))
    with pytest.raises(ValueError, match="area must be one of"):
        check_spec(inputs, dataclasses.replace(SPEC, area="t_phase"))
    with pytest.raises(ValueError, match="epochs must be > 0"):
        check_spec(inputs, dataclasses.replace(SPEC, epochs=0))
    # V1's window is F1-F4; all of them may train.
    assert [fold.name for fold in check_spec(inputs, SPEC)] == list(TRAIN_FOLDS)


def test_undecided_device_is_refused(tmp_path: Path) -> None:
    inputs = build_workspace(tmp_path, device="undecided")
    with pytest.raises(ValueError, match="still undecided"):
        run_trial(inputs, SPEC)


def test_sample_file_must_be_the_one_lightgbm_trained_on(tmp_path: Path) -> None:
    inputs = build_workspace(tmp_path)
    paths = inputs.paths
    record = paths.ranker_dataset_dir("full", TARGET, RATIO) / "dataset.json"
    content = json.loads(record.read_text())
    content["sample_sha256"]["F2"] = "0" * 64
    record.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="not the one LightGBM trained on"):
        run_trial(inputs, SPEC)
    # A file changed on disk no longer matches its own record either.
    build = build_workspace(tmp_path / "other")
    sample = build.paths.ranker_sample_file("full", "F3", RATIO)
    sample.write_bytes(sample.read_bytes() + b"x")
    with pytest.raises(ValueError, match="sha256"):
        run_trial(build, SPEC)
    # Without a LightGBM Dataset of the week there is nothing to compare with.
    shutil.rmtree(paths.ranker_dataset_dir("full", TARGET, RATIO))
    with pytest.raises(FileNotFoundError, match="LightGBM has no Dataset"):
        run_trial(inputs, SPEC)


def test_wide_table_that_is_not_the_recorded_one_is_refused(tmp_path: Path) -> None:
    inputs = build_workspace(tmp_path)
    # The record of the target week's wide table names another file content.
    record = inputs.paths.features_metadata_file(TARGET)
    content = json.loads(record.read_text())
    content["parts"][0]["sha256"] = "0" * 64
    record.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="sha256"):
        run_trial(inputs, SPEC)


def test_a_trial_is_never_replaced(trained: tuple[TrialInputs, TrialResult]) -> None:
    inputs, _ = trained
    with pytest.raises(FileExistsError, match="finished trial"):
        run_trial(inputs, SPEC)


# ---------------------------------------------------------------- records


def _reference_workspace(tmp_path: Path) -> TrialInputs:
    """A workspace with the real ranker lock and a small copy of the M7 metric table."""
    inputs = build_workspace(tmp_path)
    paths = inputs.paths
    paths.locks_dir.mkdir(parents=True)
    shutil.copyfile(REPO / "configs" / "locks" / "ranker.lock.yaml", paths.ranker_lock_file)
    lock = load_ranker_lock(paths.ranker_lock_file)
    header = ["experiment", "config_id", "fold", "num_trees", "n_queries", "map@12"]
    rows = [
        ["rrf_baseline", "rrf_baseline", "V1", "0", "30", "0.018511"],
        ["full", lock.selection.config_id, "V1", str(lock.model.num_trees), "30", "0.030112"],
        ["full", lock.selection.config_id, "V1", "100", "30", "0.029"],  # another checkpoint
        ["full", "ffffffffffff", "V1", str(lock.model.num_trees), "30", "0.031"],  # another config
        ["full", lock.selection.config_id, "V2", str(lock.model.num_trees), "30", "0.032"],
    ]
    paths.results_root.mkdir(parents=True, exist_ok=True)
    with paths.v_phase_metrics_file.open("w", newline="") as handle:
        csv.writer(handle).writerows([header, *rows])
    return inputs


def test_reference_scores_are_read_from_the_m7_records(tmp_path: Path) -> None:
    inputs = _reference_workspace(tmp_path)
    reference = read_reference_scores(inputs.paths, "V1", "map@12")
    # The RRF row, and the row of the LOCKED configuration at the LOCKED number of trees.
    assert reference == ReferenceScores("V1", "map@12", 0.018511, 0.030112)
    with pytest.raises(ValueError, match="RRF baseline on V3"):
        read_reference_scores(inputs.paths, "V3", "map@12")


def test_trial_rows_are_written_metrics_first(
    trained: tuple[TrialInputs, TrialResult], tmp_path: Path
) -> None:
    inputs, result = trained
    paths = dataclasses.replace(inputs.paths, results_root=tmp_path)
    inputs = dataclasses.replace(inputs, paths=paths)
    peak = ProcessPeakMemory(2 * 1024**3, 0)
    record_trial(inputs, SPEC, result, 0.0, peak, STAMP)
    metric_rows = read_rows(paths.deepfm_metrics_file, metric_columns(METRICS))
    assert [(row["fold"], row["epoch"], row["seed"]) for row in metric_rows] == [
        ("V1", "1", "42"),
        ("V1", "2", "42"),
    ]
    assert float(metric_rows[1]["map@12"]) == result.metrics[1].scores["map@12"]
    trial_rows = read_rows(paths.deepfm_trials_file, TRIAL_COLUMNS)
    assert len(trial_rows) == 1
    row = trial_rows[0]
    assert (row["area"], row["train_folds"], row["device"]) == ("v_phase", "F1 F2 F3 F4", "cpu")
    assert (row["n_train_rows"], row["epochs"], row["git_commit"]) == ("1440", "2", "abc123")
    assert float(row["process_max_rss_gb"]) == 2.0
    assert json.loads(row["config"])["model"]["embedding_dim"] == 4
    # A true peak above the limit stops the run AFTER its rows are written.
    with pytest.raises(MemoryError, match="above the limit"):
        record_trial(inputs, SPEC, result, 0.0, ProcessPeakMemory(13 * 1024**3, 0), STAMP)
    assert len(read_rows(paths.deepfm_trials_file, TRIAL_COLUMNS)) == 2


def test_config_id_ignores_seed_epochs_and_device_only(tmp_path: Path) -> None:
    config = build_workspace(tmp_path).config
    base = deepfm_config_id(config)
    training = config.training
    for same in (
        dataclasses.replace(training, seed=7),
        dataclasses.replace(training, epochs=1),
        dataclasses.replace(training, device="mps"),
    ):
        assert deepfm_config_id(dataclasses.replace(config, training=same)) == base
    different = dataclasses.replace(training, learning_rate=0.003)
    assert deepfm_config_id(dataclasses.replace(config, training=different)) != base
    assert deepfm_config_id(dataclasses.replace(config, n_bins=8)) != base
    assert with_seed(config, 44).training.seed == 44 and len(base) == 12


def test_speed_test_spec_and_its_stop_rule(trained: tuple[TrialInputs, TrialResult]) -> None:
    inputs, result = trained
    spec = speed_test_spec(inputs)
    assert (spec.area, spec.target_fold, spec.seed, spec.epochs) == (AREA_SPEED_TEST, "V1", 42, 1)
    score = result.metrics[-1].scores["map@12"]
    assert not below_baseline(result, ReferenceScores("V1", "map@12", score - 0.01, 0.9))
    assert below_baseline(result, ReferenceScores("V1", "map@12", score + 0.01, 0.9))
    require_clean_tree(STAMP)
    with pytest.raises(RuntimeError, match="starts only from committed code"):
        require_clean_tree(RunStamp("abc123", True))


# ---------------------------------------------------------------- device benchmark


def test_device_rule() -> None:
    # MPS must be at least 1.5 times faster than CPU, otherwise CPU.
    assert choose_device(cpu_step_ms=30.0, mps_step_ms=20.0, min_mps_speedup=1.5) == "mps"
    assert choose_device(cpu_step_ms=30.0, mps_step_ms=20.1, min_mps_speedup=1.5) == "cpu"
    assert choose_device(cpu_step_ms=10.0, mps_step_ms=30.0, min_mps_speedup=1.5) == "cpu"
    with pytest.raises(ValueError, match="must be > 0"):
        choose_device(0.0, 1.0, 1.5)


def test_benchmark_on_cpu_repeats_exactly_and_is_recorded(tmp_path: Path) -> None:
    inputs = build_workspace(tmp_path, device="undecided")  # the benchmark names its device
    speed_test = dataclasses.replace(
        inputs.config.speed_test,
        benchmark_steps=10,
        benchmark_discard_steps=2,
        benchmark_scoring_rows=500,
    )
    inputs = dataclasses.replace(
        inputs, config=dataclasses.replace(inputs.config, speed_test=speed_test)
    )
    torch.set_num_threads(1)
    result = run_benchmark(inputs, "cpu")
    assert (result.device, result.steps, result.discarded_steps) == ("cpu", 10, 2)
    assert result.same_seed_identical and result.same_seed_max_difference == 0.0
    assert result.median_step_ms > 0 and result.scoring_rows_per_second > 0
    assert result.scoring_rows == 500 and result.target_fold == "V1"
    record_benchmark(inputs, result, finish(inputs, "test"), STAMP)
    rows = read_rows(inputs.paths.deepfm_benchmark_file, BENCHMARK_COLUMNS)
    assert [(row["device"], row["steps"], row["same_seed_losses_identical"]) for row in rows] == [
        ("cpu", "10", "True")
    ]
    with pytest.raises(ValueError, match="device must be one of"):
        run_benchmark(inputs, "cuda")
    # 1,440 training rows in batches of 64 give 23 batches: 300 steps are not possible.
    too_long = dataclasses.replace(speed_test, benchmark_steps=300)
    more = dataclasses.replace(
        inputs, config=dataclasses.replace(inputs.config, speed_test=too_long)
    )
    with pytest.raises(ValueError, match="only 23 batches"):
        run_benchmark(more, "cpu")
    assert MANIFEST.get("V1").phase == "V" and file_sha256(inputs.paths.category_codes_file)
