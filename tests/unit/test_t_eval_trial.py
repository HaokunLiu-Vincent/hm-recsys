"""Tests for hmrec.t_eval.trial and hmrec.t_eval.rehearsal on the tiny workspace (§16.12).

The workspace (t_eval_example): folds F4 ... T4 with sample files as M7 and M8
left them, and the V-phase record of the chosen configuration for V4 (trained
here with the V-phase code). The rehearsal then runs the T code path on V4 and
must reproduce that record bit for bit.
"""

import csv
from pathlib import Path

import polars as pl
import pytest
from deepfm_example import RATIO
from t_eval_example import (
    COMMIT,
    METRICS,
    NOW,
    OTHER_COMMIT,
    SEEDS,
    build_t_eval_workspace,
    context_of,
    copy_workspace,
    record_rehearsal,
    run_and_record,
    t_eval_config,
    write_v_records,
)

from hmrec.csv_records import read_rows
from hmrec.deepfm.records import TRIAL_COLUMNS, deepfm_config_id, metric_columns
from hmrec.deepfm.trial import TrialInputs, TrialSpec, run_trial
from hmrec.eval.t_eval_labels import TEvalPermit
from hmrec.t_eval.gate import (
    FAILED,
    PASSED,
    PURPOSE,
    REHEARSAL_CHECK_COLUMNS,
    begin_formal_run,
)
from hmrec.t_eval.plan import STAGE_FORMAL, STAGE_REHEARSAL, TrialKey, area_of
from hmrec.t_eval.rehearsal import (
    PER_QUERY_FILE,
    PREDICTIONS_FILE,
    Check,
    check_rehearsal,
    compare_tables,
    compare_values,
    write_check_row,
)
from hmrec.t_eval.trial import chosen_trial_inputs, run_t_eval_trial, t_eval_door

SCRIPT = "scripts/m9_t_eval.py"
PERMIT = TEvalPermit(COMMIT, NOW, NOW, PURPOSE, SCRIPT)
CONFIG_ID = t_eval_config().config_id


@pytest.fixture(scope="module")
def rehearsed(tmp_path_factory: pytest.TempPathFactory) -> TrialInputs:
    """A workspace with the V records and a rehearsal of every seed, recorded at COMMIT."""
    inputs = build_t_eval_workspace(tmp_path_factory.mktemp("t_eval") / "workspace")
    write_v_records(inputs)
    context = context_of(inputs)
    for seed in SEEDS:
        run_and_record(inputs, context, TrialKey(STAGE_REHEARSAL, seed, "V4"))
    return inputs


def _passed(checks: list[Check]) -> dict[str, bool]:
    return {check.name: check.passed for check in checks}


# ---------------------------------------------------------------- the configuration


def test_the_chosen_configuration_is_built_from_the_config_files(rehearsed: TrialInputs) -> None:
    chosen = chosen_trial_inputs(rehearsed, t_eval_config(), 43)
    config = chosen.config
    assert (config.model.numeric_input, config.training.loss) == (
        "standardised",
        "listwise_softmax",
    )
    assert (config.training.seed, config.training.epochs) == (43, 1)
    assert deepfm_config_id(config) == CONFIG_ID
    # The inputs read from the config files are not changed.
    assert rehearsed.config.training.loss == "pointwise_bce"


def test_another_configuration_than_the_recorded_one_is_never_trained(
    rehearsed: TrialInputs,
) -> None:
    with pytest.raises(ValueError, match="Nothing is trained"):
        chosen_trial_inputs(rehearsed, t_eval_config(config_id="0" * 12), 42)
    # A changed setting gives another id than the recorded one.
    with pytest.raises(ValueError, match="chosen on V"):
        chosen_trial_inputs(rehearsed, t_eval_config(dropout=0.1), 42)
    context = context_of(rehearsed, config_id="0" * 12)
    with pytest.raises(ValueError, match="Nothing is trained"):
        run_t_eval_trial(context, rehearsed, TrialKey(STAGE_REHEARSAL, 42, "V4"), None)


# ---------------------------------------------------------------- the rehearsal


def test_the_rehearsal_reproduces_the_v_records_bit_for_bit(rehearsed: TrialInputs) -> None:
    context = context_of(rehearsed)
    checks = check_rehearsal(context, METRICS)
    expected = [
        f"seed {seed}: {what}"
        for seed in SEEDS
        for what in ("code_state", "values", "top_k", "per_query")
    ]
    assert [check.name for check in checks] == expected
    assert all(check.passed for check in checks), [c for c in checks if not c.passed]
    # The rehearsal trained on the M7 window of V4 and wrote into its own area and tables.
    paths = rehearsed.paths
    rows = read_rows(paths.m9_t_trials_file, TRIAL_COLUMNS)
    area = area_of(STAGE_REHEARSAL, COMMIT)
    assert [(row["area"], row["fold"], row["train_folds"]) for row in rows] == [
        (area, "V4", "F4 V1 V2 V3")
    ] * len(SEEDS)
    assert (paths.deepfm_trial_dir(area, CONFIG_ID, 42, "V4") / PREDICTIONS_FILE).is_file()
    # The V tables hold the V records only.
    v_rows = read_rows(paths.deepfm_trials_file, TRIAL_COLUMNS)
    assert {row["area"] for row in v_rows} == {"v_phase"}


def test_the_check_row_says_passed_only_if_every_check_passed(tmp_path: Path) -> None:
    context = context_of(build_t_eval_workspace(tmp_path))
    good = [Check("seed 42: values", True, "equal"), Check("seed 42: top_k", True, "equal")]
    assert write_check_row(context, good, NOW) == PASSED
    bad = [*good, Check("seed 43: values", False, "map@12: 0.1 != 0.2")]
    assert write_check_row(context, bad, NOW) == FAILED
    # No check at all is not a pass.
    assert write_check_row(context, [], NOW) == FAILED
    rows = read_rows(context.paths.m9_t_rehearsal_check_file, REHEARSAL_CHECK_COLUMNS)
    assert [(row["status"], row["failed_checks"]) for row in rows] == [
        (PASSED, ""),
        (FAILED, "seed 43: values"),
        (FAILED, ""),
    ]
    assert rows[0]["seeds"] == "42 43" and rows[0]["config_id"] == context.config.config_id
    assert (rows[0]["git_commit"], rows[0]["git_dirty"]) == (COMMIT, "False")


def test_a_changed_v_record_fails_the_rehearsal(rehearsed: TrialInputs, tmp_path: Path) -> None:
    inputs = copy_workspace(rehearsed, tmp_path / "copy")
    paths = inputs.paths
    # 1. Another metric value in the V record of seed 42 (the last bit of MAP@12).
    with paths.deepfm_metrics_file.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    changed = float(rows[0]["map@12"]) + 1e-17 + abs(float(rows[0]["map@12"])) * 3e-16
    assert changed != float(rows[0]["map@12"])
    rows[0]["map@12"] = repr(changed)
    with paths.deepfm_metrics_file.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_columns(METRICS)))
        writer.writeheader()
        writer.writerows(rows)
    # 2. Another score in the saved Top-K of seed 43.
    top_file = paths.deepfm_trial_dir("v_phase", CONFIG_ID, 43, "V4") / PREDICTIONS_FILE
    top = pl.read_parquet(top_file)
    first = pl.int_range(pl.len()) == 0
    top.with_columns(
        pl.when(first).then(pl.col("score") + 1.0).otherwise(pl.col("score")).alias("score")
    ).write_parquet(top_file)
    passed = _passed(check_rehearsal(context_of(inputs), METRICS))
    assert not passed["seed 42: values"] and not passed["seed 43: top_k"]
    assert passed["seed 42: top_k"] and passed["seed 43: values"] and passed["seed 43: per_query"]


def test_a_rehearsal_of_another_commit_does_not_count(rehearsed: TrialInputs) -> None:
    # At another commit the area is another one: nothing is recorded there.
    other = _passed(check_rehearsal(context_of(rehearsed, commit=OTHER_COMMIT), METRICS))
    assert other == {f"seed {seed}: code_state": False for seed in SEEDS}


def test_values_and_tables_are_compared_exactly() -> None:
    row = {
        "epoch": "1",
        "n_queries": "30",
        "map@12": "0.1",
        "recall@12": "0.2",
        "ndcg@12": "0.3",
        "train_loss": "1.5",
    }
    assert compare_values("values", row, dict(row), METRICS).passed
    for column, value in (("n_queries", "31"), ("ndcg@12", "0.30000000000000004"), ("epoch", "2")):
        check = compare_values("values", row, {**row, column: value}, METRICS)
        assert not check.passed and column in check.detail
    assert not compare_values("values", row, {**row, "train_loss": "1.6"}, METRICS).passed
    table = pl.DataFrame({"query_id": [1, 2], "score": [0.5, 0.25]})
    assert compare_tables("top_k", table, table.clone()).passed
    assert not compare_tables("top_k", table, table.reverse()).passed
    assert not compare_tables("top_k", table, table.head(1)).passed
    assert not compare_tables("top_k", table, table.cast({"score": pl.Float32})).passed


# ---------------------------------------------------------------- the formal run


def test_a_formal_training_needs_the_permit_before_any_file_is_read(tmp_path: Path) -> None:
    inputs = build_t_eval_workspace(tmp_path)
    context = context_of(inputs)
    key = TrialKey(STAGE_FORMAL, 42, "T2")
    chosen = chosen_trial_inputs(inputs, context.config, 42)
    # Remove everything a training would read: without a permit none of it is touched.
    for fold in ("V2", "V3", "V4", "T1"):
        inputs.paths.ranker_sample_file("full", fold, RATIO).unlink()
    inputs.paths.labels_file("T2").unlink()
    for not_a_permit in (None, "permit"):
        with pytest.raises(TypeError, match="only with a TEvalPermit"):
            t_eval_door(context, chosen, key, not_a_permit)  # type: ignore[arg-type]  # wrong type on purpose
        with pytest.raises(TypeError, match="only with a TEvalPermit"):
            run_t_eval_trial(context, inputs, key, not_a_permit)  # type: ignore[arg-type]  # wrong type on purpose
    assert not inputs.paths.deepfm_dir.exists()


def test_a_formal_training_predicts_a_t_fold_with_the_window_of_m8(tmp_path: Path) -> None:
    inputs = build_t_eval_workspace(tmp_path)
    context = context_of(inputs)
    record_rehearsal(context)
    begin_formal_run(context, NOW, SCRIPT)
    result = run_and_record(inputs, context, TrialKey(STAGE_FORMAL, 43, "T2"))
    assert result.train_folds == ("V2", "V3", "V4", "T1")
    assert result.config_id == CONFIG_ID
    assert result.metrics[0].n_queries == 30
    paths = inputs.paths
    directory = paths.deepfm_trial_dir("m9_t", CONFIG_ID, 43, "T2")
    per_query = pl.read_parquet(directory / PER_QUERY_FILE)
    assert per_query["fold"].unique().to_list() == ["T2"] and per_query.height == 30
    rows = read_rows(paths.m9_t_metrics_file, metric_columns(METRICS))
    assert [(row["area"], row["seed"], row["fold"], row["epoch"]) for row in rows] == [
        ("m9_t", "43", "T2", "1")
    ]
    assert float(rows[0]["map@12"]) == result.metrics[0].scores["map@12"]
    # Nothing was written to the V tables.
    assert not paths.deepfm_trials_file.exists() and not paths.deepfm_metrics_file.exists()


def test_a_sample_file_that_is_not_the_one_of_m8_is_refused(tmp_path: Path) -> None:
    inputs = build_t_eval_workspace(tmp_path)
    context = context_of(inputs)
    # Another file under the name of T1's sample: its sha256 is not the recorded one.
    sample = inputs.paths.ranker_sample_file("full", "T1", RATIO)
    pl.read_parquet(sample).head(100).write_parquet(sample)
    with pytest.raises(ValueError, match="sha256"):
        run_t_eval_trial(context, inputs, TrialKey(STAGE_FORMAL, 42, "T2"), PERMIT)
    assert not inputs.paths.deepfm_trial_dir("m9_t", CONFIG_ID, 42, "T2").exists()


def test_trainings_outside_the_evaluation_are_refused_before_anything_is_read(
    tmp_path: Path,
) -> None:
    inputs = build_t_eval_workspace(tmp_path)
    context = context_of(inputs)
    with pytest.raises(ValueError, match="only the configured rehearsal fold V4"):
        run_t_eval_trial(context, inputs, TrialKey(STAGE_REHEARSAL, 42, "T1"), None)
    with pytest.raises(ValueError, match="predicts only the folds"):
        run_t_eval_trial(context, inputs, TrialKey(STAGE_FORMAL, 42, "V4"), PERMIT)
    with pytest.raises(ValueError, match="got seed 7"):
        run_t_eval_trial(context, inputs, TrialKey(STAGE_FORMAL, 7, "T1"), PERMIT)
    assert not inputs.paths.deepfm_dir.exists()
    # The V-phase entry point still refuses every T fold.
    chosen = chosen_trial_inputs(inputs, context.config, 42)
    with pytest.raises(ValueError, match="T folds are evaluated in M8"):
        run_trial(chosen, TrialSpec("v_phase", "T1", 42, 1))


def test_a_model_never_depends_on_its_target_week_or_later_weeks(tmp_path: Path) -> None:
    # Time cut-off: the model that predicts T1 trains on V1 V2 V3 V4. "Poison" in a copy of
    # the workspace: other sample files for T1, T2 and T3 and other labels for T1 ... T4.
    # The T1 model's Top-K (with its scores) and its input transform must not change.
    clean = build_t_eval_workspace(tmp_path / "clean")
    poisoned = copy_workspace(clean, tmp_path / "poisoned")
    paths = poisoned.paths
    for fold in ("T1", "T2", "T3"):
        sample = paths.ranker_sample_file("full", fold, RATIO)
        pl.read_parquet(sample).with_columns(1 - pl.col("label")).write_parquet(sample)
    for fold in ("T1", "T2", "T3", "T4"):
        labels = pl.read_parquet(paths.labels_file(fold))
        labels.with_columns(pl.col("article_idx").reverse()).write_parquet(paths.labels_file(fold))
    key = TrialKey(STAGE_FORMAL, 42, "T1")
    for inputs in (clean, poisoned):
        run_t_eval_trial(context_of(inputs), inputs, key, PERMIT)
    one = clean.paths.deepfm_trial_dir("m9_t", CONFIG_ID, 42, "T1")
    two = paths.deepfm_trial_dir("m9_t", CONFIG_ID, 42, "T1")
    assert pl.read_parquet(one / PREDICTIONS_FILE).equals(pl.read_parquet(two / PREDICTIONS_FILE))
    assert (one / "transform.json").read_bytes() == (two / "transform.json").read_bytes()
    # The poison is real: the evaluation of T1 itself did change.
    assert not pl.read_parquet(one / PER_QUERY_FILE).equals(pl.read_parquet(two / PER_QUERY_FILE))
