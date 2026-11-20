"""Tests for hmrec.kaggle_deepfm.report and .context (the DeepFM Kaggle submission).

The submission file, the rehearsal's three checks, the gate before the real run
and the record of the one submission with its pre-recorded reading.
"""

import dataclasses
from pathlib import Path

import polars as pl
import pytest
from deepfm_example import REPO
from kaggle_deepfm_example import (
    N_FOLD_CUSTOMERS,
    N_REHEARSAL_CUSTOMERS,
    QUERY_ID_BASE,
    SCRIPT,
    build_kaggle_deepfm_workspace,
    kaggle_context,
)
from t_eval_example import COMMIT, OTHER_COMMIT, copy_workspace

from hmrec.config import (
    load_kaggle_deepfm_config,
    load_t_eval_config,
    load_yaml,
    parse_kaggle_deepfm_config,
)
from hmrec.csv_records import read_rows
from hmrec.deepfm.trial import TrialInputs
from hmrec.kaggle.context import RUN_COLUMNS
from hmrec.kaggle.plan import RUN_REHEARSAL, RUN_SUBMISSION
from hmrec.kaggle.rehearsal import TopComparison
from hmrec.kaggle.report import FILES_COLUMNS, SubmissionResult
from hmrec.kaggle.submission import FileCheck
from hmrec.kaggle_deepfm.context import (
    DeepFMKaggleContext,
    check_settings,
    finish_process,
    training_folds,
)
from hmrec.kaggle_deepfm.report import (
    DEEPFM_SUBMISSION_COLUMNS,
    DIFFERENT,
    REHEARSAL_COLUMNS,
    SAME,
    RehearsalOutcome,
    format_checks_passed,
    read_all_tops,
    reading_of,
    record_files,
    record_rehearsal,
    record_submission,
    rehearsal_outcome,
    relative_difference,
    require_passed_rehearsal,
    run_shards,
    stage2_train_loss,
    write_and_check,
)
from hmrec.kaggle_deepfm.score import score_run
from hmrec.kaggle_deepfm.train import chosen_inputs, train_run_model
from hmrec.memory import ProcessPeakMemory

CONFIG_FILE = REPO / "configs" / "kaggle_deepfm.yaml"
PASSED_CHECK = FileCheck(35, 100, "c" * 64, 40, "g" * 64, {"header": True, "12 ids": True})


@pytest.fixture(scope="module")
def rehearsed(tmp_path_factory: pytest.TempPathFactory) -> TrialInputs:
    """A workspace with the rehearsal trained and scored (built once)."""
    inputs = build_kaggle_deepfm_workspace(tmp_path_factory.mktemp("report") / "workspace")
    context = kaggle_context(inputs, RUN_REHEARSAL)
    chosen = chosen_inputs(context, inputs)
    train_run_model(context, chosen, SCRIPT)
    score_run(context, chosen, run_shards(context))
    return inputs


def _id_tables(n_customers: int) -> tuple[pl.DataFrame, pl.DataFrame, pl.Series]:
    """Customer ids, article ids and a sample order (reversed) for a tiny run."""
    customer_ids = pl.DataFrame(
        {
            "customer_idx": pl.Series(range(n_customers + 5), dtype=pl.Int32),
            "customer_id": [f"{k:064d}" for k in range(n_customers + 5)],
        }
    )
    article_ids = pl.DataFrame(
        {
            "article_idx": pl.Series(range(12), dtype=pl.Int32),
            "article_id": [f"0{k:09d}" for k in range(12)],
        }
    )
    # The sample file holds ALL customers, in another order than the run's.
    return customer_ids, article_ids, customer_ids["customer_id"].reverse()


# ---------------------------------------------------------------- configs and context


def test_the_recorded_config_fits_the_configuration_chosen_on_v() -> None:
    settings = load_kaggle_deepfm_config(CONFIG_FILE)
    assert (settings.seed, settings.config_id, settings.reference_area) == (
        42,
        "f8811fb59b79",
        "m9_t",
    )
    assert (settings.lightgbm_private_score, settings.lightgbm_public_score) == (0.036, 0.03548)
    assert settings.same_below_relative_difference == 0.01
    chosen = load_t_eval_config(REPO / "configs" / "m9_t_eval.yaml")
    check_settings(settings, chosen)
    with pytest.raises(ValueError, match="has id f8811fb59b79"):
        check_settings(dataclasses.replace(settings, config_id="0" * 12), chosen)
    with pytest.raises(ValueError, match="not a seed of the stage-2 evaluation"):
        check_settings(dataclasses.replace(settings, seed=7), chosen)


def test_config_errors_are_refused() -> None:
    raw = load_yaml(CONFIG_FILE)
    with pytest.raises(ValueError, match="unknown keys"):
        parse_kaggle_deepfm_config({**raw, "seeds": [42, 43]})
    with pytest.raises(ValueError, match="same_below_relative_difference"):
        parse_kaggle_deepfm_config(
            {**raw, "reading": {**raw["reading"], "same_below_relative_difference": 1.5}}
        )
    with pytest.raises(ValueError, match="seed"):
        parse_kaggle_deepfm_config({**raw, "seed": 0})


def test_training_folds_of_the_two_runs(rehearsed: TrialInputs) -> None:
    rehearsal = training_folds(kaggle_context(rehearsed, RUN_REHEARSAL), 4)
    submission = training_folds(kaggle_context(rehearsed, RUN_SUBMISSION), 4)
    assert [fold.name for fold in rehearsal] == ["V4", "T1", "T2", "T3"]
    assert [fold.name for fold in submission] == ["T1", "T2", "T3", "T4"]


def test_finish_writes_its_own_run_table_then_checks_the_memory_limit(
    rehearsed: TrialInputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = copy_workspace(rehearsed, tmp_path / "copy")
    context = kaggle_context(inputs, RUN_REHEARSAL)
    peaks = iter([ProcessPeakMemory(2 * 1024**3, 0), ProcessPeakMemory(13 * 1024**3, 0)])
    monkeypatch.setattr(
        "hmrec.kaggle_deepfm.context.log_process_peak_memory", lambda *_: next(peaks)
    )
    finish_process(context, "train", "done", 10, 0.0, "detail")
    with pytest.raises(MemoryError, match="above the limit"):
        finish_process(context, "score", "done", 20, 0.0, "detail")
    rows = read_rows(inputs.paths.kaggle_deepfm_runs_file, RUN_COLUMNS)
    assert [(row["run"], row["step"], row["n_rows"]) for row in rows] == [
        ("rehearsal", "train", "10"),
        ("rehearsal", "score", "20"),
    ]
    # The run table of the LightGBM submission is not written.
    assert not inputs.paths.kaggle_runs_file.exists()


# ---------------------------------------------------------------- the submission file


def test_the_file_follows_the_sample_order_and_passes_the_nine_checks(
    rehearsed: TrialInputs, tmp_path: Path
) -> None:
    context = kaggle_context(rehearsed, RUN_REHEARSAL)
    customer_ids, article_ids, sample_ids = _id_tables(N_REHEARSAL_CUSTOMERS)
    queries = pl.read_parquet(rehearsed.paths.kaggle_queries_file(RUN_REHEARSAL))
    files = (tmp_path / "submission.csv", tmp_path / "submission.csv.gz")
    top = read_all_tops(context)
    check = write_and_check(top, queries, customer_ids, article_ids, sample_ids, files, 12)
    assert check.passed and len(check.checks) == 9 and check.n_rows == N_REHEARSAL_CUSTOMERS
    lines = files[0].read_text().splitlines()
    assert lines[0] == "customer_id,prediction"
    # The run's 35 customers in the sample's (reversed) order: customer 34 first.
    assert lines[1].split(",")[0] == f"{34:064d}"
    # Customer 0's row: its 12 article ids, best first.
    best_first = top.filter(pl.col("query_id") == QUERY_ID_BASE).sort("rank")["article_idx"]
    assert lines[-1] == f"{0:064d}," + " ".join(f"0{k:09d}" for k in best_first.to_list())
    # A second call rewrites nothing and checks the same bytes again.
    again = write_and_check(top, queries, customer_ids, article_ids, sample_ids, files, 12)
    assert (again.csv_sha256, again.gzip_sha256) == (check.csv_sha256, check.gzip_sha256)
    # A Top-K that lacks a customer is refused before anything is written.
    other = (tmp_path / "other.csv", tmp_path / "other.csv.gz")
    with pytest.raises(ValueError, match="queries"):
        write_and_check(
            top.filter(pl.col("query_id") != QUERY_ID_BASE),
            queries,
            customer_ids,
            article_ids,
            sample_ids,
            other,
            12,
        )
    assert not other[0].exists()


def test_written_files_are_recorded_per_run_and_commit(
    rehearsed: TrialInputs, tmp_path: Path
) -> None:
    inputs = copy_workspace(rehearsed, tmp_path / "copy")
    context = kaggle_context(inputs, RUN_REHEARSAL)
    with pytest.raises(ValueError, match="no submission file of the rehearsal run"):
        format_checks_passed(context)
    failed = dataclasses.replace(PASSED_CHECK, checks={"header": True, "12 ids": False})
    record_files(context, failed)
    assert not format_checks_passed(context)
    record_files(context, PASSED_CHECK)
    assert format_checks_passed(context)  # the LAST recorded file counts
    rows = read_rows(inputs.paths.kaggle_deepfm_files_file, FILES_COLUMNS)
    assert rows[1]["csv_file"] == "artifacts/kaggle/deepfm/rehearsal/submission.csv"
    assert rows[1]["gzip_file"] == "artifacts/kaggle/deepfm/rehearsal/submission.csv.gz"
    assert [(row["failed_checks"], row["passed"]) for row in rows] == [
        ("12 ids", "False"),
        ("", "True"),
    ]
    with pytest.raises(ValueError, match="at this commit"):
        format_checks_passed(kaggle_context(inputs, RUN_REHEARSAL, commit=OTHER_COMMIT))
    assert not inputs.paths.kaggle_files_file.exists()


# ---------------------------------------------------------------- the rehearsal's checks


def _outcome(**changes: object) -> RehearsalOutcome:
    values = {
        "n_customers": 35,
        "n_shards": 2,
        "new_loss": 3.5,
        "stage2_loss": 3.5,
        "comparison": TopComparison(1000, 999),
        "min_identical_share": 0.999,
        "format_checks_passed": True,
    }
    return RehearsalOutcome(**{**values, **changes})  # type: ignore[arg-type]  # test helper


def test_all_three_checks_are_needed_to_pass() -> None:
    assert _outcome().passed  # 999 of 1000 identical is exactly the required share
    assert not _outcome(comparison=TopComparison(1000, 998)).passed
    # The losses must be the same float64: the next float is not equal.
    assert not _outcome(new_loss=3.5000000000000004).passed
    assert not _outcome(format_checks_passed=False).passed


def test_the_rehearsal_of_the_workspace_passes_and_is_recorded(
    rehearsed: TrialInputs, tmp_path: Path
) -> None:
    inputs = copy_workspace(rehearsed, tmp_path / "copy")
    context = kaggle_context(inputs, RUN_REHEARSAL)
    record_files(context, PASSED_CHECK)
    outcome = rehearsal_outcome(context)
    assert outcome.passed and outcome.loss_equal
    assert (outcome.n_customers, outcome.n_shards) == (N_REHEARSAL_CUSTOMERS, 2)
    assert (outcome.comparison.n_queries, outcome.comparison.n_identical) == (
        N_FOLD_CUSTOMERS,
        N_FOLD_CUSTOMERS,
    )
    record_rehearsal(context, outcome)
    rows = read_rows(inputs.paths.kaggle_deepfm_rehearsal_file, REHEARSAL_COLUMNS)
    assert len(rows) == 1 and rows[0]["passed"] == "True"
    assert float(rows[0]["new_train_loss"]) == float(rows[0]["stage2_train_loss"])
    assert (rows[0]["n_identical_top"], rows[0]["identical_top_share"]) == ("30", "1.0")
    # Only a rehearsal is compared with stage 2.
    with pytest.raises(ValueError, match="only a rehearsal"):
        rehearsal_outcome(kaggle_context(inputs, RUN_SUBMISSION))
    # Without the stage-2 record of the seed there is nothing to compare with.
    other_seed = dataclasses.replace(context.settings, seed=43)
    with pytest.raises(ValueError, match="expected exactly one"):
        stage2_train_loss(dataclasses.replace(context, settings=other_seed))


def test_the_real_run_needs_a_passed_rehearsal_of_the_same_code(
    rehearsed: TrialInputs, tmp_path: Path
) -> None:
    inputs = copy_workspace(rehearsed, tmp_path / "copy")
    context = kaggle_context(inputs, RUN_REHEARSAL)
    with pytest.raises(RuntimeError, match="no rehearsal is recorded"):
        require_passed_rehearsal(context, lambda commit: False)
    record_rehearsal(context, _outcome(comparison=TopComparison(1000, 998)))
    with pytest.raises(RuntimeError, match="did\\s+not pass"):
        require_passed_rehearsal(context, lambda commit: False)
    record_rehearsal(context, _outcome())
    assert require_passed_rehearsal(context, lambda commit: False) == COMMIT
    with pytest.raises(RuntimeError, match="the code changed since"):
        require_passed_rehearsal(context, lambda commit: True)
    # A rehearsal from a dirty tree does not count.
    dirty = kaggle_context(inputs, RUN_REHEARSAL, dirty=True)
    record_rehearsal(dirty, _outcome())
    with pytest.raises(RuntimeError, match="clean working tree"):
        require_passed_rehearsal(context, lambda commit: False)


# ---------------------------------------------------------------- the submission's record


def test_the_reading_fixed_before_the_run() -> None:
    # Hand-computed against the LightGBM Private score 0.03600 with the 1% threshold:
    # 0.03630 -> +0.83% (same); 0.03570 -> -0.83% (same); 0.03636 -> +1.0% (different);
    # 0.03500 -> -2.78% (different).
    assert relative_difference(0.0363, 0.036) == pytest.approx(0.0083333, abs=1e-6)
    assert relative_difference(0.035, 0.036) == pytest.approx(-0.0277778, abs=1e-6)
    assert reading_of(relative_difference(0.0363, 0.036), 0.01) == SAME
    assert reading_of(relative_difference(0.0357, 0.036), 0.01) == SAME
    assert reading_of(relative_difference(0.03637, 0.036), 0.01) == DIFFERENT
    assert reading_of(relative_difference(0.035, 0.036), 0.01) == DIFFERENT
    # The threshold itself is "different": the rule is "below 1%".
    assert reading_of(0.01, 0.01) == DIFFERENT and reading_of(-0.01, 0.01) == DIFFERENT
    assert SAME == "cannot be told apart on this week"
    with pytest.raises(ValueError, match="must be > 0"):
        relative_difference(0.03, 0.0)


def _result(scores: tuple[float, ...] | None, private: float = 0.0363) -> SubmissionResult:
    return SubmissionResult(
        "777", "2026-11-20 10:00:00", "complete", 0.0355, private, "m", 3, scores
    )


def test_the_one_submission_is_recorded_once_with_its_reading(
    rehearsed: TrialInputs, tmp_path: Path
) -> None:
    inputs = copy_workspace(rehearsed, tmp_path / "copy")
    context: DeepFMKaggleContext = kaggle_context(inputs, RUN_SUBMISSION)
    with pytest.raises(ValueError, match="no checked submission file"):
        record_submission(context, _result(None))
    record_files(context, PASSED_CHECK)
    row = record_submission(context, _result((0.03792, 0.03741, 0.03623, 0.03563)))
    assert (row["public_score"], row["private_score"]) == ("0.0355", "0.0363")
    # Two leaderboard scores are above 0.0363: it would stand at place 3.
    assert (row["private_position"], row["leaderboard_teams"]) == ("3", "4")
    assert row["position_label"] == "late submission, not an official rank"
    assert (row["file_sha256"], row["run_git_commit"]) == ("g" * 64, COMMIT)
    assert row["file"] == "artifacts/kaggle/deepfm/submission/submission.csv.gz"
    assert float(row["private_relative_difference"]) == pytest.approx(0.0083333, abs=1e-6)
    assert float(row["public_relative_difference"]) == pytest.approx(0.0005637, abs=1e-6)
    assert (row["lightgbm_private_score"], row["reading"]) == ("0.036", SAME)
    rows = read_rows(inputs.paths.kaggle_deepfm_submission_results_file, DEEPFM_SUBMISSION_COLUMNS)
    assert rows == [row]
    with pytest.raises(ValueError, match="one submission is made"):
        record_submission(context, _result(None))
    # The record of the LightGBM submission is not written.
    assert not inputs.paths.kaggle_submission_results_file.exists()


def test_without_a_leaderboard_no_position_is_recorded(
    rehearsed: TrialInputs, tmp_path: Path
) -> None:
    inputs = copy_workspace(rehearsed, tmp_path / "copy")
    context = kaggle_context(inputs, RUN_SUBMISSION)
    record_files(context, PASSED_CHECK)
    row = record_submission(context, _result(None, private=0.035))
    assert (row["private_position"], row["leaderboard_teams"], row["position_label"]) == (
        "",
        "",
        "",
    )
    assert row["reading"] == DIFFERENT
