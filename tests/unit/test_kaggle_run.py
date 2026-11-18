"""Tests for the small pieces of a Kaggle run: config, paths, context, prepare, report, driver.

Tiny data throughout (a handful of customers under tmp_path); nothing of the
project's data is read except the real config files.
"""

import dataclasses
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from hmrec.config import (
    load_base_config,
    load_kaggle_config,
    load_preprocess_config,
    load_yaml,
    parse_kaggle_config,
)
from hmrec.csv_records import read_rows
from hmrec.data.folds import load_fold_manifest
from hmrec.kaggle import context as context_module
from hmrec.kaggle.context import (
    ALL_SHARDS,
    DONE,
    RUN_COLUMNS,
    KaggleContext,
    finish_process,
    load_context,
    require_clean_tree,
)
from hmrec.kaggle.plan import RUN_REHEARSAL, RUN_SUBMISSION, build_queries, write_queries
from hmrec.kaggle.prepare import (
    check_free_disk,
    check_sample_customers,
    required_free_gb,
    run_customers,
)
from hmrec.kaggle.report import (
    CANDIDATES_COLUMNS,
    FILES_COLUMNS,
    POSITION_LABEL,
    SUBMISSION_COLUMNS,
    SubmissionResult,
    channel_rows,
    format_checks_passed,
    position_among,
    read_all_tops,
    read_statistics,
    record_candidates,
    record_files,
    record_submission,
    run_shards,
)
from hmrec.kaggle.run import run_steps, steps_after_recall, steps_before_shards
from hmrec.kaggle.submission import FileCheck
from hmrec.locks import ALL_CHANNELS
from hmrec.memory import ProcessPeakMemory
from hmrec.paths import build_paths, get_paths
from hmrec.run_records import RunStamp

REPO = Path(__file__).resolve().parents[2]
BASE = load_base_config(REPO / "configs" / "base.yaml")
CONFIG_FILE = REPO / "configs" / "kaggle_submission.yaml"
KAGGLE = load_kaggle_config(CONFIG_FILE)
MANIFEST = load_fold_manifest(REPO / "configs" / "fold_manifest.yaml")
PREPROCESS = load_preprocess_config(REPO / "configs" / "preprocess.yaml")
STAMP = RunStamp("abc123", False)


def _context(tmp_path: Path, run: str = RUN_REHEARSAL, shard_customers: int = 2) -> KaggleContext:
    context = load_context(run, get_paths(), STAMP)
    return dataclasses.replace(
        context,
        paths=build_paths(BASE, tmp_path),
        config=dataclasses.replace(KAGGLE, shard_customers=shard_customers),
    )


# ---------------------------------------------------------------- config and paths


def test_real_config_describes_the_week_after_the_data() -> None:
    assert KAGGLE.competition == "h-and-m-personalized-fashion-recommendations"
    assert (KAGGLE.week_name, KAGGLE.week_anchor, KAGGLE.week_label_end) == (
        "S",
        date(2020, 9, 23),
        date(2020, 9, 29),
    )
    assert (KAGGLE.shard_customers, KAGGLE.predictions_per_customer) == (70_000, 12)
    assert KAGGLE.rehearsal.target_fold == "T4" and KAGGLE.rehearsal.extra_customers == 50_000
    assert KAGGLE.rehearsal.min_identical_top12_share == 0.999
    assert KAGGLE.disk_safety_factor == 1.5


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        (None, "shard_customers", 0, "shard_customers"),
        (None, "query_id_base", -1, "query_id_base"),
        ("rehearsal", "min_identical_top12_share", 1.5, "min_identical_top12_share"),
        ("rehearsal", "extra_customers", 0, "extra_customers"),
        ("disk", "safety_factor", 0.5, "safety_factor"),
        ("submission_week", "label_end", date(2020, 9, 1), "before the anchor"),
    ],
)
def test_config_values_are_validated(
    section: str | None, key: str, value: object, message: str
) -> None:
    raw = load_yaml(CONFIG_FILE)
    if section is None:
        raw[key] = value
    else:
        raw[section] = {**raw[section], key: value}
    with pytest.raises(ValueError, match=message):
        parse_kaggle_config(raw)


def test_unknown_config_key_is_refused() -> None:
    with pytest.raises(ValueError, match="tune"):
        parse_kaggle_config({**load_yaml(CONFIG_FILE), "tune": True})


def test_every_new_file_is_under_artifacts_kaggle_or_a_kaggle_result_table() -> None:
    paths = get_paths()
    files = [
        paths.kaggle_queries_file("submission"),
        paths.kaggle_candidates_file("submission", 3, "r4_itemcf"),
        paths.kaggle_r1_bucket_scores_file("submission"),
        paths.kaggle_vectors_dir("submission", 3, "wals"),
        paths.kaggle_twotower_checkpoint_file("submission", "T4"),
        paths.kaggle_item_table_file("submission"),
        paths.kaggle_type_share_file("submission"),
        paths.kaggle_features_dir("submission", 3),
        paths.kaggle_shard_stats_file("submission", 3),
        paths.kaggle_top_file("submission", 3),
        paths.kaggle_sample_file("submission", "T4", 40),
        paths.kaggle_dataset_dir("submission"),
        paths.kaggle_model_file("submission"),
        paths.kaggle_log_file("submission", "driver"),
        paths.kaggle_submission_file("rehearsal"),
        paths.kaggle_submission_gzip_file("rehearsal"),
    ]
    assert all(paths.kaggle_dir in file.parents for file in files)
    assert paths.kaggle_shard_dir("rehearsal", 7).name == "007"
    # The real run's file is exactly artifacts/kaggle/submission.csv (+ .gz).
    assert paths.kaggle_submission_file("submission") == paths.kaggle_dir / "submission.csv"
    assert paths.kaggle_submission_gzip_file("submission").name == "submission.csv.gz"
    tables = [
        paths.kaggle_runs_file,
        paths.kaggle_label_access_file,
        paths.kaggle_twotower_results_file,
        paths.kaggle_candidates_results_file,
        paths.kaggle_rehearsal_file,
        paths.kaggle_files_file,
        paths.kaggle_submission_results_file,
    ]
    assert all(table.parent == paths.results_root for table in tables)
    assert all(table.name.startswith("kaggle_") for table in tables)


# ---------------------------------------------------------------- context


def test_context_knows_the_predicted_week_of_each_run() -> None:
    paths = get_paths()
    assert load_context(RUN_SUBMISSION, paths, STAMP).target.anchor == date(2020, 9, 23)
    assert load_context(RUN_REHEARSAL, paths, STAMP).target == MANIFEST.get("T4")
    with pytest.raises(ValueError, match="run must be one of"):
        load_context("formal", paths, STAMP)


def test_a_dirty_working_tree_is_refused() -> None:
    require_clean_tree(STAMP)
    with pytest.raises(RuntimeError, match="starts only from committed code"):
        require_clean_tree(RunStamp("abc123", True))


def test_finish_writes_the_run_row_then_checks_the_memory_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path)
    finish_process(context, "recall_r4_itemcf", ALL_SHARDS, DONE, 123, 0.0, "2 shards")
    rows = read_rows(context.paths.kaggle_runs_file, RUN_COLUMNS)
    assert [(row["run"], row["step"], row["shard"], row["n_rows"]) for row in rows] == [
        ("rehearsal", "recall_r4_itemcf", "all", "123")
    ]
    assert rows[0]["git_commit"] == "abc123" and float(rows[0]["process_max_rss_gb"]) > 0
    # A true peak above the limit stops the run, AFTER its row is on record.
    too_much = ProcessPeakMemory(13 * 1024**3, 0)
    monkeypatch.setattr(context_module, "log_process_peak_memory", lambda *args: too_much)
    with pytest.raises(MemoryError, match=r"13\.00 GB is above the limit of 12\.00 GB"):
        finish_process(context, "features", "3", DONE, 1, 0.0, "")
    assert len(read_rows(context.paths.kaggle_runs_file, RUN_COLUMNS)) == 2


# ---------------------------------------------------------------- prepare


def test_disk_requirement_is_the_estimate_times_the_safety_factor(tmp_path: Path) -> None:
    assert required_free_gb(_context(tmp_path, RUN_REHEARSAL)) == pytest.approx(4.0 * 1.5)
    assert required_free_gb(_context(tmp_path, RUN_SUBMISSION)) == pytest.approx(40.0 * 1.5)
    tmp_path.joinpath("artifacts").mkdir()
    assert check_free_disk(_context(tmp_path)) > 0
    huge = dataclasses.replace(KAGGLE, rehearsal_disk_gb=1e9)
    with pytest.raises(OSError, match="the run needs"):
        check_free_disk(dataclasses.replace(_context(tmp_path), config=huge))


def test_sample_file_must_hold_exactly_the_customer_table() -> None:
    table = pl.DataFrame(
        {"customer_idx": [0, 1, 2], "customer_id": ["cA", "cB", "cC"]},
        schema={"customer_idx": pl.Int32, "customer_id": pl.String},
    )
    check_sample_customers(table, pl.Series("customer_id", ["cC", "cA", "cB"]))
    with pytest.raises(ValueError, match="repeats"):
        check_sample_customers(table, pl.Series("customer_id", ["cC", "cA", "cA"]))
    with pytest.raises(ValueError, match="has 2 customers"):
        check_sample_customers(table, pl.Series("customer_id", ["cC", "cA"]))
    with pytest.raises(ValueError, match="1 unknown"):
        check_sample_customers(table, pl.Series("customer_id", ["cC", "cA", "cX"]))


def test_customers_of_each_run(tmp_path: Path) -> None:
    everyone = np.arange(20, dtype=np.int32)[::-1].copy()
    submission = _context(tmp_path, RUN_SUBMISSION)
    assert run_customers(submission, everyone).tolist() == list(range(20))
    # Rehearsal: the fold's query customers (read from the query list, not from labels)
    # plus the configured number of others.
    rehearsal = _context(tmp_path, RUN_REHEARSAL)
    fold_queries = pl.DataFrame(
        {"query_id": [0, 1], "customer_idx": [4, 9]},
        schema={"query_id": pl.Int32, "customer_idx": pl.Int32},
    )
    rehearsal.paths.queries_file("T4").parent.mkdir(parents=True)
    fold_queries.write_parquet(rehearsal.paths.queries_file("T4"))
    few = dataclasses.replace(KAGGLE.rehearsal, extra_customers=5)
    rehearsal = dataclasses.replace(rehearsal, config=dataclasses.replace(KAGGLE, rehearsal=few))
    customers = run_customers(rehearsal, everyone)
    assert customers.size == 7 and {4, 9} <= set(customers.tolist())
    assert np.array_equal(customers, run_customers(rehearsal, everyone))


# ---------------------------------------------------------------- report


def _finished_run(tmp_path: Path) -> KaggleContext:
    """A run of 3 customers in 2 shards with candidates, query statistics and Top-K files."""
    context = _context(tmp_path, RUN_REHEARSAL, shard_customers=2)
    paths, run = context.paths, context.run
    queries = build_queries(np.array([0, 1, 2], dtype=np.int32), context.target, 500)
    write_queries(queries, paths.kaggle_queries_file(run))
    candidates_per_shard = {0: [500, 500, 501], 1: [502]}
    for shard, query_ids in candidates_per_shard.items():
        for channel in ALL_CHANNELS:
            file = paths.kaggle_candidates_file(run, shard, channel)
            file.parent.mkdir(parents=True, exist_ok=True)
            pl.DataFrame({"query_id": query_ids}, schema={"query_id": pl.Int32}).write_parquet(file)
        stats = pl.DataFrame(
            {
                "query_id": sorted(set(query_ids)),
                "customer_idx": [q - 500 for q in sorted(set(query_ids))],
                "n_candidates": [10 * (q - 499) for q in sorted(set(query_ids))],
                "has_history": [q != 502 for q in sorted(set(query_ids))],
            },
            schema={
                "query_id": pl.Int32,
                "customer_idx": pl.Int32,
                "n_candidates": pl.Int32,
                "has_history": pl.Boolean,
            },
        )
        stats.write_parquet(paths.kaggle_shard_stats_file(run, shard))
        top = pl.DataFrame(
            {"query_id": sorted(set(query_ids)), "rank": 1, "article_idx": 7, "score": 0.5},
            schema={
                "query_id": pl.Int32,
                "rank": pl.Int32,
                "article_idx": pl.Int32,
                "score": pl.Float64,
            },
        )
        top.write_parquet(paths.kaggle_top_file(run, shard))
    return context


def test_candidate_counts_are_recorded_per_channel_and_per_customer(tmp_path: Path) -> None:
    context = _finished_run(tmp_path)
    assert run_shards(context) == 2
    assert channel_rows(context, "r4_itemcf") == 4  # 3 rows in shard 0, 1 in shard 1
    stats = read_statistics(context)
    # n_candidates 10, 20, 30 for the three customers; customer 2 has no history.
    assert (stats.n_customers, stats.n_candidate_rows) == (3, 60)
    assert stats.per_customer_mean == pytest.approx(20.0)
    assert stats.no_history_share == pytest.approx(1 / 3)
    record_candidates(context)
    rows = read_rows(context.paths.kaggle_candidates_results_file, CANDIDATES_COLUMNS)
    assert len(rows) == 1
    assert (rows[0]["run"], rows[0]["week"], rows[0]["anchor"]) == ("rehearsal", "T4", "2020-09-16")
    assert (rows[0]["n_customers"], rows[0]["n_shards"], rows[0]["candidate_rows"]) == (
        "3",
        "2",
        "60",
    )
    assert all(rows[0][f"{channel}_rows"] == "4" for channel in ALL_CHANNELS)


def test_tops_of_all_shards_are_needed(tmp_path: Path) -> None:
    context = _finished_run(tmp_path)
    assert read_all_tops(context)["query_id"].to_list() == [500, 501, 502]
    context.paths.kaggle_top_file(context.run, 1).unlink()
    with pytest.raises(FileNotFoundError, match="shard 1 is not scored yet"):
        read_all_tops(context)


def test_written_file_is_recorded_with_the_outcome_of_its_checks(tmp_path: Path) -> None:
    context = _finished_run(tmp_path)
    with pytest.raises(ValueError, match="no submission file of the rehearsal run"):
        format_checks_passed(context)
    failed = FileCheck(3, 100, "c" * 64, 40, "g" * 64, {"header": True, "12 ids": False})
    record_files(context, failed)
    assert not format_checks_passed(context)
    record_files(context, dataclasses.replace(failed, checks={"header": True, "12 ids": True}))
    assert format_checks_passed(context)  # the LAST recorded file counts
    rows = read_rows(context.paths.kaggle_files_file, FILES_COLUMNS)
    assert [(row["failed_checks"], row["passed"]) for row in rows] == [
        ("12 ids", "False"),
        ("", "True"),
    ]
    assert rows[0]["csv_file"] == "artifacts/kaggle/rehearsal/submission.csv"
    assert (rows[0]["csv_sha256"], rows[0]["n_rows"], rows[0]["n_checks"]) == ("c" * 64, "3", "2")
    # A file recorded at another commit says nothing about this code.
    other = dataclasses.replace(context, stamp=RunStamp("other", False))
    with pytest.raises(ValueError, match="at this commit"):
        format_checks_passed(other)


def _result(scores: tuple[float, ...] | None) -> SubmissionResult:
    return SubmissionResult("777", "2026-11-19 12:00:00", "complete", 0.02, 0.021, "msg", 2, scores)


def test_position_is_one_plus_the_scores_above() -> None:
    scores = [0.05, 0.03, 0.03, 0.01]
    assert position_among(scores, 0.06) == 1
    assert position_among(scores, 0.04) == 2
    assert position_among(scores, 0.03) == 2  # a tie shares the place
    assert position_among(scores, 0.02) == 4
    assert position_among(scores, 0.0) == 5


def test_the_one_submission_is_recorded_once_with_the_checked_file(tmp_path: Path) -> None:
    context = _finished_run(tmp_path)
    with pytest.raises(ValueError, match="no checked submission file"):
        record_submission(context, _result(None))
    record_files(context, FileCheck(3, 100, "c" * 64, 40, "g" * 64, {"header": True}))
    row = record_submission(context, _result((0.05, 0.03, 0.01)))
    rows = read_rows(context.paths.kaggle_submission_results_file, SUBMISSION_COLUMNS)
    assert rows == [row]
    assert (row["public_score"], row["private_score"], row["status"]) == (
        "0.02",
        "0.021",
        "complete",
    )
    # The row names the gzip file that passed the checks, and the commit that wrote it.
    assert (row["file_sha256"], row["run_git_commit"]) == ("g" * 64, "abc123")
    assert row["file"] == "artifacts/kaggle/rehearsal/submission.csv.gz"
    # 0.021 would stand behind 0.05 and 0.03: place 3 of 3 teams, and it is labelled.
    assert (row["private_position"], row["leaderboard_teams"]) == ("3", "3")
    assert row["position_label"] == POSITION_LABEL == "late submission, not an official rank"
    assert row["submissions_before"] == "2"
    with pytest.raises(ValueError, match="one submission is made"):
        record_submission(context, _result(None))


def test_without_a_leaderboard_no_position_is_recorded(tmp_path: Path) -> None:
    context = _finished_run(tmp_path)
    record_files(context, FileCheck(3, 100, "c" * 64, 40, "g" * 64, {"header": True}))
    row = record_submission(context, _result(None))
    assert (row["private_position"], row["leaderboard_teams"], row["position_label"]) == (
        "",
        "",
        "",
    )


# ---------------------------------------------------------------- driver


def test_steps_of_the_two_runs() -> None:
    before = steps_before_shards(RUN_REHEARSAL)
    assert before[0] == ("kaggle_prepare.py", [])
    assert [arguments[1] for _, arguments in before[1:7]] == list(ALL_CHANNELS)
    assert before[-1] == ("kaggle_features.py", ["--prepare"])
    rehearsal = steps_after_recall(RUN_REHEARSAL, 2)
    assert rehearsal == [
        ("kaggle_features.py", ["--shard", "0"]),
        ("kaggle_features.py", ["--shard", "1"]),
        ("kaggle_ranker.py", ["--step", "dataset"]),
        ("kaggle_ranker.py", ["--step", "train"]),
        ("kaggle_ranker.py", ["--step", "score"]),
        ("kaggle_submission.py", []),
        ("kaggle_ranker.py", ["--step", "check"]),
    ]
    submission = steps_after_recall(RUN_SUBMISSION, 1)
    # Only the real run samples a fold (T4); only the rehearsal is compared with M8.
    assert ("kaggle_ranker.py", ["--step", "sample"]) in submission
    assert ("kaggle_ranker.py", ["--step", "sample"]) not in rehearsal
    assert ("kaggle_ranker.py", ["--step", "check"]) not in submission
    with pytest.raises(ValueError, match="at least one shard"):
        steps_after_recall(RUN_SUBMISSION, 0)
    # Every script the driver starts exists.
    scripts = {script for script, _ in [*before, *rehearsal, *submission]}
    assert all((REPO / "scripts" / script).is_file() for script in scripts)


def test_driver_runs_the_steps_in_order_and_stops_at_the_first_failure(tmp_path: Path) -> None:
    log = tmp_path / "log.txt"
    (tmp_path / "ok.py").write_text(
        "import sys\nopen(sys.argv[-1], 'a').write(' '.join(sys.argv[1:-1]) + '\\n')\n"
    )
    (tmp_path / "bad.py").write_text("raise SystemExit(3)\n")
    run_steps(tmp_path, "rehearsal", [("ok.py", ["a", str(log)]), ("ok.py", ["b", str(log)])])
    assert log.read_text().splitlines() == ["--run rehearsal a", "--run rehearsal b"]
    import subprocess

    with pytest.raises(subprocess.CalledProcessError):
        run_steps(tmp_path, "rehearsal", [("bad.py", []), ("ok.py", ["c", str(log)])])
    assert len(log.read_text().splitlines()) == 2  # the step after the failure never ran
