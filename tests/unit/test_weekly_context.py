"""Tests for weekly.context and weekly.run_step: the record a step leaves, and a step that fails.

Workspace of weekly_example.py (real calendar, tiny files). The two functions
that need a real repository or image (hmrec.git_state, hmrec.platform_fields) are
replaced where a step record is written.
"""

import time
from pathlib import Path

import pytest
from run_manifest_example import TORCH_PLATFORM
from weekly_example import STAMP, V1, V2, build_workspace, tick_context, write_processed

from hmrec.paths import ProjectPaths
from hmrec.run_manifest import EVAL_MODE, SERVE_MODE, MetricRecord, StepRecord
from hmrec.run_records import RunStamp
from hmrec.weekly import context as context_module
from hmrec.weekly import run_step as run_step_module
from hmrec.weekly.context import (
    OutputFile,
    finish_step,
    load_tick_context,
    read_step_output,
    require_committed_code,
)
from hmrec.weekly.run_step import dispatch, run_one_step
from hmrec.weekly.steps import staging_dir, step_by_name


@pytest.fixture
def paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ProjectPaths:
    def record(name: str, seconds: float, detail: str, repo_root: Path) -> StepRecord:
        return StepRecord(name, "done", seconds, 0.5, TORCH_PLATFORM, detail)

    monkeypatch.setattr(context_module, "step_record_of_this_process", record)
    return build_workspace(tmp_path)


def test_context_names_the_two_weeks_of_a_tick(paths: ProjectPaths) -> None:
    first = tick_context(paths, V1)
    assert (first.tick.name, first.closing.name, first.first_week.name) == ("V1", "F4", "V1")
    # The week before the first tick is closed in the bootstrap state.
    assert not first.closes_a_week and first.ranker_experiment == "full"
    second = tick_context(paths, V2, SERVE_MODE)
    assert second.closes_a_week and second.mode == SERVE_MODE
    assert second.view.week_anchors == (("V1", V1), ("V2", V2))
    with pytest.raises(ValueError, match="mode must be one of"):
        load_tick_context(paths, V1, "train", "run", STAMP)


def test_a_step_context_writes_into_its_own_staging_folder(paths: ProjectPaths) -> None:
    plain = tick_context(paths, V2)
    sample = step_by_name("C3_sample", plain.tick, plain.closing, EVAL_MODE)
    context = load_tick_context(paths, V2, EVAL_MODE, "run", STAMP, sample)
    week = paths.pipeline_state_dir / "weeks" / "2020-07-29"
    # C3 writes the CLOSED week's sample; everything else it reads from finished folders.
    assert context.view.ranker_sample_file("full", "V1", 40).parent == week / ".staging" / "sample"
    assert context.view.features_dir("V1") == week / "features"


def test_uncommitted_code_is_refused() -> None:
    require_committed_code(STAMP)
    with pytest.raises(RuntimeError, match="runs only from committed code"):
        require_committed_code(RunStamp(STAMP.git_commit, True))


def test_a_finished_step_leaves_its_record_in_its_folder(paths: ProjectPaths) -> None:
    plain = tick_context(paths, V2)
    step = step_by_name("C3_sample", plain.tick, plain.closing, EVAL_MODE)
    context = load_tick_context(paths, V2, EVAL_MODE, "run", STAMP, step)
    folder = staging_dir(context.layout, step)
    folder.mkdir(parents=True)
    (folder / "r40.parquet").write_bytes(b"sample")
    metrics = [MetricRecord("sample/rows", context.closing.anchor, 180.0)]
    files = [OutputFile("closed_week/sample", "r40.parquet")]
    finish_step(context, step, time.perf_counter(), "180 rows", metrics, files)
    output = read_step_output(folder)
    assert (output.name, output.detail, output.process_max_rss_gb) == ("C3_sample", "180 rows", 0.5)
    assert output.platform == TORCH_PLATFORM and output.seconds >= 0
    assert output.metrics == tuple(metrics) and output.files == tuple(files)
    # A record is written once.
    with pytest.raises(FileExistsError, match="never replaced"):
        finish_step(context, step, time.perf_counter(), "again", metrics, files)


def test_a_step_that_did_not_write_its_outputs_leaves_no_record(paths: ProjectPaths) -> None:
    plain = tick_context(paths, V1)
    step = step_by_name("O1_queries", plain.tick, plain.closing, EVAL_MODE)
    context = load_tick_context(paths, V1, EVAL_MODE, "run", STAMP, step)
    folder = staging_dir(context.layout, step)
    folder.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="did not write its outputs"):
        finish_step(
            context, step, time.perf_counter(), "", files=[OutputFile("queries", "queries.parquet")]
        )
    with pytest.raises(FileNotFoundError, match="left no step.json"):
        read_step_output(folder)


def test_a_step_above_the_memory_limit_fails_and_leaves_no_record(
    paths: ProjectPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hmrec.memory import ProcessPeakMemory

    plain = tick_context(paths, V1)
    step = step_by_name("C1_ingest", plain.tick, plain.closing, EVAL_MODE)
    context = load_tick_context(paths, V1, EVAL_MODE, "run", STAMP, step)
    folder = staging_dir(context.layout, step)
    folder.mkdir(parents=True)
    peak = ProcessPeakMemory(process_max_rss_bytes=13 * 1024**3, children_max_rss_bytes=0)
    monkeypatch.setattr(context_module, "log_process_peak_memory", lambda *args: peak)
    with pytest.raises(MemoryError, match="above the limit of 12.00 GB"):
        finish_step(context, step, time.perf_counter(), "too much")
    assert not (folder / "step.json").exists()


# ---------------------------------------------------------------- run_step


def test_a_plain_step_is_dispatched_without_loading_a_model_library(paths: ProjectPaths) -> None:
    import sys

    write_processed(paths, [(0, 1, 670), (1, 2, 676)], n_customers=2, n_articles=3)
    plain = tick_context(paths, V1)
    step = step_by_name("C1_ingest", plain.tick, plain.closing, EVAL_MODE)
    context = load_tick_context(paths, V1, EVAL_MODE, "run", STAMP, step)
    staging_dir(context.layout, step).mkdir(parents=True)
    lightgbm_before = "lightgbm" in sys.modules
    detail, _, files = dispatch(context, step)
    assert (
        "2 purchase lines before 2020-07-29" in detail and files[0].name == "transactions.parquet"
    )
    # Running a plain step imported no LightGBM into this (torch-side) test process.
    assert ("lightgbm" in sys.modules) == lightgbm_before


def test_a_failing_step_writes_its_reason_before_the_error_goes_on(
    paths: ProjectPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    plain = tick_context(paths, V1)
    step = step_by_name("O2_features", plain.tick, plain.closing, EVAL_MODE)
    context = load_tick_context(paths, V1, EVAL_MODE, "run-7", STAMP, step)

    def failing(context, step):
        raise ValueError("structural check failed: 9 feature rows, 10 candidates")

    monkeypatch.setattr(run_step_module, "start_step", lambda *args: (context, step))
    monkeypatch.setattr(run_step_module, "dispatch", failing)
    with pytest.raises(ValueError, match="structural check failed"):
        run_one_step(V1, EVAL_MODE, "run-7", "O2_features")
    reason = context.layout.step_failure_file("run-7", "O2_features").read_text()
    assert reason == "ValueError: structural check failed: 9 feature rows, 10 candidates\n"
