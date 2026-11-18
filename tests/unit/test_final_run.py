"""Tests for hmrec.final.run: which steps the driver starts, in which order, and when it refuses."""

from pathlib import Path

import pytest
from final_example import COMMIT, NOW, InProcessWorker, copy_workspace, load_context

from hmrec.csv_records import read_rows
from hmrec.final.context import MODE_FORMAL, MODE_REHEARSAL
from hmrec.final.gate import LABEL_ACCESS_COLUMNS, FinalRunRefused, begin_formal_run
from hmrec.final.run import (
    STEP_BASELINE,
    STEP_DATASET,
    STEP_MONITOR,
    STEP_SAMPLE,
    STEP_TRIAL,
    StepRequest,
    check_request,
    ensure_step,
    run_formal,
    run_rehearsal,
    step_arguments,
    step_is_done,
)

T_FOLDS = ("T1", "T2", "T3", "T4")


def test_step_arguments_are_the_command_line_of_the_worker_script() -> None:
    request = StepRequest(STEP_TRIAL, MODE_FORMAL, "T3", "minus_r6")
    assert step_arguments(request) == [
        "--step", "trial", "--mode", "formal", "--fold", "T3", "--experiment", "minus_r6",
    ]  # fmt: skip


def test_unknown_steps_and_rehearsal_sampling_are_refused() -> None:
    check_request(StepRequest(STEP_SAMPLE, MODE_FORMAL, "T1"))  # no error
    for request in (
        StepRequest("tune", MODE_FORMAL, "T1"),
        StepRequest(STEP_TRIAL, "dry_run", "T1"),
    ):
        with pytest.raises(ValueError, match="unknown step or mode"):
            check_request(request)
    with pytest.raises(ValueError, match="experiment must be one of"):
        check_request(StepRequest(STEP_TRIAL, MODE_FORMAL, "T1", "minus_r5"))
    # The rehearsal trains on M7's sample files only: it never samples.
    with pytest.raises(ValueError, match="samples are drawn only in the formal run"):
        check_request(StepRequest(STEP_SAMPLE, MODE_REHEARSAL, "V4"))


def test_the_rehearsal_runs_six_steps_on_the_rehearsal_fold(tmp_path: Path) -> None:
    context = copy_workspace(tmp_path)
    worker = InProcessWorker(context)
    row = run_rehearsal(context, worker, tmp_path, NOW)
    assert row["status"] == "passed"
    assert worker.calls == [
        StepRequest(STEP_MONITOR, MODE_REHEARSAL, "V4"),
        StepRequest(STEP_BASELINE, MODE_REHEARSAL, "V4"),
        StepRequest(STEP_DATASET, MODE_REHEARSAL, "V4", "full"),
        StepRequest(STEP_TRIAL, MODE_REHEARSAL, "V4", "full"),
        StepRequest(STEP_DATASET, MODE_REHEARSAL, "V4", "minus_r6"),
        StepRequest(STEP_TRIAL, MODE_REHEARSAL, "V4", "minus_r6"),
    ]
    assert all(request.fold == "V4" for request in worker.calls)  # no T fold is touched
    # A second rehearsal starts nothing: every step is recorded.
    again = InProcessWorker(context)
    run_rehearsal(context, again, tmp_path, NOW)
    assert again.calls == []


def test_the_formal_run_is_refused_without_a_rehearsal_and_starts_nothing(tmp_path: Path) -> None:
    context = copy_workspace(tmp_path)
    for fold in T_FOLDS:
        context.paths.labels_file(fold).unlink()  # any read of a T label would be an error
    worker = InProcessWorker(context)
    with pytest.raises(FinalRunRefused, match="no rehearsal is recorded"):
        run_formal(context, worker, tmp_path, NOW)
    assert worker.calls == []
    assert read_rows(context.paths.t_phase_label_access_file, LABEL_ACCESS_COLUMNS) == []


def test_a_failed_rehearsal_blocks_the_formal_run(tmp_path: Path) -> None:
    context = copy_workspace(tmp_path)
    # Make M7's record differ from what the code computes: halve the baseline's MAP.
    metrics_file = context.paths.v_phase_metrics_file
    lines = metrics_file.read_text(encoding="utf-8").splitlines()
    index = next(i for i, line in enumerate(lines) if line.startswith("rrf_baseline,"))
    cells = lines[index].split(",")
    cells[5] = repr(float(cells[5]) / 2)
    lines[index] = ",".join(cells)
    metrics_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    worker = InProcessWorker(context)
    with pytest.raises(FinalRunRefused, match="rrf_baseline metrics at 0 trees"):
        run_rehearsal(context, worker, tmp_path, NOW)
    for fold in T_FOLDS:
        context.paths.labels_file(fold).unlink()
    formal = InProcessWorker(context)
    with pytest.raises(FinalRunRefused, match="did NOT pass: rrf_baseline metrics at 0 trees"):
        run_formal(context, formal, tmp_path, NOW)
    assert formal.calls == []
    assert not context.paths.t_phase_results_file.exists()
    assert not context.paths.t_phase_label_access_file.exists()


def test_the_formal_run_is_refused_with_changed_code_or_a_dirty_tree(tmp_path: Path) -> None:
    context = copy_workspace(tmp_path, rehearsed=True)
    for changed in (
        load_context(context.paths, "c" * 40),
        load_context(context.paths, COMMIT, True),
    ):
        worker = InProcessWorker(changed)
        with pytest.raises(FinalRunRefused):
            run_formal(changed, worker, tmp_path, NOW)
        assert worker.calls == []
    assert not context.paths.t_phase_label_access_file.exists()


def test_the_order_of_the_formal_run(tmp_path: Path) -> None:
    context = copy_workspace(tmp_path, rehearsed=True)
    worker = InProcessWorker(context)
    run_formal(context, worker, tmp_path, NOW)
    steps = [(request.step, request.fold, request.experiment) for request in worker.calls]
    assert all(request.mode == MODE_FORMAL for request in worker.calls)
    new_folds = ("V4", "T1", "T2", "T3")
    expected = [(STEP_MONITOR, fold, "full") for fold in T_FOLDS]
    expected += [(STEP_BASELINE, fold, "full") for fold in T_FOLDS]
    for experiment in ("full", "minus_r6"):  # the main result first, then the ablation
        expected += [(STEP_SAMPLE, fold, experiment) for fold in new_folds]
        for fold in T_FOLDS:
            expected += [(STEP_DATASET, fold, experiment), (STEP_TRIAL, fold, experiment)]
    assert steps == expected
    # V1, V2, V3 are never sampled; T4 is never sampled.
    sampled = {fold for step, fold, _ in steps if step == STEP_SAMPLE}
    assert sampled == set(new_folds)


def test_recorded_steps_are_skipped(tmp_path: Path) -> None:
    context = copy_workspace(tmp_path, rehearsed=True)
    request = StepRequest(STEP_MONITOR, MODE_FORMAL, "T1")
    assert not step_is_done(context, request)
    begin_formal_run(context, NOW)
    worker = InProcessWorker(context)
    ensure_step(context, worker, tmp_path, request)
    assert worker.calls == [request] and step_is_done(context, request)
    ensure_step(context, worker, tmp_path, request)  # recorded: not started again
    assert worker.calls == [request]
    # Other steps of the same fold are not done yet.
    for step in (STEP_BASELINE, STEP_DATASET, STEP_TRIAL):
        assert not step_is_done(context, StepRequest(step, MODE_FORMAL, "T1"))
    assert not step_is_done(context, StepRequest(STEP_SAMPLE, MODE_FORMAL, "T1"))
    # The rehearsal's steps on V4 are done, in their own area.
    assert step_is_done(context, StepRequest(STEP_TRIAL, MODE_REHEARSAL, "V4", "minus_r6"))
