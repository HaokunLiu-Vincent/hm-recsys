"""Tests for hmrec.t_eval.run: the order of the trainings, the gate before the formal run.

Trainings run in this process with the code of scripts/m9_t_trial.py
(t_eval_example.in_process_launcher) on the tiny workspace.
"""

from pathlib import Path

import pytest
from t_eval_example import (
    COMMIT,
    METRICS,
    NOW,
    OTHER_COMMIT,
    SEEDS,
    T_FOLDS,
    build_t_eval_workspace,
    context_of,
    copy_workspace,
    in_process_launcher,
    write_v_records,
)

from hmrec.csv_records import read_rows
from hmrec.deepfm.records import TRIAL_COLUMNS
from hmrec.deepfm.trial import TrialInputs
from hmrec.t_eval.gate import (
    ACCESS_COLUMNS,
    EVENT_FIRST,
    EVENT_RESUMED,
    PASSED,
    REHEARSAL_CHECK_COLUMNS,
    TEvalRefused,
)
from hmrec.t_eval.plan import STAGE_FORMAL, STAGE_REHEARSAL, TrialKey
from hmrec.t_eval.run import (
    key_arguments,
    missing_keys,
    run_formal,
    run_rehearsal,
    stage_keys,
)

SCRIPT = "scripts/m9_t_eval.py"
FORMAL_KEYS = [TrialKey(STAGE_FORMAL, seed, fold) for seed in SEEDS for fold in T_FOLDS]


def _now() -> str:
    return NOW


@pytest.fixture(scope="module")
def with_v_records(tmp_path_factory: pytest.TempPathFactory) -> TrialInputs:
    """A workspace with the V records of the chosen configuration (built once)."""
    inputs = build_t_eval_workspace(tmp_path_factory.mktemp("t_eval_run") / "workspace")
    write_v_records(inputs)
    return inputs


@pytest.fixture
def inputs(with_v_records: TrialInputs, tmp_path: Path) -> TrialInputs:
    return copy_workspace(with_v_records, tmp_path / "copy")


def _access_rows(inputs: TrialInputs) -> list[dict[str, str]]:
    return read_rows(inputs.paths.m9_t_label_access_file, ACCESS_COLUMNS)


def test_the_trainings_of_each_stage(inputs: TrialInputs) -> None:
    context = context_of(inputs)
    assert stage_keys(context, STAGE_REHEARSAL) == [
        TrialKey(STAGE_REHEARSAL, seed, "V4") for seed in SEEDS
    ]
    assert stage_keys(context, STAGE_FORMAL) == FORMAL_KEYS
    with pytest.raises(ValueError, match="unknown stage"):
        stage_keys(context, "tuning")
    assert key_arguments(TrialKey(STAGE_FORMAL, 43, "T2")) == [
        "--stage",
        "formal",
        "--seed",
        "43",
        "--target",
        "T2",
    ]


def test_the_rehearsal_trains_every_seed_once_and_passes(inputs: TrialInputs) -> None:
    context = context_of(inputs)
    calls: list[TrialKey] = []
    status, checks = run_rehearsal(context, in_process_launcher(inputs, calls=calls), METRICS, _now)
    assert status == PASSED and len(checks) == 4 * len(SEEDS)
    assert calls == stage_keys(context, STAGE_REHEARSAL)
    rows = read_rows(inputs.paths.m9_t_rehearsal_check_file, REHEARSAL_CHECK_COLUMNS)
    assert [(row["status"], row["git_commit"], row["checked_at"]) for row in rows] == [
        (PASSED, COMMIT, NOW)
    ]
    # Run again at the same commit: nothing is trained again, the comparison is repeated.
    status, _ = run_rehearsal(context, in_process_launcher(inputs, calls=calls), METRICS, _now)
    assert status == PASSED and len(calls) == len(SEEDS)
    # No T label was read and nothing was recorded as read.
    assert not inputs.paths.m9_t_label_access_file.exists()


def test_a_rehearsal_at_another_commit_trains_again(inputs: TrialInputs) -> None:
    run_rehearsal(context_of(inputs), in_process_launcher(inputs), METRICS, _now)
    other = context_of(inputs, commit=OTHER_COMMIT)
    assert missing_keys(other, stage_keys(other, STAGE_REHEARSAL)) == stage_keys(
        other, STAGE_REHEARSAL
    )
    calls: list[TrialKey] = []
    launcher = in_process_launcher(inputs, commit=OTHER_COMMIT, calls=calls)
    status, _ = run_rehearsal(other, launcher, METRICS, _now)
    assert status == PASSED and len(calls) == len(SEEDS)


def test_a_dirty_tree_runs_no_rehearsal(inputs: TrialInputs) -> None:
    calls: list[TrialKey] = []
    with pytest.raises(TEvalRefused, match="clean working tree"):
        run_rehearsal(
            context_of(inputs, dirty=True), in_process_launcher(inputs, calls=calls), METRICS, _now
        )
    assert calls == [] and not inputs.paths.m9_t_rehearsal_check_file.exists()


def test_no_formal_training_without_a_passed_rehearsal(inputs: TrialInputs) -> None:
    calls: list[TrialKey] = []
    with pytest.raises(TEvalRefused, match="no rehearsal is recorded"):
        run_formal(context_of(inputs), in_process_launcher(inputs, calls=calls), _now, SCRIPT)
    assert calls == []
    assert not inputs.paths.m9_t_label_access_file.exists()
    assert not inputs.paths.m9_t_trials_file.exists()
    # A rehearsal of another commit does not open the gate either.
    run_rehearsal(context_of(inputs), in_process_launcher(inputs), METRICS, _now)
    other = context_of(inputs, commit=OTHER_COMMIT)
    launcher = in_process_launcher(inputs, commit=OTHER_COMMIT, calls=calls)
    with pytest.raises(TEvalRefused, match="only at the commit of a passed rehearsal"):
        run_formal(other, launcher, _now, SCRIPT)
    assert calls == [] and not inputs.paths.m9_t_label_access_file.exists()


def test_the_formal_run_records_the_read_first_and_trains_everything_once(
    inputs: TrialInputs,
) -> None:
    context = context_of(inputs)
    run_rehearsal(context, in_process_launcher(inputs), METRICS, _now)
    calls: list[TrialKey] = []
    rows_at_launch: list[int] = []
    worker = in_process_launcher(inputs, calls=calls)

    def launcher(key: TrialKey) -> None:
        # The access row exists BEFORE the first training starts.
        rows_at_launch.append(len(_access_rows(inputs)))
        worker(key)

    assert run_formal(context, launcher, _now, SCRIPT) == len(FORMAL_KEYS)
    assert calls == FORMAL_KEYS and rows_at_launch == [1] * len(FORMAL_KEYS)
    assert [row["event"] for row in _access_rows(inputs)] == [EVENT_FIRST]
    formal_rows = [
        row
        for row in read_rows(inputs.paths.m9_t_trials_file, TRIAL_COLUMNS)
        if row["area"] == "m9_t"
    ]
    assert [(int(row["seed"]), row["fold"]) for row in formal_rows] == [
        (key.seed, key.target_fold) for key in FORMAL_KEYS
    ]
    assert formal_rows[1]["train_folds"] == "V2 V3 V4 T1"
    # Everything is recorded: the same command does nothing and records no new read.
    assert run_formal(context, launcher, _now, SCRIPT) == 0
    assert len(calls) == len(FORMAL_KEYS) and len(_access_rows(inputs)) == 1


def test_an_interrupted_formal_run_is_continued_by_the_same_command(inputs: TrialInputs) -> None:
    context = context_of(inputs)
    run_rehearsal(context, in_process_launcher(inputs), METRICS, _now)
    calls: list[TrialKey] = []
    worker = in_process_launcher(inputs, calls=calls)

    def failing_launcher(key: TrialKey) -> None:
        if len(calls) == 3:
            raise RuntimeError("the process was interrupted")
        worker(key)

    with pytest.raises(RuntimeError, match="interrupted"):
        run_formal(context, failing_launcher, _now, SCRIPT)
    assert calls == FORMAL_KEYS[:3]
    assert run_formal(context, worker, _now, SCRIPT) == len(FORMAL_KEYS) - 3
    assert calls == FORMAL_KEYS
    assert [row["event"] for row in _access_rows(inputs)] == [EVENT_FIRST, EVENT_RESUMED]
    # The formal area does not hold the commit: at any commit these trainings count as done.
    assert missing_keys(context_of(inputs, commit=OTHER_COMMIT), FORMAL_KEYS) == []


def test_a_training_that_leaves_no_row_stops_the_run(inputs: TrialInputs) -> None:
    context = context_of(inputs)
    run_rehearsal(context, in_process_launcher(inputs), METRICS, _now)
    with pytest.raises(RuntimeError, match="finished without its trial row"):
        run_formal(context, lambda key: None, _now, SCRIPT)
