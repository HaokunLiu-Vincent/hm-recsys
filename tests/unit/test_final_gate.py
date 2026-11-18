"""Tests for hmrec.final.gate: no formal run without a passed rehearsal, and only one formal run.

The workspace holds configs only (tests/unit/final_example.py): the gate reads the
rehearsal record and the label access record, never data or labels.
"""

import json
from pathlib import Path

import pytest
from final_example import COMMIT, NOW, OTHER_COMMIT, load_context, write_configs

from hmrec.csv_records import append_row, read_rows
from hmrec.final.context import FinalContext
from hmrec.final.gate import (
    FAILED,
    FIRST_T_LABEL_READ,
    LABEL_ACCESS_COLUMNS,
    PASSED,
    REHEARSAL_CHECK_COLUMNS,
    FinalRunRefused,
    begin_formal_run,
    formal_run_permit,
    passed_rehearsal,
    rehearsal_sample_sha256,
    require_clean_tree,
)

SAMPLES = {"full": {"V1": "1" * 64, "V2": "2" * 64}, "minus_r6": {"V1": "3" * 64}}


def _context(tmp_path: Path, commit: str = COMMIT, dirty: bool = False) -> FinalContext:
    return load_context(write_configs(tmp_path), commit, dirty)


def _record_rehearsal(context: FinalContext, **changes: str) -> None:
    """Append a rehearsal check row; `changes` replace values of a passed row at COMMIT."""
    row = {
        "status": PASSED,
        "target_fold": context.config.rehearsal_target_fold,
        "config_id": context.locked.config_id,
        "num_trees": str(context.locked.num_trees),
        "n_checks": "10",
        "failed_checks": "",
        "checks_json": "[]",
        "sample_sha256_json": json.dumps(SAMPLES),
        "git_commit": COMMIT,
        "git_dirty": "False",
        "checked_at": "2026-11-18T11:00:00",
    }
    append_row(context.paths.m8_rehearsal_check_file, REHEARSAL_CHECK_COLUMNS, {**row, **changes})


def _access_rows(context: FinalContext) -> list[dict[str, str]]:
    return read_rows(context.paths.t_phase_label_access_file, LABEL_ACCESS_COLUMNS)


# ---------------------------------------------------------------- rule 1: the rehearsal


def test_without_a_rehearsal_the_formal_run_is_refused(tmp_path: Path) -> None:
    context = _context(tmp_path)
    with pytest.raises(FinalRunRefused, match="no rehearsal is recorded"):
        begin_formal_run(context, NOW)
    with pytest.raises(FinalRunRefused, match="no rehearsal is recorded"):
        formal_run_permit(context)
    assert _access_rows(context) == []  # nothing was recorded: the run did not start


def test_a_failed_rehearsal_refuses_the_formal_run(tmp_path: Path) -> None:
    context = _context(tmp_path)
    _record_rehearsal(context, status=FAILED, failed_checks="full metrics at 10 trees")
    with pytest.raises(FinalRunRefused, match="did NOT pass: full metrics at 10 trees"):
        begin_formal_run(context, NOW)
    assert _access_rows(context) == []


def test_the_latest_rehearsal_counts(tmp_path: Path) -> None:
    context = _context(tmp_path)
    _record_rehearsal(context)
    _record_rehearsal(context, status=FAILED, failed_checks="recall monitoring rows")
    # Passed once, then failed: the formal run does not start.
    with pytest.raises(FinalRunRefused, match="did NOT pass"):
        passed_rehearsal(context)
    _record_rehearsal(context, checked_at="2026-11-18T11:30:00")
    assert passed_rehearsal(context)["checked_at"] == "2026-11-18T11:30:00"


def test_a_rehearsal_of_other_code_or_of_a_dirty_tree_does_not_count(tmp_path: Path) -> None:
    context = _context(tmp_path)
    _record_rehearsal(context, git_commit=OTHER_COMMIT)  # the code changed since then
    with pytest.raises(FinalRunRefused, match="are not the same. Run the rehearsal again"):
        begin_formal_run(context, NOW)
    _record_rehearsal(context, git_dirty="True")  # at COMMIT, but with uncommitted edits
    with pytest.raises(FinalRunRefused, match="dirty: True"):
        begin_formal_run(context, NOW)
    assert _access_rows(context) == []


def test_a_rehearsal_of_another_configuration_does_not_count(tmp_path: Path) -> None:
    context = _context(tmp_path)
    for changes in ({"config_id": "0" * 12}, {"num_trees": "20"}, {"target_fold": "V3"}):
        _record_rehearsal(context, **changes)
        with pytest.raises(FinalRunRefused, match="but the locked configuration is"):
            passed_rehearsal(context)


def test_a_dirty_working_tree_is_refused_before_anything_is_read(tmp_path: Path) -> None:
    dirty = _context(tmp_path, dirty=True)
    _record_rehearsal(dirty)
    with pytest.raises(FinalRunRefused, match="needs a clean working tree"):
        begin_formal_run(dirty, NOW)
    with pytest.raises(FinalRunRefused, match="the rehearsal needs a clean working tree"):
        require_clean_tree(dirty, "the rehearsal")
    assert _access_rows(dirty) == []


# ---------------------------------------------------------------- rules 2 and 3: the start


def test_the_first_read_of_t_labels_is_recorded_once(tmp_path: Path) -> None:
    context = _context(tmp_path)
    _record_rehearsal(context)
    permit = begin_formal_run(context, NOW)
    assert _access_rows(context) == [
        {
            "event": FIRST_T_LABEL_READ,
            "started_at": NOW,
            "git_commit": COMMIT,
            "git_dirty": "False",
            "rehearsal_commit": COMMIT,
            "rehearsal_checked_at": "2026-11-18T11:00:00",
        }
    ]
    assert permit.first_label_read_at == NOW and permit.git_commit == COMMIT
    # Continuing an interrupted run later: the record is left as it is.
    again = begin_formal_run(context, "2026-11-19T08:00:00")
    assert len(_access_rows(context)) == 1 and again.first_label_read_at == NOW
    assert formal_run_permit(context) == again


def test_a_worker_never_starts_the_formal_run_itself(tmp_path: Path) -> None:
    context = _context(tmp_path)
    _record_rehearsal(context)
    with pytest.raises(FinalRunRefused, match="the start of the formal run is not recorded"):
        formal_run_permit(context)
    assert _access_rows(context) == []


def test_after_the_start_changed_code_is_refused(tmp_path: Path) -> None:
    context = _context(tmp_path)
    _record_rehearsal(context)
    begin_formal_run(context, NOW)
    # A later process at another commit, even with its own passed rehearsal: the formal
    # run started with COMMIT and T labels have been read; it is not continued.
    later = load_context(context.paths, OTHER_COMMIT)
    _record_rehearsal(later, git_commit=OTHER_COMMIT, checked_at="2026-11-19T09:00:00")
    with pytest.raises(FinalRunRefused, match="is done once and is not continued"):
        begin_formal_run(later, "2026-11-19T10:00:00")
    with pytest.raises(FinalRunRefused, match="is done once and is not continued"):
        formal_run_permit(later)
    assert len(_access_rows(context)) == 1  # still only the first start


def test_the_rehearsal_row_carries_the_sample_hashes(tmp_path: Path) -> None:
    context = _context(tmp_path)
    _record_rehearsal(context)
    assert rehearsal_sample_sha256(passed_rehearsal(context)) == SAMPLES
