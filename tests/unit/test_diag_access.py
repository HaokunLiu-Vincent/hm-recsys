"""Tests for hmrec.diagnostics.access: T labels are read again only after checks and a record."""

from pathlib import Path

import pytest
from diag_example import (
    COMMIT,
    FORMAL_COMMIT,
    NOW,
    SCRIPT,
    build_diag_workspace,
    make_context,
    write_goal_rows,
)

from hmrec.csv_records import read_rows
from hmrec.diagnostics.access import (
    ACCESS_COLUMNS,
    M8_PROTECTED_PATHS,
    DiagnosticsRefused,
    formal_run_commit,
    issue_diagnostic_access,
)

REPO = Path(__file__).resolve().parents[2]


def _goal(goal: str, commit: str = FORMAL_COMMIT) -> dict[str, str]:
    return {"goal": goal, "git_commit": commit}


def test_formal_run_commit_needs_the_three_verdicts_of_one_commit() -> None:
    assert formal_run_commit([_goal("G1"), _goal("G2"), _goal("G3")]) == FORMAL_COMMIT
    for unfinished in ([], [_goal("G1"), _goal("G2")], [_goal("G1")] * 3):
        with pytest.raises(DiagnosticsRefused, match="not finished"):
            formal_run_commit(unfinished)
    with pytest.raises(DiagnosticsRefused, match="several commits"):
        formal_run_commit([_goal("G1"), _goal("G2"), _goal("G3", "e" * 40)])


def test_the_read_is_recorded_before_the_access_is_handed_out(tmp_path: Path) -> None:
    context = build_diag_workspace(tmp_path)
    access = issue_diagnostic_access(context, NOW, SCRIPT)
    assert (access.git_commit, access.formal_run_commit) == (COMMIT, FORMAL_COMMIT)
    assert (access.recorded_at, access.purpose) == (NOW, "descriptive diagnostics")
    rows = read_rows(context.paths.t_diag_label_access_file, ACCESS_COLUMNS)
    assert rows == [
        {
            "event": "t_labels_read_for_diagnostics",
            "recorded_at": NOW,
            "purpose": "descriptive diagnostics",
            "script": SCRIPT,
            "git_commit": COMMIT,
            "git_dirty": "False",
            "formal_run_commit": FORMAL_COMMIT,
        }
    ]
    # Every run records its own read: a second run appends a second row.
    issue_diagnostic_access(context, "2026-11-20T09:00:00", SCRIPT)
    rows = read_rows(context.paths.t_diag_label_access_file, ACCESS_COLUMNS)
    assert [row["recorded_at"] for row in rows] == [NOW, "2026-11-20T09:00:00"]
    # The record of the formal run itself is another file and is not touched.
    assert not context.paths.t_phase_label_access_file.exists()


def test_refusals_write_no_record(tmp_path: Path) -> None:
    paths = build_diag_workspace(tmp_path).paths
    refused = (
        (make_context(paths, dirty=True), "clean working tree"),
        (make_context(paths, changed=("M src/hmrec/final/gate.py",)), "changed since the formal"),
        (make_context(paths, changed=("D configs/m8_final.yaml",)), "changed since the formal"),
    )
    for context, message in refused:
        with pytest.raises(DiagnosticsRefused, match=message):
            issue_diagnostic_access(context, NOW, SCRIPT)
    assert not paths.t_diag_label_access_file.exists()


def test_an_unfinished_formal_run_is_refused(tmp_path: Path) -> None:
    context = build_diag_workspace(tmp_path)
    context.paths.t_phase_goals_file.unlink()
    with pytest.raises(DiagnosticsRefused, match="not finished"):
        issue_diagnostic_access(context, NOW, SCRIPT)
    write_goal_rows(context.paths, goals=("G1", "G2"))  # the verdict on G3 is missing
    with pytest.raises(DiagnosticsRefused, match="not finished"):
        issue_diagnostic_access(context, NOW, SCRIPT)
    assert not context.paths.t_diag_label_access_file.exists()


def test_the_gate_asks_about_the_formal_run_commit_and_the_protected_paths(tmp_path: Path) -> None:
    paths = build_diag_workspace(tmp_path).paths
    asked: list[tuple[str, tuple[str, ...]]] = []

    def changed_existing(commit: str, protected: tuple[str, ...]) -> list[str]:
        asked.append((commit, tuple(protected)))
        return []

    context = make_context(paths)
    context = type(context)(**{**context.__dict__, "changed_existing": changed_existing})
    issue_diagnostic_access(context, NOW, SCRIPT)
    assert asked == [(FORMAL_COMMIT, M8_PROTECTED_PATHS)]


def test_the_protected_paths_cover_the_m8_code_and_the_configs() -> None:
    assert "configs" in M8_PROTECTED_PATHS
    for relative in M8_PROTECTED_PATHS:
        assert (REPO / relative).exists(), relative
    # Every file of the final evaluation (the list of tests/unit/test_final_code_rules.py).
    m8_files = [
        REPO / "scripts" / "m8_final_eval.py",
        REPO / "scripts" / "m8_step.py",
        *sorted((REPO / "src" / "hmrec" / "final").glob("*.py")),
        REPO / "src" / "hmrec" / "eval" / "final_eval.py",
        REPO / "src" / "hmrec" / "ranker" / "final_window.py",
    ]
    for path in m8_files:
        relative = path.relative_to(REPO).as_posix()
        covered = any(
            relative == protected or relative.startswith(f"{protected}/")
            for protected in M8_PROTECTED_PATHS
        )
        assert covered, relative
    # The diagnostics' own code is NOT protected: it is new and may still change.
    assert not any("diagnostics" in protected for protected in M8_PROTECTED_PATHS)
