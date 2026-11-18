"""The gate of the diagnostics' label door: a read of T labels is checked and recorded first.

hmrec.eval.diagnostic_labels opens the labels of a T fold only with a
DiagnosticAccess. This module is the ONLY place that creates one, and only if:

    1. The formal run of M8 is finished: results/t_phase_goals.csv holds the
       verdict on G1, G2 and G3. Reading the labels again cannot change it.
    2. The working tree is clean (code and configs equal HEAD), so the
       diagnostics' numbers belong to one commit.
    3. The M8 code and every config file that existed at the commit of the
       formal run are unchanged at HEAD. NEW files are allowed (the diagnostics
       code and configs/m8_diagnostics.yaml are new files).
    4. A row was appended to results/t_diag_label_access.csv BEFORE the access
       is handed out: the time, the git commit, the purpose and the script.
       One row per run; rows are never changed.

The access proves the read is recorded. It does not allow anything else: the
diagnostics describe results and must not be used to select or change anything
(AGENTS.md §3.2).
"""

from collections.abc import Mapping, Sequence

from hmrec.csv_records import append_row, read_rows
from hmrec.diagnostics.context import DiagnosticsContext
from hmrec.eval.diagnostic_labels import DiagnosticAccess
from hmrec.final.goals import GOAL_COLUMNS

PURPOSE = "descriptive diagnostics"
EVENT = "t_labels_read_for_diagnostics"
FINISHED_GOALS = ("G1", "G2", "G3")

# The code of the formal run (tests/unit/test_final_code_rules.py: M8_FILES) and the
# configs. A file under these paths that existed at the formal run must be unchanged.
M8_PROTECTED_PATHS: tuple[str, ...] = (
    "configs",
    "src/hmrec/final",
    "src/hmrec/eval/final_eval.py",
    "src/hmrec/ranker/final_window.py",
    "scripts/m8_final_eval.py",
    "scripts/m8_step.py",
)

ACCESS_COLUMNS: tuple[str, ...] = (
    "event",
    "recorded_at",
    "purpose",
    "script",
    "git_commit",
    "git_dirty",
    "formal_run_commit",
)


class DiagnosticsRefused(RuntimeError):
    """The diagnostics may not read T labels (see the message); nothing was read."""


def formal_run_commit(goal_rows: Sequence[Mapping[str, str]]) -> str:
    """The commit of the finished formal run, from the rows of results/t_phase_goals.csv.

    Raises:
        DiagnosticsRefused: If the verdict on G1-G3 is not recorded exactly once
            each, or the rows name different commits.
    """
    goals = [row["goal"] for row in goal_rows]
    if sorted(goals) != list(FINISHED_GOALS):
        raise DiagnosticsRefused(
            f"the formal run of M8 is not finished: results/t_phase_goals.csv holds the goals "
            f"{goals}, expected one row each for {list(FINISHED_GOALS)}. The diagnostics read "
            "T labels only after the verdict is recorded."
        )
    commits = sorted({row["git_commit"] for row in goal_rows})
    if len(commits) != 1:
        raise DiagnosticsRefused(f"the verdict rows name several commits: {commits}")
    return commits[0]


def check_before_access(context: DiagnosticsContext) -> str:
    """Rules 1-3 of the module docstring; returns the commit of the formal run.

    Raises:
        DiagnosticsRefused: If a rule is violated. Nothing was read or written.
    """
    recorded_commit = formal_run_commit(read_rows(context.paths.t_phase_goals_file, GOAL_COLUMNS))
    if context.stamp.git_dirty:
        raise DiagnosticsRefused(
            "the diagnostics need a clean working tree, but code or configs differ from HEAD "
            f"({context.stamp.git_commit[:12]}). Commit first."
        )
    changed = context.changed_existing(recorded_commit, M8_PROTECTED_PATHS)
    if changed:
        raise DiagnosticsRefused(
            f"the M8 code or configs changed since the formal run ({recorded_commit[:12]}): "
            f"{changed}. Files that existed then must be unchanged; only new files are allowed."
        )
    return recorded_commit


def issue_diagnostic_access(context: DiagnosticsContext, now: str, script: str) -> DiagnosticAccess:
    """Check the rules, RECORD the read, then hand out the access (rules 1-4).

    Args:
        context: The context of this process.
        now: The current time as ISO text; written to the record.
        script: The script that reads the labels, e.g. "scripts/m8_diagnostics.py".

    Returns:
        The access hmrec.eval.diagnostic_labels asks for. The row in
        results/t_diag_label_access.csv exists when this function returns.

    Raises:
        DiagnosticsRefused: If a rule is violated. Nothing was read or written.
    """
    recorded_commit = check_before_access(context)
    row = {
        "event": EVENT,
        "recorded_at": now,
        "purpose": PURPOSE,
        "script": script,
        "git_commit": context.stamp.git_commit,
        "git_dirty": str(context.stamp.git_dirty),
        "formal_run_commit": recorded_commit,
    }
    append_row(context.paths.t_diag_label_access_file, ACCESS_COLUMNS, row)
    return DiagnosticAccess(
        git_commit=context.stamp.git_commit,
        formal_run_commit=recorded_commit,
        recorded_at=now,
        purpose=PURPOSE,
    )
