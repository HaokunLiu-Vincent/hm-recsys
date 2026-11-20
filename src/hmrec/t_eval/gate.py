"""The gate of the M9 evaluation on T1-T4: no T label before a passed rehearsal and a record.

The evaluation is run ONCE and is reported only. This module enforces what code
can enforce (proposal §16.12):

    1. No T label without a passed rehearsal AT THE COMMIT OF HEAD. The latest
       row of results/m9_t_rehearsal_check.csv must say "passed", must have been
       written from a clean working tree at the very commit this process runs
       at, and must be about the configured rehearsal fold, seeds and
       configuration. A rehearsal of other code proves nothing about this code.
    2. Every process needs a clean working tree.
    3. The read is recorded BEFORE any T label is opened: the driver of the
       formal run appends a row to results/m9_t_label_access.csv (time, commit,
       purpose, script) and only then gets its permit (begin_formal_run). A
       driver that continues an interrupted run appends its own row.
    4. A worker process never records anything itself: it gets a permit only if
       such a row of ITS OWN commit exists (worker_permit).
    5. All rows of the access table must carry the commit of HEAD: a formal run
       that was started is continued with the same code or not at all.

Passing the gate gives a TEvalPermit, which hmrec.eval.t_eval_labels requires
before it opens a T label file. Only this module creates permits. The sample
files of T1-T3 hold T labels too (M8 wrote them), so the trial code asks for the
permit before it opens any file.

This module imports neither torch nor LightGBM.
"""

from collections.abc import Mapping

from hmrec.csv_records import append_row, read_rows
from hmrec.eval.t_eval_labels import TEvalPermit
from hmrec.t_eval.plan import TEvalContext

PASSED = "passed"
FAILED = "failed"
PURPOSE = "single evaluation on T1-T4 of the DeepFM configuration chosen on V (reported only)"
EVENT_FIRST = "t_labels_read_for_m9_evaluation"
EVENT_RESUMED = "t_labels_read_for_m9_evaluation_resumed"

REHEARSAL_CHECK_COLUMNS: tuple[str, ...] = (
    "status",
    "target_fold",
    "seeds",
    "config_id",
    "n_checks",
    "failed_checks",
    "checks_json",
    "git_commit",
    "git_dirty",
    "checked_at",
)
ACCESS_COLUMNS: tuple[str, ...] = (
    "event",
    "recorded_at",
    "purpose",
    "script",
    "git_commit",
    "git_dirty",
    "rehearsal_commit",
    "rehearsal_checked_at",
)


class TEvalRefused(RuntimeError):
    """A step of the evaluation may not run (see the message); nothing was read."""


def require_clean_tree(context: TEvalContext, what: str) -> None:
    """Fail if code or configs differ from HEAD: a result must belong to one commit.

    Raises:
        TEvalRefused: If the working tree has uncommitted changes outside results/.
    """
    if context.stamp.git_dirty:
        raise TEvalRefused(
            f"{what} needs a clean working tree, but code or configs differ from HEAD "
            f"({context.stamp.git_commit[:12]}). Commit first."
        )


def passed_rehearsal(context: TEvalContext) -> Mapping[str, str]:
    """The row of the rehearsal that allows the formal run, after checking it (rules 1 and 2).

    Returns:
        The latest row of results/m9_t_rehearsal_check.csv.

    Raises:
        TEvalRefused: If there is no rehearsal, the latest one failed, it ran on a
            dirty tree or at another commit than HEAD, or it checked another
            fold, other seeds or another configuration than the configured ones.
    """
    require_clean_tree(context, "the formal run")
    check_file = context.paths.m9_t_rehearsal_check_file
    rows = read_rows(check_file, REHEARSAL_CHECK_COLUMNS)
    if not rows:
        raise TEvalRefused(
            f"no rehearsal is recorded in {check_file}. Run the rehearsal first; no T label "
            "is read before it has passed."
        )
    latest = rows[-1]
    if latest["status"] != PASSED:
        raise TEvalRefused(
            f"the latest rehearsal ({latest['checked_at']}) did NOT pass: "
            f"{latest['failed_checks']}. No T label is read; a human decides."
        )
    config = context.config
    expected = (
        config.rehearsal_target_fold,
        " ".join(str(seed) for seed in config.seeds),
        config.config_id,
    )
    recorded = (latest["target_fold"], latest["seeds"], latest["config_id"])
    if recorded != expected:
        raise TEvalRefused(
            f"the latest rehearsal checked (fold, seeds, configuration) = {recorded}, but the "
            f"configured ones are {expected}"
        )
    if latest["git_dirty"] != "False" or latest["git_commit"] != context.stamp.git_commit:
        raise TEvalRefused(
            f"the latest rehearsal ran at commit {latest['git_commit'][:12]} "
            f"(dirty: {latest['git_dirty']}), but HEAD is {context.stamp.git_commit[:12]}. "
            "T labels are read only at the commit of a passed rehearsal: run the rehearsal "
            "again at this commit."
        )
    return latest


def _permit(
    context: TEvalContext, rehearsal: Mapping[str, str], access: Mapping[str, str]
) -> TEvalPermit:
    """The permit of this process (the ONLY place a TEvalPermit is created)."""
    return TEvalPermit(
        git_commit=context.stamp.git_commit,
        rehearsal_checked_at=rehearsal["checked_at"],
        recorded_at=access["recorded_at"],
        purpose=access["purpose"],
        script=access["script"],
    )


def _require_same_commit(context: TEvalContext, rows: list[dict[str, str]]) -> None:
    """Rule 5: every recorded read was made at the commit of HEAD."""
    other = sorted({row["git_commit"] for row in rows} - {context.stamp.git_commit})
    if other:
        raise TEvalRefused(
            f"T labels were already read by this evaluation at commit(s) "
            f"{[commit[:12] for commit in other]}, but HEAD is "
            f"{context.stamp.git_commit[:12]}. The evaluation is run once and is not "
            "continued with other code; a human decides."
        )


def begin_formal_run(context: TEvalContext, now: str, script: str) -> TEvalPermit:
    """Open the gate for the driver: check the rehearsal and RECORD the read (rules 1-3, 5).

    Every call appends one row: the first one "t_labels_read_for_m9_evaluation",
    a later one (the run was interrupted and is continued) "..._resumed".

    Args:
        context: The context of this process.
        now: The current time as ISO text; written to the record.
        script: The script that starts the run, e.g. "scripts/m9_t_eval.py".

    Returns:
        The permit. The row in results/m9_t_label_access.csv exists when this
        function returns.

    Raises:
        TEvalRefused: If a rule is violated. Nothing was read or written then.
    """
    rehearsal = passed_rehearsal(context)
    access_file = context.paths.m9_t_label_access_file
    rows = read_rows(access_file, ACCESS_COLUMNS)
    _require_same_commit(context, rows)
    access = {
        "event": EVENT_RESUMED if rows else EVENT_FIRST,
        "recorded_at": now,
        "purpose": PURPOSE,
        "script": script,
        "git_commit": context.stamp.git_commit,
        "git_dirty": str(context.stamp.git_dirty),
        "rehearsal_commit": rehearsal["git_commit"],
        "rehearsal_checked_at": rehearsal["checked_at"],
    }
    append_row(context.paths.m9_t_label_access_file, ACCESS_COLUMNS, access)
    return _permit(context, rehearsal, access)


def worker_permit(context: TEvalContext) -> TEvalPermit:
    """The permit of a worker process of the formal run (rules 1, 2, 4 and 5).

    A worker never records the read itself: the driver must have done it
    (begin_formal_run), so that the record always exists before a label is read.

    Raises:
        TEvalRefused: If the rehearsal did not pass at this commit, the driver has
            not recorded the read, or a read was recorded at another commit.
    """
    rehearsal = passed_rehearsal(context)
    access_file = context.paths.m9_t_label_access_file
    rows = read_rows(access_file, ACCESS_COLUMNS)
    if not rows:
        raise TEvalRefused(
            f"the read of T labels is not recorded in {access_file}. Start the formal run "
            "with scripts/m9_t_eval.py --stage formal, which records it before any training."
        )
    _require_same_commit(context, rows)
    return _permit(context, rehearsal, rows[-1])
