"""The gate of the formal run (M8): it starts only after a passed rehearsal, and only once.

The labels of the T folds had never been read before M8, and the formal run is
done ONCE: after its results are seen, no configuration, code or choice may
change. This module enforces what code can enforce:

    1. No formal step without a passed rehearsal. The latest row of
       results/m8_rehearsal_check.csv must say "passed", must have been written
       from a clean working tree, and the code and configs at HEAD must be the
       same as at the commit of that rehearsal (hmrec.git_state.code_changed_since:
       src/, scripts/, configs/, pyproject.toml, uv.lock). A rehearsal of other
       code proves nothing about this code.
    2. The start of the formal run is recorded BEFORE any T label is opened:
       one row "first_t_label_read" in results/t_phase_label_access.csv with
       the time and the git commit (begin_formal_run, called by the driver).
    3. Once that row exists, every later process must run the same code: if the
       code or configs at HEAD differ from the recorded commit, it refuses.
       So an interrupted formal run can be continued, but not with changed code.
    4. Every process of the formal run needs a clean working tree.

Passing the gate gives a FormalRunPermit, which hmrec.eval.final_eval requires
before it opens a T label file. Only this module creates permits.
"""

import json
from collections.abc import Mapping

from hmrec.csv_records import append_row, read_rows
from hmrec.eval.final_eval import FormalRunPermit
from hmrec.final.context import FinalContext

PASSED = "passed"
FAILED = "failed"
FIRST_T_LABEL_READ = "first_t_label_read"

REHEARSAL_CHECK_COLUMNS: tuple[str, ...] = (
    "status",
    "target_fold",
    "config_id",
    "num_trees",
    "n_checks",
    "failed_checks",
    "checks_json",
    "sample_sha256_json",
    "git_commit",
    "git_dirty",
    "checked_at",
)
LABEL_ACCESS_COLUMNS: tuple[str, ...] = (
    "event",
    "started_at",
    "git_commit",
    "git_dirty",
    "rehearsal_commit",
    "rehearsal_checked_at",
)


class FinalRunRefused(RuntimeError):
    """A step of the final evaluation may not run (see the message); nothing was read."""


def require_clean_tree(context: FinalContext, what: str) -> None:
    """Fail if code or configs differ from HEAD: an M8 result must belong to one commit.

    Raises:
        FinalRunRefused: If the working tree has uncommitted changes outside results/.
    """
    if context.stamp.git_dirty:
        raise FinalRunRefused(
            f"{what} needs a clean working tree, but code or configs differ from HEAD "
            f"({context.stamp.git_commit[:12]}). Commit first."
        )


def passed_rehearsal(context: FinalContext) -> Mapping[str, str]:
    """The row of the rehearsal that allows the formal run, after checking it (rule 1).

    Returns:
        The latest row of results/m8_rehearsal_check.csv.

    Raises:
        FinalRunRefused: If there is no rehearsal, the latest one failed, it ran on
            a dirty tree, it checked another configuration, or the code changed
            since it ran.
    """
    require_clean_tree(context, "the formal run")
    check_file = context.paths.m8_rehearsal_check_file
    rows = read_rows(check_file, REHEARSAL_CHECK_COLUMNS)
    if not rows:
        raise FinalRunRefused(
            f"no rehearsal is recorded in {check_file}. Run the rehearsal first; the formal "
            "run does not start before it has passed."
        )
    latest = rows[-1]
    if latest["status"] != PASSED:
        raise FinalRunRefused(
            f"the latest rehearsal ({latest['checked_at']}) did NOT pass: "
            f"{latest['failed_checks']}. The formal run does not start; a human decides."
        )
    locked = context.locked
    expected = (context.config.rehearsal_target_fold, locked.config_id, str(locked.num_trees))
    recorded = (latest["target_fold"], latest["config_id"], latest["num_trees"])
    if recorded != expected:
        raise FinalRunRefused(
            f"the latest rehearsal checked (fold, config, trees) = {recorded}, but the locked "
            f"configuration is {expected}"
        )
    if latest["git_dirty"] != "False" or context.code_changed(latest["git_commit"]):
        raise FinalRunRefused(
            f"the latest rehearsal ran at commit {latest['git_commit'][:12]} "
            f"(dirty: {latest['git_dirty']}), and the code or configs at HEAD "
            f"({context.stamp.git_commit[:12]}) are not the same. Run the rehearsal again "
            "with this code."
        )
    return latest


def rehearsal_sample_sha256(rehearsal: Mapping[str, str]) -> dict[str, dict[str, str]]:
    """Experiment -> {fold -> sha256} of the sample files the passed rehearsal trained on."""
    return json.loads(rehearsal["sample_sha256_json"])


def _permit(
    context: FinalContext, rehearsal: Mapping[str, str], first_read: Mapping[str, str]
) -> FormalRunPermit:
    """The permit of this process (the ONLY place a FormalRunPermit is created)."""
    return FormalRunPermit(
        git_commit=context.stamp.git_commit,
        rehearsal_commit=rehearsal["git_commit"],
        rehearsal_checked_at=rehearsal["checked_at"],
        first_label_read_at=first_read["started_at"],
    )


def _require_same_code_as_start(context: FinalContext, first_read: Mapping[str, str]) -> None:
    """Rule 3: the formal run continues only with the code it started with."""
    if context.code_changed(first_read["git_commit"]):
        raise FinalRunRefused(
            f"the formal run started at {first_read['started_at']} with commit "
            f"{first_read['git_commit'][:12]}; labels of T folds have been read since then. "
            f"The code or configs at HEAD ({context.stamp.git_commit[:12]}) differ from that "
            "commit. The formal run is done once and is not continued with changed code; "
            "a human decides."
        )


def begin_formal_run(context: FinalContext, now: str) -> FormalRunPermit:
    """Open the gate for the driver: check the rehearsal and record the start (rules 1-4).

    The first call appends the "first_t_label_read" row; a later call (the run was
    interrupted and is continued) leaves it untouched and checks rule 3.

    Args:
        context: The context of this process.
        now: The current time as ISO text; written to the record on the first call.

    Raises:
        FinalRunRefused: If any rule is violated. Nothing was read or written then.
    """
    rehearsal = passed_rehearsal(context)
    access_file = context.paths.t_phase_label_access_file
    rows = read_rows(access_file, LABEL_ACCESS_COLUMNS)
    if rows:
        _require_same_code_as_start(context, rows[0])
        return _permit(context, rehearsal, rows[0])
    first_read = {
        "event": FIRST_T_LABEL_READ,
        "started_at": now,
        "git_commit": context.stamp.git_commit,
        "git_dirty": str(context.stamp.git_dirty),
        "rehearsal_commit": rehearsal["git_commit"],
        "rehearsal_checked_at": rehearsal["checked_at"],
    }
    append_row(access_file, LABEL_ACCESS_COLUMNS, first_read)
    return _permit(context, rehearsal, first_read)


def formal_run_permit(context: FinalContext) -> FormalRunPermit:
    """The permit of a worker process of the formal run (rules 1, 3 and 4).

    A worker never records the start itself: the driver must have done it
    (begin_formal_run), so that the record always exists before a label is read.

    Raises:
        FinalRunRefused: If the rehearsal did not pass, the driver has not
            recorded the start of the formal run, or the code changed since.
    """
    rehearsal = passed_rehearsal(context)
    access_file = context.paths.t_phase_label_access_file
    rows = read_rows(access_file, LABEL_ACCESS_COLUMNS)
    if not rows:
        raise FinalRunRefused(
            f"the start of the formal run is not recorded in {access_file}. Start it with "
            "scripts/m8_final_eval.py --stage formal, which records it before any step runs."
        )
    _require_same_code_as_start(context, rows[0])
    return _permit(context, rehearsal, rows[0])
