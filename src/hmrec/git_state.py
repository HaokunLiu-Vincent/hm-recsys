"""The code state of a run: HEAD commit and whether the working tree differs from it.

Every result row records `git_commit` and `git_dirty` (AGENTS.md §3.4): a result
must belong to one definite commit.

This module is at the bottom layer and imports only the standard library and
hmrec.logging_utils. That matters for the Ranker (M7): its processes must NOT
import torch, because torch and LightGBM each bring their own copy of the OpenMP
runtime (libomp), and two copies in one process abort, hang or crash. Until
2026-11-18 these two functions lived in hmrec.hpo.runner, which imports the
recall channels and with them torch; hmrec.hpo.runner still re-exports them.
"""

import subprocess
from collections.abc import Sequence
from pathlib import Path

from hmrec.logging_utils import get_logger

# The name the functions logged under before they moved here, kept so that run logs
# look the same in every milestone ("hmrec.hpo.runner | WARNING | !! UNCOMMITTED ...").
LOGGER = get_logger("hpo.runner")

UNCOMMITTED_BANNER = "!" * 78


def warn_uncommitted_changes(commit: str) -> None:
    """Log a hard-to-miss warning: results will not match any commit exactly.

    Does not stop the run (AGENTS.md: the user commits). Rows record
    git_dirty=True, so they can be found later.
    """
    LOGGER.warning(UNCOMMITTED_BANNER)
    LOGGER.warning("!! UNCOMMITTED CHANGES in the working tree (HEAD = %s).", commit[:12])
    LOGGER.warning("!! Results are recorded with git_dirty=True: the code that produced them")
    LOGGER.warning("!! is HEAD plus uncommitted edits. Commit first for reproducible results.")
    LOGGER.warning(UNCOMMITTED_BANNER)


def git_state(repo_root: Path, results_root: Path) -> tuple[str, bool]:
    """HEAD commit, and whether the code or configs differ from HEAD.

    "Dirty" means any tracked change or untracked file in the working tree,
    EXCEPT under results_root. results/ holds this runner's own output: after
    one channel finishes, results/d_phase.csv is new or modified, and that must
    not mark the next channel's trials as git_dirty (the code is still HEAD).
    data/ and artifacts/ are in .gitignore, so git never reports them anyway.
    Everything else (src/, configs/, scripts/, docs/, pyproject.toml, ...) counts.

    Args:
        repo_root: Root of the git repository.
        results_root: Folder of committed result tables; must be inside repo_root.

    Returns:
        (HEAD commit hash, True if there are changes outside results_root).
    """
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=True
    ).stdout.strip()
    # A git "exclude" pathspec: list changes in the whole tree ("."), minus results_root.
    results_relative = results_root.resolve().relative_to(repo_root.resolve()).as_posix()
    status_command = ["git", "status", "--porcelain", "--", ".", f":(exclude){results_relative}"]
    status = subprocess.run(
        status_command, cwd=repo_root, capture_output=True, text=True, check=True
    ).stdout
    return commit, bool(status.strip())


# What "the code" is when two commits are compared: everything a result depends on.
# docs/ and results/ are left out on purpose: committing result tables or notes
# between two runs does not change what the code computes.
CODE_PATHS = ("src", "scripts", "configs", "pyproject.toml", "uv.lock")


def code_changed_since(repo_root: Path, earlier_commit: str) -> bool:
    """True if the code or configs at HEAD differ from those at `earlier_commit`.

    Compares the committed trees of CODE_PATHS (`git diff --quiet`). Uncommitted
    changes are NOT seen here; git_state reports those.

    Args:
        repo_root: Root of the git repository.
        earlier_commit: A full commit hash recorded by an earlier run.

    Raises:
        ValueError: If git cannot compare the two commits (e.g. the recorded
            commit does not exist in this repository).
    """
    command = ["git", "diff", "--quiet", earlier_commit, "HEAD", "--", *CODE_PATHS]
    result = subprocess.run(command, cwd=repo_root, capture_output=True, text=True, check=False)
    # git diff --quiet: 0 = no difference, 1 = differences, anything else = an error.
    if result.returncode not in (0, 1):
        raise ValueError(
            f"git cannot compare {earlier_commit!r} with HEAD: {result.stderr.strip()}"
        )
    return result.returncode == 1


def changed_existing_files(repo_root: Path, earlier_commit: str, paths: Sequence[str]) -> list[str]:
    """Files under `paths` that existed at `earlier_commit` and are changed or gone at HEAD.

    A file ADDED since `earlier_commit` is not listed: new files are allowed. A
    renamed file counts as deleted (renames are not followed), so moving a file
    away is seen. Compares committed trees only, like code_changed_since.

    Args:
        repo_root: Root of the git repository.
        earlier_commit: A full commit hash recorded by an earlier run.
        paths: Files or folders relative to repo_root, e.g. ("configs", "src/hmrec/final").

    Returns:
        One entry per changed file, "<status> <path>" with git's status letter
        (M = modified, D = deleted, T = type changed), in git's order. Empty if
        every file that existed then is unchanged.

    Raises:
        ValueError: If git cannot compare the two commits.
    """
    command = ["git", "diff", "--name-status", "--no-renames", earlier_commit, "HEAD", "--"]
    result = subprocess.run(
        [*command, *paths], cwd=repo_root, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise ValueError(
            f"git cannot compare {earlier_commit!r} with HEAD: {result.stderr.strip()}"
        )
    changed: list[str] = []
    for line in result.stdout.splitlines():
        status, _, name = line.partition("\t")
        if status != "A":  # "A" = added since the earlier commit, which is allowed
            changed.append(f"{status} {name}")
    return changed
