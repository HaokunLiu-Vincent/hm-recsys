"""The code state of a run: HEAD commit and whether the working tree differs from it.

Every result row records `git_commit` and `git_dirty` (AGENTS.md §3.4): a result
must belong to one definite commit.

This module is at the bottom layer and imports only the standard library and
hmrec.logging_utils. That matters for the Ranker (M7): its processes must NOT
import torch, because torch and LightGBM each bring their own copy of the OpenMP
runtime (libomp), and two copies in one process abort, hang or crash. Until
2026-11-18 these two functions lived in hmrec.hpo.runner, which imports the
recall channels and with them torch; hmrec.hpo.runner still re-exports them.

Inside a container image (docs/mlops.md §8.1) there is no `.git` directory, so
git cannot be asked. The image carries its commit instead: it is built only
from a clean, committed tree (`make image-build` refuses anything else; a CI
checkout is clean by construction), and the commit hash is stored in the
environment variable HMREC_IMAGE_GIT_COMMIT (and as an image label). git_state
returns that commit there. On the laptop the variable is not set and git is
asked, exactly as before.

The image also carries the hash of its code tree (hmrec.code_tree, decision of
2026-11-22): the Dockerfile stores it in IMAGE_CODE_TREE_FILE at build time. In
an image git_state computes the hash of the code again and refuses the run if
the two differ, so "not dirty" is a comparison and no longer only a property of
the build. code_tree_of_run gives the hash for the run records.
"""

import os
import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

from hmrec.code_tree import code_tree_sha256, read_recorded_code_tree
from hmrec.logging_utils import get_logger

# The name the functions logged under before they moved here, kept so that run logs
# look the same in every milestone ("hmrec.hpo.runner | WARNING | !! UNCOMMITTED ...").
LOGGER = get_logger("hpo.runner")

UNCOMMITTED_BANNER = "!" * 78

# Set by the Dockerfile from the build argument GIT_COMMIT; never set on the laptop.
IMAGE_COMMIT_VARIABLE = "HMREC_IMAGE_GIT_COMMIT"
FULL_COMMIT_HASH = re.compile(r"[0-9a-f]{40}")
# Written by the Dockerfile at build time: the hash of the code tree of the image
# (hmrec.code_tree). It lies outside the code folder and belongs to root.
IMAGE_CODE_TREE_FILE = Path("/opt/hmrec/code-tree.sha256")
# What the run records hold for the tree hash on the laptop, where git describes the code.
NOT_AN_IMAGE = "not an image"


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


def image_commit(environment: Mapping[str, str]) -> str | None:
    """The commit a container image was built from, or None outside an image.

    Args:
        environment: The environment variables of the process (os.environ).

    Returns:
        The full commit hash in HMREC_IMAGE_GIT_COMMIT, or None if the variable
        is not set (the laptop).

    Raises:
        ValueError: If the variable is set but is not a full 40-character
            lower-case hexadecimal hash. A short hash, a branch name or an empty
            value would make a run record point to no definite commit.
    """
    value = environment.get(IMAGE_COMMIT_VARIABLE)
    if value is None:
        return None
    if FULL_COMMIT_HASH.fullmatch(value) is None:
        raise ValueError(
            f"{IMAGE_COMMIT_VARIABLE}={value!r} is not a full 40-character commit hash; "
            "the image must be built with --build-arg GIT_COMMIT=$(git rev-parse HEAD)"
        )
    return value


def verified_image_code_tree(repo_root: Path, recorded_file: Path) -> str:
    """The hash of the code tree of an image, after comparing it with the one stored at build.

    Args:
        repo_root: Root of the code inside the image.
        recorded_file: The file the Dockerfile wrote at build time (IMAGE_CODE_TREE_FILE).

    Returns:
        The hash (64 hexadecimal characters); it equals the stored one.

    Raises:
        FileNotFoundError: If the stored hash or an entry of the code tree is missing.
        ValueError: If the stored value is not a sha256.
        RuntimeError: If the code under repo_root does not give the stored hash:
            a file was changed, added, removed or mounted over since the build.
    """
    recorded = read_recorded_code_tree(recorded_file)
    actual = code_tree_sha256(repo_root)
    if actual != recorded:
        raise RuntimeError(
            f"the code under {repo_root} has tree hash {actual}, but the image was built "
            f"with {recorded} ({recorded_file}). The code that runs is not the code of the "
            "image's commit (a folder mounted over the code, or a changed file). Run the "
            "image as it was built."
        )
    return actual


def image_git_state(repo_root: Path, commit: str, code_tree_file: Path) -> tuple[str, bool]:
    """The code state inside a container image: the image's commit, never dirty.

    Two things are checked before "not dirty" is returned:

      1. No `.git` directory lies next to the code. A working tree mounted over
         the image's code brings one, and then the commit of the image no longer
         describes the code.
      2. The code tree gives the hash that was stored when the image was built
         (verified_image_code_tree). This also sees a mount over a part of the
         code, which brings no `.git`.

    Args:
        repo_root: Root of the code inside the image.
        commit: The commit the image was built from (image_commit).
        code_tree_file: The file with the tree hash stored at build time.

    Raises:
        RuntimeError: If repo_root contains `.git`, or the code tree differs from
            the one the image was built with.
        FileNotFoundError: If the stored tree hash is missing.
    """
    if (repo_root / ".git").exists():
        raise RuntimeError(
            f"{IMAGE_COMMIT_VARIABLE} is set (this process runs in a container image built "
            f"from commit {commit[:12]}), but {repo_root} contains a .git directory: a "
            "working tree is mounted over the image's code, so the image's commit does not "
            "describe the code that runs. Run the image without mounting the repository "
            "over it, or run natively without the variable."
        )
    verified_image_code_tree(repo_root, code_tree_file)
    return commit, False


def code_tree_of_run(repo_root: Path) -> str:
    """The code-tree hash a run records next to its commit.

    Args:
        repo_root: Root of the code.

    Returns:
        Inside a container image: the verified tree hash of the image. On the
        laptop: NOT_AN_IMAGE, because there git_commit and git_dirty describe
        the code and no tree hash was stored by a build.

    Raises:
        RuntimeError / FileNotFoundError / ValueError: Inside an image, as
            verified_image_code_tree.
    """
    if image_commit(os.environ) is None:
        return NOT_AN_IMAGE
    return verified_image_code_tree(repo_root, IMAGE_CODE_TREE_FILE)


def git_state(repo_root: Path, results_root: Path) -> tuple[str, bool]:
    """HEAD commit, and whether the code or configs differ from HEAD.

    Inside a container image (HMREC_IMAGE_GIT_COMMIT is set, see the module
    docstring) the answer is the commit the image was built from and False
    (image_git_state), after the code tree was compared with the hash stored at
    build time; git is not called. Everything below describes the
    laptop, where the variable is not set.

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
        Inside a container image: (the image's commit, False).

    Raises:
        ValueError: Inside an image, if HMREC_IMAGE_GIT_COMMIT is not a full hash.
        RuntimeError: Inside an image, if a working tree is mounted over the code
            or the code tree is not the one the image was built with.
    """
    commit_of_image = image_commit(os.environ)
    if commit_of_image is not None:
        return image_git_state(repo_root, commit_of_image, IMAGE_CODE_TREE_FILE)
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
