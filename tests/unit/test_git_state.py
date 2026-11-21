"""Tests for hmrec.git_state.

The behaviour of git_state and warn_uncommitted_changes (clean repo, results/ is
ignored, code changes are dirty, the loud warning) is tested in test_hpo_runner.py
through hmrec.hpo.runner, which re-exports them. Here: they are the same functions,
and the module (unlike hmrec.hpo.runner) does not pull in torch.

Also here: the branch for a container image (docs/mlops.md §8.1), where there is no
`.git` and the commit comes from HMREC_IMAGE_GIT_COMMIT. Those tests set the variable
themselves, so they run on the laptop too.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

import hmrec.git_state as git_state_module
import hmrec.hpo.runner as runner
from hmrec.git_state import IMAGE_COMMIT_VARIABLE

REPO = Path(__file__).resolve().parents[2]


def test_runner_re_exports_the_same_functions() -> None:
    assert runner.git_state is git_state_module.git_state
    assert runner.warn_uncommitted_changes is git_state_module.warn_uncommitted_changes
    assert runner.UNCOMMITTED_BANNER == git_state_module.UNCOMMITTED_BANNER
    # The warning keeps the logger name it always had, so run logs look the same.
    assert git_state_module.LOGGER is runner.LOGGER


def test_git_state_of_this_repository() -> None:
    commit, dirty = git_state_module.git_state(REPO, REPO / "results")
    assert len(commit) == 40 and set(commit) <= set("0123456789abcdef")
    assert isinstance(dirty, bool)
    if (REPO / ".git").exists():
        # The laptop (and a CI checkout): git was asked, and the image variable is not set.
        assert IMAGE_COMMIT_VARIABLE not in os.environ
        assert commit == _git(REPO, "rev-parse", "HEAD")
    else:
        # A container image: no .git; the commit is the one the image was built from.
        assert commit == os.environ[IMAGE_COMMIT_VARIABLE]
        assert dirty is False


# ---------------------------------------------------------------- inside a container image

IMAGE_COMMIT = "0123456789abcdef0123456789abcdef01234567"


def test_image_commit_is_none_on_the_laptop_and_the_hash_in_an_image() -> None:
    assert git_state_module.image_commit({}) is None
    assert git_state_module.image_commit({"PATH": "/usr/bin"}) is None
    assert git_state_module.image_commit({IMAGE_COMMIT_VARIABLE: IMAGE_COMMIT}) == IMAGE_COMMIT


@pytest.mark.parametrize(
    "value",
    [
        "",  # the build argument was not passed
        "0123456",  # a short hash
        "main",  # a branch name
        IMAGE_COMMIT.upper(),  # git prints lower case; anything else was typed by hand
        IMAGE_COMMIT + "0",
        IMAGE_COMMIT + "\n",
    ],
)
def test_image_commit_must_be_a_full_hash(value: str) -> None:
    with pytest.raises(ValueError, match="not a full 40-character commit hash"):
        git_state_module.image_commit({IMAGE_COMMIT_VARIABLE: value})


def test_git_state_in_an_image_is_the_image_commit_and_git_is_not_asked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # tmp_path is not a git repository: asking git would fail here.
    (tmp_path / "results").mkdir()
    monkeypatch.setenv(IMAGE_COMMIT_VARIABLE, IMAGE_COMMIT)

    def no_git(*arguments: object, **options: object) -> None:
        raise AssertionError("git must not be called inside a container image")

    monkeypatch.setattr(git_state_module.subprocess, "run", no_git)
    assert git_state_module.git_state(tmp_path, tmp_path / "results") == (IMAGE_COMMIT, False)


def test_git_state_in_an_image_refuses_a_mounted_working_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A .git next to the code means a working tree was mounted over the image's code:
    # the image's commit no longer says which code runs.
    (tmp_path / ".git").mkdir()
    monkeypatch.setenv(IMAGE_COMMIT_VARIABLE, IMAGE_COMMIT)
    with pytest.raises(RuntimeError, match="a working tree is mounted over the image's code"):
        git_state_module.git_state(tmp_path, tmp_path / "results")


def test_git_state_in_an_image_refuses_a_bad_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(IMAGE_COMMIT_VARIABLE, "unknown")
    with pytest.raises(ValueError, match="not a full 40-character commit hash"):
        git_state_module.git_state(tmp_path, tmp_path / "results")


def test_git_state_without_the_variable_asks_git_as_before(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The laptop branch, also when this test itself runs inside an image.
    monkeypatch.delenv(IMAGE_COMMIT_VARIABLE, raising=False)
    _git(tmp_path, "init", "-q")
    head = _commit(tmp_path, "src/model.py", "x = 1\n")
    (tmp_path / "results").mkdir()
    assert git_state_module.git_state(tmp_path, tmp_path / "results") == (head, False)
    (tmp_path / "src" / "model.py").write_text("x = 2\n", encoding="utf-8")
    assert git_state_module.git_state(tmp_path, tmp_path / "results") == (head, True)


def test_importing_git_state_does_not_import_torch_or_polars() -> None:
    code = (
        "import sys, hmrec.git_state; "
        "print('torch' in sys.modules, 'polars' in sys.modules, 'lightgbm' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=REPO
    )
    assert result.stdout.split() == ["False", "False", "False"]


# ---------------------------------------------------------------- code_changed_since


def _git(repo: Path, *arguments: str) -> str:
    command = ["git", "-c", "user.name=test", "-c", "user.email=test@example.com", *arguments]
    result = subprocess.run(command, cwd=repo, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def _commit(repo: Path, relative_path: str, content: str) -> str:
    """Write a file, commit it and return the new commit hash."""
    path = repo / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    _git(repo, "add", relative_path)
    _git(repo, "commit", "-m", f"change {relative_path}")
    return _git(repo, "rev-parse", "HEAD")


def test_code_changed_since_sees_code_and_configs_but_not_results_or_docs(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    start = _commit(tmp_path, "src/model.py", "x = 1\n")
    assert not git_state_module.code_changed_since(tmp_path, start)
    # Result tables and notes committed after a run do not change what the code computes.
    _commit(tmp_path, "results/t_phase.csv", "fold,value\n")
    _commit(tmp_path, "docs/decisions.md", "a note\n")
    assert not git_state_module.code_changed_since(tmp_path, start)
    # A config, a script, the code or the locked dependencies do.
    for relative_path in ("configs/m8_final.yaml", "scripts/run.py", "src/model.py", "uv.lock"):
        before = _git(tmp_path, "rev-parse", "HEAD")
        _commit(tmp_path, relative_path, f"changed {relative_path}\n")
        assert git_state_module.code_changed_since(tmp_path, before), relative_path
        assert git_state_module.code_changed_since(tmp_path, start)
    assert not git_state_module.code_changed_since(tmp_path, _git(tmp_path, "rev-parse", "HEAD"))


def test_code_changed_since_fails_on_a_commit_git_does_not_know(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    _commit(tmp_path, "src/model.py", "x = 1\n")
    with pytest.raises(ValueError, match="git cannot compare"):
        git_state_module.code_changed_since(tmp_path, "0" * 40)


def test_changed_existing_files_allows_new_files_only(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    _commit(tmp_path, "configs/m8_final.yaml", "goals: 1\n")
    _commit(tmp_path, "src/hmrec/final/gate.py", "x = 1\n")
    start = _commit(tmp_path, "scripts/m8_step.py", "y = 1\n")
    protected = ("configs", "src/hmrec/final", "scripts/m8_step.py")
    assert git_state_module.changed_existing_files(tmp_path, start, protected) == []
    # NEW files under the protected paths, and any change outside them, are allowed.
    _commit(tmp_path, "configs/m8_diagnostics.yaml", "seed: 42\n")
    _commit(tmp_path, "src/hmrec/final/new_module.py", "z = 1\n")
    _commit(tmp_path, "src/hmrec/diagnostics/run.py", "w = 1\n")
    _commit(tmp_path, "scripts/m8_diagnostics.py", "v = 1\n")
    assert git_state_module.changed_existing_files(tmp_path, start, protected) == []
    # A file that existed then and is modified, deleted or moved away is reported.
    _commit(tmp_path, "configs/m8_final.yaml", "goals: 2\n")
    _git(tmp_path, "rm", "-q", "scripts/m8_step.py")
    _git(tmp_path, "mv", "src/hmrec/final/gate.py", "src/hmrec/final/renamed.py")
    _git(tmp_path, "commit", "-q", "-m", "delete and rename")
    assert git_state_module.changed_existing_files(tmp_path, start, protected) == [
        "M configs/m8_final.yaml",
        "D scripts/m8_step.py",
        "D src/hmrec/final/gate.py",
    ]


def test_changed_existing_files_fails_on_a_commit_git_does_not_know(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    _commit(tmp_path, "src/model.py", "x = 1\n")
    with pytest.raises(ValueError, match="git cannot compare"):
        git_state_module.changed_existing_files(tmp_path, "0" * 40, ("src",))
