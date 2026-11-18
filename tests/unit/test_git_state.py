"""Tests for hmrec.git_state.

The behaviour of git_state and warn_uncommitted_changes (clean repo, results/ is
ignored, code changes are dirty, the loud warning) is tested in test_hpo_runner.py
through hmrec.hpo.runner, which re-exports them. Here: they are the same functions,
and the module (unlike hmrec.hpo.runner) does not pull in torch.
"""

import subprocess
import sys
from pathlib import Path

import pytest

import hmrec.git_state as git_state_module
import hmrec.hpo.runner as runner

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
