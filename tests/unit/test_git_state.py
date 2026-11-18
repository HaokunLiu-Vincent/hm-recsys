"""Tests for hmrec.git_state.

The behaviour of git_state and warn_uncommitted_changes (clean repo, results/ is
ignored, code changes are dirty, the loud warning) is tested in test_hpo_runner.py
through hmrec.hpo.runner, which re-exports them. Here: they are the same functions,
and the module (unlike hmrec.hpo.runner) does not pull in torch.
"""

import subprocess
import sys
from pathlib import Path

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
