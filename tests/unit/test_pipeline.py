"""Tests for hmrec.pipeline: step order, one process per step, the fold summary line."""

import os
import subprocess
from pathlib import Path

import pytest

from hmrec.locks import ALL_CHANNELS
from hmrec.pipeline import MONITOR_STEP, fold_steps, fold_summary, run_script


def test_steps_are_the_six_channels_in_order_then_the_monitor() -> None:
    steps = fold_steps("F2", Path("generate.py"), Path("monitor.py"))
    assert [step for step, _, _ in steps] == [*ALL_CHANNELS, MONITOR_STEP]
    # R6 (which fine-tunes) is the last channel; the monitor reads all six files after it.
    assert steps[5] == (
        "r6_twotower",
        Path("generate.py"),
        ["--fold", "F2", "--channel", "r6_twotower"],
    )
    assert steps[6] == (MONITOR_STEP, Path("monitor.py"), ["--fold", "F2"])


def test_run_script_runs_a_separate_process_and_fails_on_a_non_zero_exit(tmp_path: Path) -> None:
    marker = tmp_path / "pid.txt"
    script = tmp_path / "step.py"
    script.write_text(
        "import os, sys\n"
        "open(sys.argv[1], 'w').write(str(os.getpid()))\n"
        "sys.exit(int(sys.argv[2]))\n",
        encoding="utf-8",
    )
    seconds = run_script(script, [str(marker), "0"])
    assert seconds > 0
    assert int(marker.read_text()) != os.getpid()  # really another process
    with pytest.raises(subprocess.CalledProcessError):
        run_script(script, [str(marker), "3"])


def _run(fold: str, step: str, peak: str, seconds: str) -> dict[str, str]:
    return {"fold": fold, "step": step, "process_max_rss_gb": peak, "total_seconds": seconds}


def _monitor(fold: str, scope: str, recall: str, mean: str, flag: str) -> dict[str, str]:
    return {
        "fold": fold,
        "scope": scope,
        "recall": recall,
        "candidates_per_query_mean": mean,
        "low_recall_flag": flag,
    }


def test_fold_summary_line() -> None:
    runs = [
        _run("F1", "r1_popularity", "1.98", "4.0"),
        _run("F1", "r4_itemcf", "4.2", "30.0"),
        _run("F2", "r4_itemcf", "9.9", "99.0"),  # another fold: ignored
    ]
    monitors = [
        _monitor("F1", "r1_popularity", "0.07", "100.0", "True"),
        _monitor("F1", "r4_itemcf", "0.0925", "89.5", "False"),
        _monitor("F1", "union", "0.2431", "306.94", ""),
    ]
    assert fold_summary("F1", runs, monitors) == (
        "F1 | union 306.9 cand/query, recall 24.31% | r1_popularity 7.00% r4_itemcf 9.25% | "
        "max process RSS 4.20 GB (r4_itemcf) | 34 s | low recall: r1_popularity"
    )


def test_fold_summary_of_a_t_fold_has_no_recall() -> None:
    runs = [_run("T1", "r1_popularity", "2.0", "5.0")]
    monitors = [
        _monitor("T1", "r1_popularity", "", "100.0", ""),
        _monitor("T1", "union", "", "300.0", ""),
    ]
    summary = fold_summary("T1", runs, monitors)
    assert "recall n/a" in summary and "r1_popularity n/a" in summary
    assert summary.endswith("low recall: none")
    with pytest.raises(ValueError, match="no rows yet"):
        fold_summary("T2", runs, monitors)
