"""Tests for hpo.robustness: ranking by the §4.2 rule, extra seeds, summary rows."""

import statistics
from pathlib import Path

import pytest
from test_hpo_runner import RULE, _record

from hmrec.config import load_yaml
from hmrec.hpo.grid import Robustness
from hmrec.hpo.records import TrialRecord
from hmrec.hpo.robustness import (
    ROBUSTNESS_STEP,
    find_report,
    rank_candidates,
    robustness_rows,
    run_robustness,
)
from hmrec.hpo.runner import StepReport

REPO = Path(__file__).resolve().parents[2]
R2 = load_yaml(REPO / "configs" / "recall" / "r2_repurchase.yaml")


def _report(scores_by_tau: dict[float, tuple[float, ...]]) -> StepReport:
    """A step of R2 tau candidates (priority = dict order) against a default of 0.10 everywhere."""
    default = _record("r2_repurchase", dict(R2), (0.10,) * 4, "tau_days")
    configs = [{**R2, "tau_days": tau} for tau in scores_by_tau]
    records = [
        _record("r2_repurchase", config, scores, "tau_days")
        for config, scores in zip(configs, scores_by_tau.values(), strict=True)
    ]
    return StepReport(
        "r2_repurchase", "tau_days", True, default, tuple(records), None, dict(R2), tuple(configs)
    )


def test_rank_follows_the_rule_then_the_rule_without_the_winner() -> None:
    # Against the default 0.10: tau 7 +5%, tau 14 +10%, tau 60 +10.5% (all pass on 4/4
    # weeks), tau 90 worse. Rank 1: best passing score 0.1105; within 1% of it are
    # 14 (0.110) and 60 (0.1105); 14 comes first in priority -> 14.
    # Rank 2 (without 14): best 0.1105; within 1% of it only 60 (7 is 5% below) -> 60.
    report = _report(
        {
            7.0: (0.105,) * 4,
            14.0: (0.110,) * 4,
            60.0: (0.1105,) * 4,
            90.0: (0.09,) * 4,
        }
    )
    ranked = rank_candidates(report, RULE, top_k=2)
    assert [config["tau_days"] for config, _ in ranked] == [14.0, 60.0]


def test_rank_falls_back_to_pooled_score_when_nothing_passes() -> None:
    report = _report({7.0: (0.099,) * 4, 14.0: (0.0995,) * 4, 60.0: (0.098,) * 4})
    ranked = rank_candidates(report, RULE, top_k=2)
    assert [config["tau_days"] for config, _ in ranked] == [14.0, 7.0]


def test_run_robustness_adds_seeds_and_summarises() -> None:
    report = _report({7.0: (0.105,) * 4, 14.0: (0.12,) * 4})
    calls: list[tuple[str, dict]] = []

    def evaluate(step: str, config: dict) -> TrialRecord:
        calls.append((step, config))
        # Use "lookback_days" as a stand-in seed key; the score moves with the "seed".
        score = 0.12 + 0.001 * (config["lookback_days"] - 180)
        return _record("r2_repurchase", config, (score,) * 4, step)

    settings = Robustness(step="tau_days", top_k=1, seed_key="lookback_days", seeds=(181, 182))
    results = run_robustness([report], settings, evaluate, RULE)
    assert [step for step, _ in calls] == [ROBUSTNESS_STEP, ROBUSTNESS_STEP]
    assert [config["lookback_days"] for _, config in calls] == [181, 182]
    rank, config, records = results[0]
    assert (rank, config["tau_days"], len(records)) == (1, 14.0, 3)
    rows = robustness_rows(results, dict(R2), "lookback_days")
    mean = next(row for row in rows if row["row"] == "mean")
    std = next(row for row in rows if row["row"] == "std")
    pooled = [0.12, 0.121, 0.122]  # the step's own run (180) + seeds 181, 182
    assert mean["score_pooled"] == pytest.approx(statistics.fmean(pooled))
    assert std["score_pooled"] == pytest.approx(statistics.stdev(pooled))
    assert mean["config"] == str({"tau_days": 14.0})


def test_find_report_needs_a_run_step() -> None:
    report = _report({7.0: (0.105,) * 4})
    skipped = StepReport("r2_repurchase", "lookback", False, report.default, (), None)
    with pytest.raises(ValueError, match="did not run"):
        find_report([report, skipped], "lookback")
    with pytest.raises(ValueError, match="no report"):
        find_report([report], "missing")
