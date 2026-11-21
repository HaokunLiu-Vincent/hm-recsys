"""Extra-seed runs of the best configurations of one step (reported only, never selects).

configs/hpo/d_phase.yaml `robustness` says, per channel, which step, how many of
its candidates (top_k), which config key holds the seed and which extra seeds
to run (proposal §5 R6 "Robustness": the top 2 update configs, 2 more seeds each).

Ranking of the step's candidates by the §4.2 rule:
    rank 1 = the candidate the rule selects against the step's default;
    rank 2 = the candidate the rule would select if rank 1 were not there; ...
    If no remaining candidate passes, the remaining ones follow by pooled score
    (highest first). The selection itself is NOT changed: it stays the seed of
    the step's own runs; the extra seeds only measure how much a score moves.
"""

import statistics
from collections.abc import Sequence
from typing import Any

from hmrec.config import SelectionRule
from hmrec.hpo.grid import RawConfig, Robustness, apply_overrides
from hmrec.hpo.records import TrialRecord
from hmrec.hpo.runner import EvaluateFn, StepReport, changed_params
from hmrec.hpo.selection import SWITCH, choose_winner

ROBUSTNESS_STEP = "seed_robustness"


def rank_candidates(
    report: StepReport, rule: SelectionRule, top_k: int
) -> list[tuple[RawConfig, TrialRecord]]:
    """The top_k (config, record) pairs of a step, ranked as in the module docstring."""
    remaining = list(zip(report.candidate_configs, report.candidates, strict=True))
    ranked: list[tuple[RawConfig, TrialRecord]] = []
    while remaining and len(ranked) < top_k:
        scores = [record.scores() for _, record in remaining]
        decision = choose_winner(report.default.scores(), scores, rule)
        if decision.action == SWITCH:
            pick = next(pair for pair in remaining if pair[1].trial_id == decision.winner_trial_id)
        else:
            pick = max(remaining, key=lambda pair: pair[1].summary.pooled)
        ranked.append(pick)
        remaining.remove(pick)
    return ranked


def find_report(reports: Sequence[StepReport], step: str) -> StepReport:
    """The report of `step`; it must have run (not skipped) and kept its configs."""
    for report in reports:
        if report.step == step:
            if not report.applied or not report.candidate_configs:
                raise ValueError(f"step {step!r} did not run, so it has no candidates to rank")
            return report
    raise ValueError(f"no report for step {step!r}")


def run_robustness(
    reports: Sequence[StepReport],
    settings: Robustness,
    evaluate: EvaluateFn,
    rule: SelectionRule,
) -> list[tuple[int, RawConfig, list[TrialRecord]]]:
    """Run every extra seed of the top_k candidates of settings.step.

    Returns:
        One entry per ranked candidate: (rank, its config, [its own record, then
        one record per extra seed]). Records are saved like any trial
        (results/d_phase.csv, step "seed_robustness") and reused on a re-run.
    """
    report = find_report(reports, settings.step)
    results: list[tuple[int, RawConfig, list[TrialRecord]]] = []
    for rank, (config, record) in enumerate(rank_candidates(report, rule, settings.top_k), 1):
        records = [record]
        for seed in settings.seeds:
            seeded = apply_overrides(config, {settings.seed_key: seed})
            records.append(evaluate(ROBUSTNESS_STEP, seeded))
        results.append((rank, config, records))
    return results


def _spread(values: Sequence[float]) -> dict[str, float]:
    """Mean, sample std (n - 1), min and max of a few scores."""
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def robustness_rows(
    results: Sequence[tuple[int, RawConfig, list[TrialRecord]]],
    step_default: RawConfig,
    seed_key: str,
) -> list[dict[str, Any]]:
    """Rows for results/d_phase_steps/{channel}_seeds.csv.

    Per ranked candidate: one row per seed (fold, pooled and group scores), then
    one row per statistic (mean / std / min / max over the seeds) of every score.
    """
    rows: list[dict[str, Any]] = []
    for rank, config, records in results:
        label = str(changed_params(step_default, config))
        score_rows = []
        for record in records:
            scores = {f"score_{fold}": value for fold, value in record.summary.fold_scores.items()}
            scores["score_pooled"] = record.summary.pooled
            for group, value in record.summary.group_scores.items():
                scores[f"score_{group}"] = value
            score_rows.append(scores)
            seed = record.config.get(seed_key)
            rows.append({"rank": rank, "config": label, "seed": seed, "row": "seed", **scores})
        for statistic in ("mean", "std", "min", "max"):
            summary = {
                column: _spread([scores[column] for scores in score_rows])[statistic]
                for column in score_rows[0]
            }
            rows.append({"rank": rank, "config": label, "seed": "", "row": statistic, **summary})
    return rows
