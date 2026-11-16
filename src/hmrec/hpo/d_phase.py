"""Glue for the D-phase recall experiments (proposal §5, M2).

Loads the four D folds once, then runs hmrec.hpo.runner.run_channel for each
requested channel and writes:
    results/d_phase.csv                  one row per trial (appended, resumable)
    results/d_phase_steps/{channel}.csv  every step: all candidates, pass / fail,
                                         per-fold, pooled and group scores, decision
    artifacts/hpo/D/{channel}/{trial_id}/{fold}.parquet
                                         R2 / R3 candidates with in_active_pool
                                         (for the M4 check, proposal §5.7 item 4)
Nothing is written to configs/locks/: the selected configs are only reported.
"""

import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import polars as pl

from hmrec.config import (
    BaseConfig,
    CatalogConfig,
    PreprocessConfig,
    SelectionRule,
    load_yaml,
)
from hmrec.data.catalog import build_active_pool
from hmrec.data.folds import Fold
from hmrec.data.metadata import scan_processed_parquet
from hmrec.eval.report import FoldEvaluator
from hmrec.hpo.grid import PhaseGrid, RawConfig
from hmrec.hpo.records import TrialRecord, append_record, read_records, result_columns
from hmrec.hpo.runner import (
    FoldContext,
    StepReport,
    TrialSettings,
    check_tuning_folds,
    evaluate_trial,
    git_state,
    make_evaluate_fn,
    run_channel,
    warn_uncommitted_changes,
)
from hmrec.logging_utils import get_logger
from hmrec.paths import ProjectPaths
from hmrec.recall.base import RecallInputs

PHASE = "D"
LOGGER = get_logger("hpo.d_phase")


def load_fold_contexts(
    paths: ProjectPaths,
    folds: Sequence[Fold],
    preprocess: PreprocessConfig,
    catalog: CatalogConfig,
) -> list[FoldContext]:
    """Queries, channel inputs and label evaluator of every fold (labels stay inside eval/)."""
    check_tuning_folds(folds)
    metadata = paths.processed_metadata_file
    transactions = scan_processed_parquet(paths.transactions_file, metadata, preprocess)
    customers = (
        scan_processed_parquet(paths.customers_file, metadata, preprocess)
        .select("customer_idx", "age_bucket")
        .collect()
    )
    articles = (
        scan_processed_parquet(paths.articles_file, metadata, preprocess)
        .select("article_idx", "product_code")
        .collect()
    )
    contexts: list[FoldContext] = []
    for fold in folds:
        active_pool = build_active_pool(
            transactions, fold.anchor, preprocess.day_zero, catalog.active_pool
        )
        inputs = RecallInputs(
            fold.anchor, preprocess.day_zero, transactions, customers, articles, active_pool
        )
        queries = pl.read_parquet(paths.queries_file(fold.name))
        evaluator = FoldEvaluator.from_file(fold.name, paths.labels_file(fold.name))
        contexts.append(FoldContext(fold, queries, inputs, evaluator))
        LOGGER.info(
            "fold %s: anchor %s, %d queries, active pool %d",
            fold.name,
            fold.anchor,
            queries.height,
            active_pool.height,
        )
    return contexts


# ---------------------------------------------------------------- step report


def _record_columns(record: TrialRecord, prefix: str = "") -> dict[str, Any]:
    """Scores of one trial for the step report."""
    row: dict[str, Any] = {f"{prefix}trial_id": record.trial_id}
    row["params_json"] = json.dumps(record.params, sort_keys=True)
    for fold, score in record.summary.fold_scores.items():
        row[f"score_{fold}"] = score
    row["score_pooled"] = record.summary.pooled
    for group, score in record.summary.group_scores.items():
        row[f"score_{group}"] = score
    return row


def step_report_rows(reports: Sequence[StepReport]) -> list[dict[str, Any]]:
    """One row per (step, trial): the default first, then candidates in priority order."""
    rows: list[dict[str, Any]] = []
    for report in reports:
        decision = report.decision
        common = {
            "channel": report.channel,
            "step": report.step,
            "applied": report.applied,
            "action": decision.action if decision else "skipped",
            "reason": decision.reason if decision else "step requires did not match",
        }
        rows.append({**common, "role": "default", "priority": 0, **_record_columns(report.default)})
        rows[-1]["chosen"] = decision is None or decision.winner_trial_id == report.default.trial_id
        checks = decision.checks if decision else ()
        for priority, (record, check) in enumerate(zip(report.candidates, checks, strict=True), 1):
            row = {**common, "role": "candidate", "priority": priority, **_record_columns(record)}
            row["relative_improvement"] = check.relative_improvement
            row["folds_better"] = check.folds_better
            row["passes"] = check.passes
            row["chosen"] = decision is not None and decision.winner_trial_id == record.trial_id
            rows.append(row)
    return rows


def write_step_report(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Write (replace) results/d_phase_steps/{channel}.csv; it is derived from d_phase.csv."""
    columns: list[str] = []
    for row in rows:
        columns.extend(column for column in row if column not in columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------- run


@dataclass(frozen=True)
class ChannelOutcome:
    """Result of one channel's D phase.

    Attributes:
        channel: Channel name.
        reports: One StepReport per step.
        selected_config: The config chosen by the rule (NOT written to configs/locks/).
    """

    channel: str
    reports: list[StepReport]
    selected_config: RawConfig


@dataclass(frozen=True)
class RunContext:
    """What every channel of one D-phase run shares.

    Attributes:
        paths: Project paths.
        contexts: The four D folds.
        grid: configs/hpo/d_phase.yaml.
        rule: configs/selection_rule.yaml.
        base: configs/base.yaml.
        columns: Columns of results/d_phase.csv.
        git_commit: HEAD when the run started.
        git_dirty: Uncommitted changes when the run started.
    """

    paths: ProjectPaths
    contexts: tuple[FoldContext, ...]
    grid: PhaseGrid
    rule: SelectionRule
    base: BaseConfig
    columns: tuple[str, ...]
    git_commit: str
    git_dirty: bool


def _run_one_channel(run: RunContext, channel: str) -> ChannelOutcome:
    """Run one channel's steps, reusing trials already in results/d_phase.csv."""
    paths = run.paths
    default_raw = load_yaml(paths.recall_config_file(channel))
    settings = TrialSettings(
        phase=PHASE,
        metric=run.rule.selection_metric[channel],
        extra_metrics=run.grid.extra_metrics,
        fold_groups=run.grid.fold_groups,
        base=run.base,
        git_commit=run.git_commit,
        git_dirty=run.git_dirty,
        flags_dir=partial(paths.hpo_trial_dir, PHASE),
    )
    folds = [context.fold.name for context in run.contexts]
    records = read_records(
        paths.d_phase_results_file,
        run.columns,
        folds,
        list(run.grid.fold_groups),
        run.grid.extra_metrics,
    )
    finished = {r.trial_id: r for r in records if r.channel == channel and r.phase == PHASE}
    run_trial = partial(
        _evaluate,
        channel=channel,
        default_raw=default_raw,
        contexts=run.contexts,
        settings=settings,
    )
    save = partial(append_record, paths.d_phase_results_file, columns=run.columns)
    evaluate = make_evaluate_fn(channel, finished, run_trial, save)
    reports, selected = run_channel(
        channel, run.grid.channels[channel], default_raw, evaluate, run.rule
    )
    write_step_report(paths.d_phase_steps_file(channel), step_report_rows(reports))
    LOGGER.info("%s selected config (not locked): %s", channel, selected)
    return ChannelOutcome(channel, reports, selected)


def _evaluate(
    step: str,
    raw_config: RawConfig,
    *,
    channel: str,
    default_raw: RawConfig,
    contexts: Sequence[FoldContext],
    settings: TrialSettings,
) -> TrialRecord:
    """evaluate_trial with the arguments in the order make_evaluate_fn expects."""
    return evaluate_trial(channel, step, raw_config, default_raw, contexts, settings)


def run_d_phase(
    paths: ProjectPaths,
    channels: Sequence[str],
    contexts: Sequence[FoldContext],
    grid: PhaseGrid,
    rule: SelectionRule,
    base: BaseConfig,
) -> list[ChannelOutcome]:
    """Run the D-phase steps of each channel; see the module docstring for the outputs."""
    unknown = sorted(set(channels) - set(grid.channels))
    if unknown:
        raise ValueError(f"channels {unknown} are not in configs/hpo/d_phase.yaml")
    folds = [context.fold.name for context in contexts]
    columns = result_columns(folds, list(grid.fold_groups), grid.extra_metrics)
    commit, dirty = git_state(paths.project_root, paths.results_root)
    if dirty:
        warn_uncommitted_changes(commit)
    run = RunContext(paths, tuple(contexts), grid, rule, base, tuple(columns), commit, dirty)
    outcomes = [_run_one_channel(run, channel) for channel in channels]
    if dirty:
        warn_uncommitted_changes(commit)  # again at the end, where it is seen
    return outcomes
