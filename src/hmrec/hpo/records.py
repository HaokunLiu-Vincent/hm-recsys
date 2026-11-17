"""One trial's result and its row in results/d_phase.csv (proposal §4.4).

A trial = one channel configuration evaluated on every fold of the phase.
results/d_phase.csv gets one row per trial, appended as soon as the trial
finishes, so an interrupted run can be resumed (completed trial_ids are skipped).

Columns:
    phase, channel, step, trial_id, variant, params_json, config_json, metric,
    score_{fold} and n_queries_{fold} for every fold,
    score_pooled (query-weighted, the primary score), score_equal_fold,
    score_{group} for every diagnostic fold group (e.g. score_D1_D2),
    {metric}_pooled for every extra metric (e.g. ndcg@12_pooled),
    seconds, peak_rss_gb, git_commit, git_dirty, finished_at, diagnostics_json
Floats are written with repr(), so reading them back gives the same value.
"""

import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hmrec.eval.report import MultiFoldSummary
from hmrec.hpo.selection import TrialScores

LEADING_COLUMNS = (
    "phase",
    "channel",
    "step",
    "trial_id",
    "variant",
    "params_json",
    "config_json",
    "metric",
)
TRAILING_COLUMNS = (
    "seconds",
    "peak_rss_gb",
    "git_commit",
    "git_dirty",
    "finished_at",
    "diagnostics_json",
)


@dataclass(frozen=True)
class TrialRecord:
    """Everything recorded about one trial.

    Attributes:
        phase: "D" (or "V" later).
        channel: e.g. "r1_popularity".
        step: The step in which the trial first ran ("default" for the start config).
        trial_id: Hash of the channel and its parsed config (hpo.runner.make_trial_id).
        config: The full raw config (as in configs/recall/*.yaml).
        params: Keys that differ from the channel's configs/recall/*.yaml.
        metric: Selection metric, e.g. "recall@100".
        summary: Per-fold, pooled, equal-fold and group scores of `metric`.
        extra_pooled: Extra metric -> pooled score.
        seconds: Wall-clock time of the trial (all folds).
        peak_rss_gb: Peak process RSS during the trial (MemoryTracker, a lower bound).
        git_commit: HEAD commit when the trial ran.
        git_dirty: True if the working tree had uncommitted changes.
        finished_at: ISO timestamp.
        diagnostics: Fold -> channel-specific numbers (e.g. empty-query share).
    """

    phase: str
    channel: str
    step: str
    trial_id: str
    config: dict[str, Any]
    params: dict[str, Any]
    metric: str
    summary: MultiFoldSummary
    extra_pooled: dict[str, float]
    seconds: float
    peak_rss_gb: float
    git_commit: str
    git_dirty: bool
    finished_at: str
    diagnostics: dict[str, Any]

    def scores(self) -> TrialScores:
        """The numbers the selection rule needs."""
        return TrialScores(self.trial_id, dict(self.summary.fold_scores), self.summary.pooled)


def result_columns(
    folds: Sequence[str], groups: Sequence[str], extra_metrics: Sequence[str]
) -> list[str]:
    """Column names of results/d_phase.csv for these folds, groups and metrics."""
    fold_columns = [f"score_{fold}" for fold in folds] + [f"n_queries_{fold}" for fold in folds]
    summary_columns = ["score_pooled", "score_equal_fold"] + [f"score_{group}" for group in groups]
    extra_columns = [f"{metric}_pooled" for metric in extra_metrics]
    return [*LEADING_COLUMNS, *fold_columns, *summary_columns, *extra_columns, *TRAILING_COLUMNS]


def _json(value: Any) -> str:
    """Stable JSON text (sorted keys) for config / params / diagnostics columns."""
    # default=str: R6 configs hold a date (base_window.end_anchor).
    return json.dumps(value, sort_keys=True, default=str)


def record_to_row(record: TrialRecord) -> dict[str, str]:
    """Flatten a record into one CSV row (all values as text)."""
    summary = record.summary
    row = {
        "phase": record.phase,
        "channel": record.channel,
        "step": record.step,
        "trial_id": record.trial_id,
        "variant": str(record.config.get("variant", "")),
        "params_json": _json(record.params),
        "config_json": _json(record.config),
        "metric": record.metric,
        "score_pooled": repr(summary.pooled),
        "score_equal_fold": repr(summary.equal_fold),
        "seconds": repr(record.seconds),
        "peak_rss_gb": repr(record.peak_rss_gb),
        "git_commit": record.git_commit,
        "git_dirty": str(record.git_dirty),
        "finished_at": record.finished_at,
        "diagnostics_json": _json(record.diagnostics),
    }
    for fold, score in summary.fold_scores.items():
        row[f"score_{fold}"] = repr(score)
        row[f"n_queries_{fold}"] = str(summary.fold_queries[fold])
    for group, score in summary.group_scores.items():
        row[f"score_{group}"] = repr(score)
    for metric, score in record.extra_pooled.items():
        row[f"{metric}_pooled"] = repr(score)
    return row


def row_to_record(
    row: dict[str, str], folds: Sequence[str], groups: Sequence[str], extra_metrics: Sequence[str]
) -> TrialRecord:
    """Rebuild a TrialRecord from a CSV row written by record_to_row."""
    summary = MultiFoldSummary(
        fold_scores={fold: float(row[f"score_{fold}"]) for fold in folds},
        fold_queries={fold: int(row[f"n_queries_{fold}"]) for fold in folds},
        pooled=float(row["score_pooled"]),
        equal_fold=float(row["score_equal_fold"]),
        group_scores={group: float(row[f"score_{group}"]) for group in groups},
    )
    return TrialRecord(
        phase=row["phase"],
        channel=row["channel"],
        step=row["step"],
        trial_id=row["trial_id"],
        config=json.loads(row["config_json"]),
        params=json.loads(row["params_json"]),
        metric=row["metric"],
        summary=summary,
        extra_pooled={metric: float(row[f"{metric}_pooled"]) for metric in extra_metrics},
        seconds=float(row["seconds"]),
        peak_rss_gb=float(row["peak_rss_gb"]),
        git_commit=row["git_commit"],
        git_dirty=row["git_dirty"] == "True",
        finished_at=row["finished_at"],
        diagnostics=json.loads(row["diagnostics_json"]),
    )


def read_records(
    path: Path,
    columns: Sequence[str],
    folds: Sequence[str],
    groups: Sequence[str],
    extra_metrics: Sequence[str],
) -> list[TrialRecord]:
    """All records in results/d_phase.csv; empty if the file does not exist yet.

    Raises:
        ValueError: If the file's header differs from `columns` (other folds,
            groups or metrics than this run).
    """
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if list(reader.fieldnames or []) != list(columns):
            raise ValueError(f"{path} has columns {reader.fieldnames}, expected {list(columns)}")
        return [row_to_record(row, folds, groups, extra_metrics) for row in reader]


def append_record(path: Path, record: TrialRecord, columns: Sequence[str]) -> None:
    """Append one row (and the header if the file is new). Existing rows are never changed."""
    is_new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        if is_new:
            writer.writeheader()
        writer.writerow(record_to_row(record))
