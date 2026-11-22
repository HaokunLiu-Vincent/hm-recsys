"""Where the monitoring rules get their numbers from: the state and the bootstrap tables.

A rule compares a week with "the previous 4 weeks" (hmrec.weekly.monitoring). The
weeks before a judged week are looked up one by one, going back in time, and the
lookup STOPS at the first week that has no value: a gap ends the history, so a
rule never averages over weeks that are not the ones right before.

Where a week's value comes from:

    a week before the first tick    the bootstrap state (data/bootstrap/monitoring/):
                                    the F1-F4 rows of results/m6_candidates.csv
                                    (recall of each channel, candidates per query)
                                    and results/m6_features.csv (missing rate and
                                    mean of every feature). Stage 1 ranked no F
                                    week, so the two output signals that need a
                                    ranking have no value there.
    a week of the pipeline          the files its tick wrote in the state.

`eval` and `serve` weeks are different populations (buyers of the week; every
customer), so the same-week numbers of one mode are never the history of the
other: a week of the other mode has no value. The bootstrap weeks are `eval`
weeks.

Findings are kept in the state (weeks/{anchor}/monitoring.json, written once per
tick) and appended to results/m12_replay/monitoring.csv.

This module imports neither torch nor lightgbm: the driver calls it.
"""

import csv
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path

from hmrec.config import MonitoringConfig
from hmrec.csv_records import append_row
from hmrec.paths import SHARD_TOP_DIR, shard_step_dir
from hmrec.run_manifest import EVAL_MODE
from hmrec.weekly.calendar import Week, previous_week
from hmrec.weekly.context import TickContext
from hmrec.weekly.monitoring import LABELLED, FeatureSummary, Finding, raised_flags
from hmrec.weekly.outputs import read_signals, read_week_evaluation
from hmrec.weekly.shard_features import FEATURE_STATS_FILE
from hmrec.weekly.storage import StateStore

BOOTSTRAP_CANDIDATES_TABLE = "m6_candidates.csv"
BOOTSTRAP_FEATURES_TABLE = "m6_features.csv"
UNION_SCOPE = "union"
CANDIDATES_PER_QUERY = "candidates_per_query"
TOP_POPULAR_SHARE = "top_popular_share"
DISTINCT_ARTICLES = "distinct_articles"
OUTPUT_SIGNALS = (CANDIDATES_PER_QUERY, TOP_POPULAR_SHARE, DISTINCT_ARTICLES)

MONITORING_COLUMNS: tuple[str, ...] = (
    "run_id",
    "tick_anchor",
    "mode",
    "layer",
    "rule",
    "subject",
    "week",
    "value",
    "reference",
    "n_history",
    "limit",
    "flag",
    "consecutive_weeks",
    "alert",
    "git_commit",
    "recorded_at",
)
NOT_COMPUTED = ""


# ---------------------------------------------------------------- reading tables


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """The rows of a CSV file with a header; empty if the file does not exist."""
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def feature_summaries(rows: Sequence[dict[str, str]]) -> dict[str, FeatureSummary]:
    """Feature -> (missing rate, mean) from rows with the columns of results/m6_features.csv.

    A feature whose missing rate was not computed (no rows) is left out.
    """
    summaries: dict[str, FeatureSummary] = {}
    for row in rows:
        if row["missing_rate"] == NOT_COMPUTED:
            continue
        mean = None if row["mean"] == NOT_COMPUTED else float(row["mean"])
        summaries[row["feature"]] = FeatureSummary(float(row["missing_rate"]), mean)
    return summaries


def merged_feature_summaries(
    tables: Sequence[Sequence[dict[str, str]]],
) -> dict[str, FeatureSummary]:
    """The summaries of a week from the statistics of its shards.

    missing rate = missing rows / rows over all shards; mean = the shards' means
    weighted by their present rows (float64).

    Args:
        tables: One list of rows per shard (columns of results/m6_features.csv).
    """
    rows_by_feature: dict[str, list[dict[str, str]]] = {}
    for table in tables:
        for row in table:
            rows_by_feature.setdefault(row["feature"], []).append(row)
    merged: dict[str, FeatureSummary] = {}
    for feature, rows in rows_by_feature.items():
        n_rows = sum(int(row["n_rows"]) for row in rows)
        n_missing = sum(int(row["n_missing"]) for row in rows)
        if n_rows == 0:
            continue
        with_mean = [row for row in rows if row["mean"] != NOT_COMPUTED]
        present = [int(row["n_rows"]) - int(row["n_missing"]) for row in with_mean]
        mean = None
        if with_mean and sum(present) > 0:
            total = math.fsum(
                float(row["mean"]) * n for row, n in zip(with_mean, present, strict=True)
            )
            mean = total / sum(present)
        merged[feature] = FeatureSummary(n_missing / n_rows, mean)
    return merged


# ---------------------------------------------------------------- one week's values


def week_mode(context: TickContext, week: Week) -> str | None:
    """The mode a pipeline week was opened in; None if its tick never started."""
    tick_file = context.layout.week_dir(week.anchor) / "tick.json"
    if not tick_file.is_file():
        return None
    return str(json.loads(tick_file.read_text(encoding="utf-8"))["mode"])


def is_bootstrap_week(context: TickContext, week: Week) -> bool:
    """True for a week before the first tick."""
    return week.ordinal < context.first_week.ordinal


def comparable(context: TickContext, week: Week) -> bool:
    """True if `week` has the population of the judged tick's mode (module docstring)."""
    if is_bootstrap_week(context, week):
        return context.mode == EVAL_MODE
    return week_mode(context, week) == context.mode


def week_feature_summaries(context: TickContext, week: Week) -> dict[str, FeatureSummary] | None:
    """The feature summaries of a week; None if it has none for this tick's mode."""
    if not comparable(context, week):
        return None
    if is_bootstrap_week(context, week):
        rows = read_csv_rows(context.paths.bootstrap_monitoring_file(BOOTSTRAP_FEATURES_TABLE))
        summaries = feature_summaries([row for row in rows if row["fold"] == week.name])
        return summaries or None
    return own_feature_summaries(context, week)


def own_feature_summaries(context: TickContext, week: Week) -> dict[str, FeatureSummary] | None:
    """The feature summaries a pipeline week wrote itself (`eval`: one file; `serve`: per shard)."""
    week_dir = context.layout.week_dir(week.anchor)
    whole = week_dir / "features" / FEATURE_STATS_FILE
    if whole.is_file():
        return feature_summaries(read_csv_rows(whole)) or None
    tables, shard = [], 0
    while (week_dir / shard_step_dir(shard, SHARD_TOP_DIR) / FEATURE_STATS_FILE).is_file():
        tables.append(
            read_csv_rows(week_dir / shard_step_dir(shard, SHARD_TOP_DIR) / FEATURE_STATS_FILE)
        )
        shard += 1
    return merged_feature_summaries(tables) or None


def week_output_signals(context: TickContext, week: Week) -> dict[str, float]:
    """The output signals of a week that exist for this tick's mode (maybe none)."""
    if not comparable(context, week):
        return {}
    if is_bootstrap_week(context, week):
        rows = read_csv_rows(context.paths.bootstrap_monitoring_file(BOOTSTRAP_CANDIDATES_TABLE))
        union = [row for row in rows if row["fold"] == week.name and row["scope"] == UNION_SCOPE]
        if len(union) != 1:
            return {}
        return {CANDIDATES_PER_QUERY: float(union[0]["candidates_per_query_mean"])}
    return own_output_signals(context, week)


def own_output_signals(context: TickContext, week: Week) -> dict[str, float]:
    """The output signals a pipeline week wrote itself; empty if it has not scored yet."""
    signals_file = context.layout.week_dir(week.anchor) / "scores" / "signals.json"
    if not signals_file.is_file():
        return {}
    signals = read_signals(signals_file)
    return {
        CANDIDATES_PER_QUERY: signals.candidates_per_query,
        TOP_POPULAR_SHARE: signals.top_popular_share,
        DISTINCT_ARTICLES: float(signals.distinct_articles),
    }


def week_channel_recall(context: TickContext, week: Week) -> dict[str, float]:
    """Channel -> recall of a closed week; empty if the week is not closed or has no labels."""
    if is_bootstrap_week(context, week):
        rows = read_csv_rows(context.paths.bootstrap_monitoring_file(BOOTSTRAP_CANDIDATES_TABLE))
        return {
            row["scope"]: float(row["recall"])
            for row in rows
            if row["fold"] == week.name and row["scope"] != UNION_SCOPE and row["recall"] != ""
        }
    metrics_file = context.layout.week_dir(week.anchor) / "evaluation" / "metrics.json"
    if not metrics_file.is_file():
        return {}
    return dict(read_week_evaluation(metrics_file).channel_recall)


# ---------------------------------------------------------------- histories


def weeks_back[Value](
    context: TickContext, week: Week, n_weeks: int, value_of: Callable[[Week], Value | None]
) -> list[Value]:
    """The values of the weeks right before `week`, oldest first; stops at the first gap.

    Args:
        context: The tick's context (for the calendar).
        week: The judged week.
        n_weeks: How many weeks back to look at most.
        value_of: A week's value, or None if it has none.
    """
    values: list[Value] = []
    current = week
    for _ in range(n_weeks):
        if current.ordinal == 0:
            break
        current = previous_week(current, context.manifest)
        value = value_of(current)
        if value is None:
            break
        values.append(value)
    return list(reversed(values))


def feature_history(
    context: TickContext, config: MonitoringConfig
) -> list[dict[str, FeatureSummary]]:
    """The feature summaries of the weeks before the tick's week, oldest first."""
    return weeks_back(
        context,
        context.tick,
        config.history_weeks,
        lambda week: week_feature_summaries(context, week),
    )


def output_history(context: TickContext, config: MonitoringConfig) -> dict[str, list[float]]:
    """Signal -> its values in the weeks before the tick's week, oldest first."""
    return {
        signal: weeks_back(
            context,
            context.tick,
            config.history_weeks,
            lambda week, name=signal: week_output_signals(context, week).get(name),
        )
        for signal in OUTPUT_SIGNALS
    }


def recall_history(
    context: TickContext, config: MonitoringConfig, channels: Sequence[str]
) -> dict[str, list[float]]:
    """Channel -> its recall in the weeks before the CLOSED week, oldest first."""
    return {
        channel: weeks_back(
            context,
            context.closing,
            config.history_weeks,
            lambda week, name=channel: week_channel_recall(context, week).get(name),
        )
        for channel in channels
    }


def earlier_labelled_flags(
    context: TickContext, config: MonitoringConfig
) -> list[frozenset[tuple[str, str]]]:
    """The raised labelled flags of the weeks before the closed week, NEWEST first.

    The labelled findings of week w are in the monitoring record of the tick that
    closed it, the tick of week w + 1. A week before the first tick has no
    record, so the count of consecutive weeks starts inside the pipeline.
    """

    def flags_of(week: Week) -> frozenset[tuple[str, str]] | None:
        if is_bootstrap_week(context, week):
            return None
        closed_by = context.layout.monitoring_file(week.end_exclusive)
        if not closed_by.is_file():
            return None
        return raised_flags(read_findings(closed_by), LABELLED)

    oldest_first = weeks_back(
        context, context.closing, config.consecutive_weeks_for_alert, flags_of
    )
    return list(reversed(oldest_first))


# ---------------------------------------------------------------- records


def finding_to_json(finding: Finding) -> dict[str, object]:
    """A finding as a JSON object (the week as an ISO date)."""
    return {**asdict(finding), "week": finding.week.isoformat()}


def read_findings(path: Path) -> list[Finding]:
    """Read weeks/{anchor}/monitoring.json."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [Finding(**{**item, "week": date.fromisoformat(item["week"])}) for item in raw]


def write_findings(store: StateStore, path: Path, findings: Sequence[Finding]) -> None:
    """Write the findings of a tick into the state (once)."""
    text = json.dumps([finding_to_json(finding) for finding in findings], indent=2) + "\n"
    store.write_text_once(path, text)


def finding_row(context: TickContext, finding: Finding) -> dict[str, str]:
    """A finding as a row of results/m12_replay/monitoring.csv (floats with repr())."""
    return {
        "run_id": context.run_id,
        "tick_anchor": context.tick.anchor.isoformat(),
        "mode": context.mode,
        "layer": finding.layer,
        "rule": finding.rule,
        "subject": finding.subject,
        "week": finding.week.isoformat(),
        "value": repr(finding.value),
        "reference": NOT_COMPUTED if finding.reference is None else repr(finding.reference),
        "n_history": str(finding.n_history),
        "limit": finding.limit,
        "flag": str(finding.flag),
        "consecutive_weeks": str(finding.consecutive_weeks),
        "alert": str(finding.alert),
        "git_commit": context.stamp.git_commit,
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
    }


def append_findings(context: TickContext, table: Path, findings: Sequence[Finding]) -> None:
    """Append the findings of a tick to the monitoring table."""
    for finding in findings:
        append_row(table, MONITORING_COLUMNS, finding_row(context, finding))
