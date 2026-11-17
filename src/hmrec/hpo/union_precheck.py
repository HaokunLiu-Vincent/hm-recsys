"""D-phase union pre-check of R1-R5: diagnostics only (proposal §5.7 items 1-3).

This is NOT the M4 union check (scripts/m4_union_check.py):
  - R6 is not in yet, so the union has 5 channels; G2's 24% target is for 6;
  - the R3 active-pool decision (§5.7 item 4) is NOT made here;
  - nothing is selected, changed or locked. The module only reports numbers.

Steps (run by scripts/m2_union_precheck.py):
  1. select_channels: replay each channel's D-phase steps over the trials that
     are already in results/d_phase.csv. No trial is run (a missing trial is an
     error), so this gives exactly the config the selection rule picked.
  2. load_or_generate_candidates: for each fold and channel, generate the
     candidates of the selected config and write
     artifacts/hpo/D/union_precheck/{fold}/{channel}_{trial_id}.parquet.
     An existing file is read back and never overwritten; the trial_id in the
     file name means it can only come from the same config.
  3. fold_tables + summarize_precheck: per query, the union recall of all
     channels and of each leave-one-out union, the number of distinct candidates
     after union + dedup, and each channel's own selection metric (to check that
     the candidates reproduce results/d_phase.csv).

Labels stay inside eval/: every number that needs labels comes from FoldEvaluator.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from hmrec.config import BaseConfig, SelectionRule, load_yaml
from hmrec.eval.metrics import FOLD, QUERY_ID, VALUE, macro_mean
from hmrec.hpo.grid import PhaseGrid, RawConfig, Step
from hmrec.hpo.records import TrialRecord, read_records, result_columns
from hmrec.hpo.runner import (
    FoldContext,
    build_channel,
    make_evaluate_fn,
    make_trial_id,
    parsed_config,
    run_channel,
)
from hmrec.hpo.selection import STOP
from hmrec.logging_utils import get_logger
from hmrec.memory import MemoryTracker
from hmrec.paths import ProjectPaths
from hmrec.recall.base import ARTICLE_IDX, validate_candidates, write_candidates

LOGGER = get_logger("hpo.union_precheck")

ALL_CHANNELS = "all"
POOLED = "pooled"
MEASURE = "measure"
RECORDED_POOLED = "recorded_pooled"
STATISTICS = {"p50": 0.5, "p90": 0.9}

CandidatesFile = Callable[[str, str, str], Path]  # (fold, channel, trial_id) -> Parquet path


# ---------------------------------------------------------------- 1. selected configs


@dataclass(frozen=True)
class SelectedChannel:
    """The D-phase selection of one channel.

    Attributes:
        channel: Channel name, e.g. "r4_itemcf".
        raw_config: The selected config (as in configs/recall/{channel}.yaml).
        record: The selected trial, as stored in results/d_phase.csv.
    """

    channel: str
    raw_config: RawConfig
    record: TrialRecord


def _refuse_new_trial(channel: str) -> Callable[[str, RawConfig], TrialRecord]:
    """A run_trial function that always fails: the pre-check must not run new trials."""

    def refuse(step: str, raw_config: RawConfig) -> TrialRecord:
        trial_id = make_trial_id(channel, raw_config)
        raise ValueError(
            f"{channel}: step {step!r} needs trial {trial_id}, which is not in "
            "results/d_phase.csv; the pre-check only replays finished D-phase trials"
        )

    return refuse


def _never_save(record: TrialRecord) -> None:
    """Never called: _refuse_new_trial fails before anything could be saved."""
    raise RuntimeError(f"the pre-check must not save trials (got {record.trial_id})")


def select_channel(
    channel: str,
    steps: Sequence[Step],
    default_raw: Mapping[str, object],
    finished: Mapping[str, TrialRecord],
    rule: SelectionRule,
) -> SelectedChannel:
    """Replay one channel's D-phase steps over finished trials and return the selection.

    Uses the same run_channel as the D-phase run, so the rule (and every
    tie-break) is applied exactly as it was then.

    Args:
        channel: Channel name.
        steps: The channel's steps from configs/hpo/d_phase.yaml.
        default_raw: configs/recall/{channel}.yaml.
        finished: trial_id -> record of this channel from results/d_phase.csv.
        rule: configs/selection_rule.yaml.

    Raises:
        ValueError: If a step needs a trial that is not in `finished`, or if a
            step ended in "stop" (a human decision is still open, so there is
            no final config).
    """
    evaluate = make_evaluate_fn(channel, finished, _refuse_new_trial(channel), _never_save)
    reports, selected_raw = run_channel(channel, steps, default_raw, evaluate, rule)
    for report in reports:
        if report.decision is not None and report.decision.action == STOP:
            raise ValueError(
                f"{channel}: step {report.step!r} ended in 'stop' "
                f"({report.decision.reason}); there is no final config yet"
            )
    record = finished[make_trial_id(channel, selected_raw)]
    return SelectedChannel(channel, selected_raw, record)


def select_channels(
    paths: ProjectPaths,
    channels: Sequence[str],
    grid: PhaseGrid,
    rule: SelectionRule,
    folds: Sequence[str],
) -> list[SelectedChannel]:
    """select_channel for each channel, reading results/d_phase.csv once."""
    columns = result_columns(folds, list(grid.fold_groups), grid.extra_metrics)
    records = read_records(
        paths.d_phase_results_file, columns, folds, list(grid.fold_groups), grid.extra_metrics
    )
    selected: list[SelectedChannel] = []
    for channel in channels:
        finished = {r.trial_id: r for r in records if r.channel == channel and r.phase == "D"}
        default_raw = load_yaml(paths.recall_config_file(channel))
        selected.append(
            select_channel(channel, grid.channels[channel], default_raw, finished, rule)
        )
        LOGGER.info("%s: selected trial %s", channel, selected[-1].record.trial_id)
    return selected


# ---------------------------------------------------------------- 2. candidates (IO)


def load_or_generate_candidates(
    selected: SelectedChannel, context: FoldContext, base: BaseConfig, path: Path
) -> pl.DataFrame:
    """Candidates of one channel on one fold: read `path` if it exists, else generate and write it.

    An existing file is never overwritten: candidates are frozen once written
    (AGENTS.md §3.1). It is validated again after reading.

    Returns:
        The channel output: one row per (query_id, article_idx), with rank (1 = best).
    """
    channel = selected.channel
    recall_channel = build_channel(channel, selected.raw_config, base)
    quota = parsed_config(channel, selected.raw_config)["quota"]
    if path.exists():
        LOGGER.info("%s %s: reading existing %s", channel, context.fold.name, path)
        candidates = pl.read_parquet(path)
        validate_candidates(candidates, recall_channel.output_spec, quota)
        return candidates
    with MemoryTracker(
        f"{channel} {context.fold.name}",
        LOGGER,
        limit_gb=base.memory.limit_gb,
        sample_interval_seconds=base.memory.sample_interval_seconds,
    ):
        candidates = recall_channel.generate(context.queries, context.inputs)
    write_candidates(candidates, recall_channel.output_spec, quota, path, overwrite=False)
    LOGGER.info("%s %s: %d rows -> %s", channel, context.fold.name, candidates.height, path)
    return candidates


# ---------------------------------------------------------------- 3. per-fold tables


@dataclass(frozen=True)
class FoldTables:
    """Per-query tables of one fold; every table has columns query_id, fold, value (float64).

    Attributes:
        union_recall: Union name -> per-query union recall. Names: "all" (every
            channel) and "without_{channel}" (leave-one-out).
        candidate_counts: Number of distinct articles per query over all
            channels (0 for a query with no candidate).
        channel_metric: Channel -> per-query value of its own selection metric.
    """

    union_recall: dict[str, pl.DataFrame]
    candidate_counts: pl.DataFrame
    channel_metric: dict[str, pl.DataFrame]


def union_recall_tables(
    candidates_by_channel: Mapping[str, pl.DataFrame], context: FoldContext
) -> dict[str, pl.DataFrame]:
    """Per-query union recall of all channels and of every leave-one-out union.

    Args:
        candidates_by_channel: Channel -> candidates (query_id, article_idx, ...).
        context: The fold (its evaluator holds the labels).

    Returns:
        {"all": ..., "without_{channel}": ... for each channel}; see FoldTables.
    """
    if len(candidates_by_channel) < 2:
        raise ValueError("leave-one-out needs at least 2 channels")
    pairs = {
        channel: candidates.select(QUERY_ID, ARTICLE_IDX)
        for channel, candidates in candidates_by_channel.items()
    }
    tables = {ALL_CHANNELS: context.evaluator.union_recall(pl.concat(list(pairs.values())))}
    for left_out in pairs:
        others = [table for channel, table in pairs.items() if channel != left_out]
        tables[f"without_{left_out}"] = context.evaluator.union_recall(pl.concat(others))
    return tables


def count_candidates_per_query(
    candidates_by_channel: Mapping[str, pl.DataFrame], queries: pl.DataFrame
) -> pl.DataFrame:
    """Number of distinct candidate articles per query after union + dedup.

    Args:
        candidates_by_channel: Channel -> candidates (query_id, article_idx, ...).
        queries: One row per query of the fold; columns query_id, fold.

    Returns:
        One row per query in `queries`, sorted by query_id: query_id, fold,
        value (float64, so the per-query averaging helpers can be reused).
        A query that no channel returned anything for gets 0.
    """
    all_pairs = pl.concat([c.select(QUERY_ID, ARTICLE_IDX) for c in candidates_by_channel.values()])
    # Dedup at (query_id, article_idx): an article from several channels counts once.
    counts = all_pairs.unique().group_by(QUERY_ID).agg(pl.len().alias("n_candidates"))
    per_query = queries.select(QUERY_ID, FOLD).join(counts, on=QUERY_ID, how="left")
    value = pl.col("n_candidates").fill_null(0).cast(pl.Float64).alias(VALUE)
    return per_query.select(QUERY_ID, FOLD, value).sort(QUERY_ID)


def fold_tables(
    candidates_by_channel: Mapping[str, pl.DataFrame],
    context: FoldContext,
    metrics: Mapping[str, str],
) -> FoldTables:
    """All per-query tables of one fold.

    Args:
        candidates_by_channel: Channel -> candidates with query_id, article_idx, rank.
        context: The fold.
        metrics: Channel -> its selection metric, e.g. "recall@100".
    """
    channel_metric = {
        channel: context.evaluator.per_query(candidates, metrics[channel])
        for channel, candidates in candidates_by_channel.items()
    }
    return FoldTables(
        union_recall=union_recall_tables(candidates_by_channel, context),
        candidate_counts=count_candidates_per_query(candidates_by_channel, context.queries),
        channel_metric=channel_metric,
    )


def zero_hit_indicator(per_query: pl.DataFrame) -> pl.DataFrame:
    """1.0 for a query whose value is 0, else 0.0. Its mean is the share of such queries.

    Applied to the "all" union recall, this marks queries where no channel found
    any of the labels.
    """
    is_zero = (pl.col(VALUE) == 0).cast(pl.Float64).alias(VALUE)
    return per_query.select(QUERY_ID, FOLD, is_zero)


# ---------------------------------------------------------------- 4. summary over folds


def precheck_scopes(
    folds: Sequence[str], fold_groups: Mapping[str, Sequence[str]]
) -> dict[str, tuple[str, ...]]:
    """Scope name -> folds: each fold alone, "pooled" (all folds), then the fold groups."""
    scopes: dict[str, tuple[str, ...]] = {fold: (fold,) for fold in folds}
    scopes[POOLED] = tuple(folds)
    for group, group_folds in fold_groups.items():
        scopes[group] = tuple(group_folds)
    return scopes


def scope_statistic(per_query: pl.DataFrame, folds: Sequence[str], statistic: str) -> float:
    """Mean ("mean"), median ("p50") or 90th percentile ("p90") of `value` over the given folds.

    The mean pools all queries of those folds (proposal §4.1, query-weighted).
    Percentiles use the "nearest" rule, so they are always an observed value.
    """
    part = per_query.filter(pl.col(FOLD).is_in(list(folds)))
    if statistic == "mean":
        return macro_mean(part)
    quantile = part[VALUE].quantile(STATISTICS[statistic], interpolation="nearest")
    if quantile is None:
        raise ValueError(f"no queries in folds {list(folds)}")
    return float(quantile)


def scope_values(
    per_query: pl.DataFrame, scopes: Mapping[str, Sequence[str]], statistic: str
) -> dict[str, float]:
    """Scope name -> scope_statistic(per_query, that scope's folds, statistic)."""
    return {scope: scope_statistic(per_query, folds, statistic) for scope, folds in scopes.items()}


def _concat(
    tables: Sequence[FoldTables], pick: Callable[[FoldTables], pl.DataFrame]
) -> pl.DataFrame:
    """Stack one per-query table over all folds (query_ids are unique across folds)."""
    return pl.concat([pick(table) for table in tables])


def _union_rows(
    tables: Sequence[FoldTables],
    scopes: Mapping[str, Sequence[str]],
    channels: Sequence[str],
) -> dict[str, dict[str, float]]:
    """Union recall of all channels, each leave-one-out union, and the drop it causes."""
    union_all = _concat(tables, lambda t: t.union_recall[ALL_CHANNELS])
    all_values = scope_values(union_all, scopes, "mean")
    rows = {"union_recall_all": all_values}
    for channel in channels:
        key = f"without_{channel}"
        without = _concat(tables, lambda t, key=key: t.union_recall[key])
        without_values = scope_values(without, scopes, "mean")
        rows[f"union_recall_{key}"] = without_values
        # Drop = union recall (all) - union recall (without this channel): how much
        # of the labels only this channel finds (proposal §5.7 item 2).
        rows[f"union_recall_drop_{key}"] = {
            scope: all_values[scope] - without_values[scope] for scope in scopes
        }
    rows["zero_hit_query_share"] = scope_values(zero_hit_indicator(union_all), scopes, "mean")
    return rows


def summarize_precheck(
    tables: Sequence[FoldTables],
    scopes: Mapping[str, Sequence[str]],
    selected: Sequence[SelectedChannel],
) -> pl.DataFrame:
    """The pre-check table: one row per measure, one column per scope (fold, pooled, group).

    Rows: union_recall_all; union_recall_without_{channel} and
    union_recall_drop_without_{channel} for every channel; zero_hit_query_share;
    candidates_per_query_{mean,p50,p90}; {channel}_{metric} for every channel.
    Column recorded_pooled holds the pooled score of the selected trial in
    results/d_phase.csv for the channel rows (null elsewhere), as a check that
    the candidates were reproduced.
    """
    rows = _union_rows(tables, scopes, [choice.channel for choice in selected])
    counts = _concat(tables, lambda t: t.candidate_counts)
    for statistic in ("mean", "p50", "p90"):
        rows[f"candidates_per_query_{statistic}"] = scope_values(counts, scopes, statistic)
    recorded: dict[str, float] = {}
    for choice in selected:
        metric_table = _concat(tables, lambda t, c=choice.channel: t.channel_metric[c])
        name = f"{choice.channel}_{choice.record.metric}"
        rows[name] = scope_values(metric_table, scopes, "mean")
        recorded[name] = choice.record.summary.pooled
    records = [
        {MEASURE: name, **values, RECORDED_POOLED: recorded.get(name)}
        for name, values in rows.items()
    ]
    schema = {MEASURE: pl.String, **dict.fromkeys(scopes, pl.Float64), RECORDED_POOLED: pl.Float64}
    return pl.DataFrame(records, schema=schema)


# ---------------------------------------------------------------- run


def run_union_precheck(
    selected: Sequence[SelectedChannel],
    contexts: Sequence[FoldContext],
    base: BaseConfig,
    candidates_file: CandidatesFile,
    fold_groups: Mapping[str, Sequence[str]],
) -> pl.DataFrame:
    """Generate (or read back) every channel's candidates on every fold, then summarise.

    One fold is held in memory at a time; per channel only query_id,
    article_idx and rank are kept.

    Args:
        selected: From select_channels.
        contexts: The D folds.
        base: configs/base.yaml (chunk sizes, threads, memory limit).
        candidates_file: (fold, channel, trial_id) -> Parquet path.
        fold_groups: Diagnostic groups, e.g. {"D1_D2": ("D1", "D2")}.

    Returns:
        The summary table (see summarize_precheck).
    """
    metrics = {choice.channel: choice.record.metric for choice in selected}
    tables: list[FoldTables] = []
    for context in contexts:
        candidates_by_channel: dict[str, pl.DataFrame] = {}
        for choice in selected:
            path = candidates_file(context.fold.name, choice.channel, choice.record.trial_id)
            candidates = load_or_generate_candidates(choice, context, base, path)
            candidates_by_channel[choice.channel] = candidates.select(QUERY_ID, ARTICLE_IDX, "rank")
        tables.append(fold_tables(candidates_by_channel, context, metrics))
        LOGGER.info("fold %s: per-query tables done", context.fold.name)
    scopes = precheck_scopes([context.fold.name for context in contexts], fold_groups)
    return summarize_precheck(tables, scopes, selected)


def write_precheck_table(table: pl.DataFrame, path: Path) -> None:
    """Write the summary CSV; an existing file is never replaced (results/ is append-only)."""
    if path.exists():
        raise FileExistsError(f"{path} already exists; move it away to run the pre-check again")
    path.parent.mkdir(parents=True, exist_ok=True)
    table.write_csv(path)
