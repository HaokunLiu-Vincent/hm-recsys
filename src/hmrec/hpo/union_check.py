"""M4 union check of the six channels on D1-D4 (proposal §5.7).

Per fold, with each channel's D-phase selection (R6: the manual choice in
configs/m4_union_check.yaml):

  1. Candidates. R1-R5 are read from the union pre-check files if they exist
     (artifacts/hpo/D/union_precheck/, same trial_id = same config), else
     generated into artifacts/m4/candidates/. R6 is NOT retrained: each fold is
     predicted with the model the chain saved BEFORE it trained on that fold
     (D1: base checkpoint; D2: chain/{trial_id}/D1.pt; ...), so the candidates
     are the ones scored in the D phase.
  2. Check: every channel's selection metric on every fold must equal the
     fold score recorded in results/d_phase.csv exactly, or the run stops.
  3. Union recall, leave-one-out, candidates per query, share of queries with
     no hit (items 1-3 of §5.7), and the active-pool diagnostics of R2 / R3 with
     the pre-registered R3 rule (item 4).

Folds are checked one at a time: a fold's context and candidates are loaded when
the fold starts and released before the next one (check_one_fold). Each stage can
be wrapped in a MemoryTracker (memory_stage_tracker), so the log shows the peak
RSS of every stage.

Memory (docs/decisions.md 2026-11-17, after the pre-check peaked at 6.71 GB):
unions are built from the HITS only. A candidate that is not a label can never
change union recall, and only about 1 in 400 candidates is a hit, so each
channel's candidates are first cut to their hits (one semi join with the
labels, inside eval/), and every union / leave-one-out union is a union of
these small tables. Candidates per query are counted in chunks of queries.
The results equal the pre-check's direct computation exactly (tests).
"""

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import polars as pl

from hmrec.config import (
    BaseConfig,
    MemoryConfig,
    PreprocessConfig,
    SelectionRule,
    TwoTowerConfig,
    UnionCheckConfig,
    load_yaml,
    parse_twotower_config,
)
from hmrec.eval.metrics import QUERY_ID, macro_mean
from hmrec.eval.monitor import count_candidates_chunked
from hmrec.hpo.grid import PhaseGrid, RawConfig, apply_overrides
from hmrec.hpo.records import TrialRecord, append_record, read_records, result_columns
from hmrec.hpo.runner import (
    IN_ACTIVE_POOL,
    POOL_FLAG_CHANNELS,
    EvaluateFn,
    FoldContext,
    TrialSettings,
    evaluate_trial,
    make_evaluate_fn,
)
from hmrec.hpo.union_precheck import (
    ALL_CHANNELS,
    MEASURE,
    POOLED,
    RECORDED_POOLED,
    CandidatesFile,
    FoldTables,
    SelectedChannel,
    load_or_generate_candidates,
    precheck_scopes,
    scope_values,
    summarize_precheck,
)
from hmrec.logging_utils import get_logger
from hmrec.memory import MemoryTracker
from hmrec.paths import ProjectPaths
from hmrec.recall.base import ARTICLE_IDX, validate_candidates, write_candidates
from hmrec.recall.build import twotower_chunks
from hmrec.recall.twotower.infer import (
    TWOTOWER_SPEC,
    TwoTowerChannel,
    TwoTowerChunks,
    TwoTowerResources,
    load_twotower_resources,
)
from hmrec.recall.twotower.train import check_model_identity, load_checkpoint

LOGGER = get_logger("hpo.union_check")

R6_CHANNEL = "r6_twotower"
R3_CHANNEL = "r3_sibling"
PAIR = [QUERY_ID, ARTICLE_IDX]


# ---------------------------------------------------------------- memory per stage

StageTracker = Callable[[str], AbstractContextManager[object]]


def no_stage_tracking(label: str) -> AbstractContextManager[object]:
    """A StageTracker that measures nothing (the default, used by tests)."""
    del label
    return nullcontext()


def memory_stage_tracker(memory: MemoryConfig) -> StageTracker:
    """A StageTracker that logs the start, end and PEAK RSS of every stage.

    Each stage runs inside its own MemoryTracker, so with a short sampling
    interval (memory.sample_interval_seconds, e.g. 0.05 s) the log shows in
    which stage a memory spike happens.
    """

    def track(label: str) -> AbstractContextManager[object]:
        return MemoryTracker(
            f"stage {label}",
            LOGGER,
            limit_gb=memory.limit_gb,
            sample_interval_seconds=memory.sample_interval_seconds,
        )

    return track


# ---------------------------------------------------------------- unions from hits


def channel_hits(
    candidates_by_channel: Mapping[str, pl.DataFrame], context: FoldContext
) -> dict[str, pl.DataFrame]:
    """Each channel's candidates cut to the rows that hit a label (all columns kept)."""
    return {
        channel: context.evaluator.hits(candidates)
        for channel, candidates in candidates_by_channel.items()
    }


def union_recall_tables_from_hits(
    hits_by_channel: Mapping[str, pl.DataFrame], context: FoldContext
) -> dict[str, pl.DataFrame]:
    """Per-query union recall of all channels and of every leave-one-out union, from hits.

    Returns:
        {"all": ..., "without_{channel}": ...}; each table has query_id, fold,
        value. Same values as union_precheck.union_recall_tables on the full
        candidates.
    """
    if len(hits_by_channel) < 2:
        raise ValueError("leave-one-out needs at least 2 channels")
    pairs = {channel: hits.select(PAIR) for channel, hits in hits_by_channel.items()}
    evaluator = context.evaluator
    tables = {ALL_CHANNELS: evaluator.union_recall_of_hits(pl.concat(list(pairs.values())))}
    for left_out in pairs:
        others = [table for channel, table in pairs.items() if channel != left_out]
        tables[f"without_{left_out}"] = evaluator.union_recall_of_hits(pl.concat(others))
    return tables


# ---------------------------------------------------------------- active-pool diagnostics


@dataclass(frozen=True)
class PoolDiagnostics:
    """Active-pool numbers of one channel (R2 or R3) on one fold (proposal §5.7 item 4).

    "Out-of-pool only" = outside A(a) AND returned by no other channel: exactly
    the candidates that disappear from the union if the channel is limited to A(a).

    Attributes:
        rows_in: Candidate rows inside A(a).
        rows_out: Candidate rows outside A(a).
        hits_in: Hits among rows_in.
        hits_out: Hits among rows_out.
        rows_out_only: Out-of-pool rows no other channel returns.
        hits_out_only: Hits among rows_out_only (the labels the union would lose).
        union_without_out: Per-query union recall with the channel's out-of-pool
            candidates removed (query_id, fold, value).
    """

    rows_in: int
    rows_out: int
    hits_in: int
    hits_out: int
    rows_out_only: int
    hits_out_only: int
    union_without_out: pl.DataFrame


def flag_active_pool(candidates: pl.DataFrame, active_pool: pl.DataFrame) -> pl.DataFrame:
    """Add the bool column in_active_pool: True if the article is in A(a)."""
    pool = active_pool[ARTICLE_IDX].implode()
    return candidates.with_columns(pl.col(ARTICLE_IDX).is_in(pool).alias(IN_ACTIVE_POOL))


def _pairs_also_elsewhere(pairs: pl.DataFrame, others: Sequence[pl.DataFrame]) -> pl.DataFrame:
    """The (query_id, article_idx) pairs of `pairs` that some table in `others` also has."""
    found = [other.select(PAIR).join(pairs, on=PAIR, how="semi") for other in others]
    return pl.concat(found).unique()


def pool_diagnostics(
    channel: str,
    candidates_by_channel: Mapping[str, pl.DataFrame],
    hits_by_channel: Mapping[str, pl.DataFrame],
    context: FoldContext,
) -> PoolDiagnostics:
    """The PoolDiagnostics of `channel`, whose candidates and hits carry in_active_pool."""
    candidates = candidates_by_channel[channel]
    hits = hits_by_channel[channel]
    inside = pl.col(IN_ACTIVE_POOL)
    out_rows = candidates.filter(~inside).select(PAIR)
    out_hits = hits.filter(~inside).select(PAIR)
    other_candidates = [c for name, c in candidates_by_channel.items() if name != channel]
    other_hits = [h.select(PAIR) for name, h in hits_by_channel.items() if name != channel]
    rows_elsewhere = _pairs_also_elsewhere(out_rows, other_candidates).height
    hits_elsewhere = _pairs_also_elsewhere(out_hits, other_hits).height
    kept_hits = pl.concat([*other_hits, hits.filter(inside).select(PAIR)])
    return PoolDiagnostics(
        rows_in=candidates.height - out_rows.height,
        rows_out=out_rows.height,
        hits_in=hits.height - out_hits.height,
        hits_out=out_hits.height,
        rows_out_only=out_rows.height - rows_elsewhere,
        hits_out_only=out_hits.height - hits_elsewhere,
        union_without_out=context.evaluator.union_recall_of_hits(kept_hits),
    )


# ---------------------------------------------------------------- one fold


@dataclass(frozen=True)
class FoldResult:
    """Everything computed on one fold.

    Attributes:
        tables: Union recall (all / leave-one-out), candidate counts, channel metrics.
        pools: Channel (R2, R3) -> its active-pool diagnostics.
    """

    tables: FoldTables
    pools: dict[str, PoolDiagnostics]


def fold_result(
    candidates_by_channel: Mapping[str, pl.DataFrame],
    context: FoldContext,
    metrics: Mapping[str, str],
    query_chunk_size: int,
    track: StageTracker = no_stage_tracking,
) -> FoldResult:
    """Union tables and pool diagnostics of one fold.

    Args:
        candidates_by_channel: Channel -> candidates with query_id, article_idx,
            rank (R2 / R3 also in_active_pool, see flag_active_pool).
        context: The fold.
        metrics: Channel -> its selection metric.
        query_chunk_size: Queries per chunk when counting candidates.
        track: Wraps each stage (to log its memory peak); default: no tracking.
    """
    fold = context.fold.name
    with track(f"{fold} | hits of every channel"):
        hits = channel_hits(candidates_by_channel, context)
    channel_metric: dict[str, pl.DataFrame] = {}
    for channel, candidates in candidates_by_channel.items():
        with track(f"{fold} | {channel} {metrics[channel]}"):
            channel_metric[channel] = context.evaluator.per_query(candidates, metrics[channel])
    with track(f"{fold} | union recall and leave-one-out"):
        union_tables = union_recall_tables_from_hits(hits, context)
    with track(f"{fold} | candidates per query"):
        counts = count_candidates_chunked(candidates_by_channel, context.queries, query_chunk_size)
    pools: dict[str, PoolDiagnostics] = {}
    for channel in POOL_FLAG_CHANNELS:
        if channel in candidates_by_channel:
            with track(f"{fold} | {channel} active-pool diagnostics"):
                pools[channel] = pool_diagnostics(channel, candidates_by_channel, hits, context)
    return FoldResult(FoldTables(union_tables, counts, channel_metric), pools)


def check_reproduces_record(choice: SelectedChannel, fold: str, per_query: pl.DataFrame) -> None:
    """Fail unless the fold score of these candidates equals the recorded D-phase score exactly."""
    score = macro_mean(per_query)
    recorded = choice.record.summary.fold_scores[fold]
    if score != recorded:
        raise ValueError(
            f"{choice.channel} {fold}: {choice.record.metric} of the candidates is {score!r}, "
            f"but results/d_phase.csv records {recorded!r} for trial {choice.record.trial_id}"
        )
    LOGGER.info(
        "%s %s: %s = %.6f, identical to the record",
        choice.channel,
        fold,
        choice.record.metric,
        score,
    )


# ---------------------------------------------------------------- variants (rule outcomes)


def variant_step(variant: str) -> str:
    """Step name of a variant's trials in results/m4_trials.csv, e.g. "m4_r3_active_pool"."""
    return f"m4_{variant}"


def apply_variant(
    selected: Sequence[SelectedChannel],
    overrides: Mapping[str, Mapping[str, object]],
    evaluator_of: Callable[[str], EvaluateFn],
    step: str,
) -> list[SelectedChannel]:
    """Replace the configs of the channels named in a variant.

    The new config's trial comes from `evaluator_of(channel)`: read from
    results/m4_trials.csv if it is there, else evaluated on the D folds and
    saved. Such a trial is the OUTCOME of a rule; it is never compared with
    other configurations.

    Args:
        selected: The channels' selections.
        overrides: Channel -> config keys to change (configs/m4_union_check.yaml, variants).
        evaluator_of: Channel -> its (step, raw config) -> TrialRecord function.
        step: Step name stored with new trials.
    """
    unknown = sorted(set(overrides) - {choice.channel for choice in selected})
    if unknown:
        raise ValueError(f"variant changes channels that are not selected: {unknown}")
    result: list[SelectedChannel] = []
    for choice in selected:
        if choice.channel not in overrides:
            result.append(choice)
            continue
        raw_config = apply_overrides(choice.raw_config, overrides[choice.channel])
        record = evaluator_of(choice.channel)(step, raw_config)
        LOGGER.info(
            "%s: variant trial %s (%s) replaces trial %s",
            choice.channel,
            record.trial_id,
            dict(overrides[choice.channel]),
            choice.record.trial_id,
        )
        result.append(SelectedChannel(choice.channel, raw_config, record))
    return result


@dataclass(frozen=True)
class VariantEvaluation:
    """What is needed to evaluate a variant trial on the D folds and record it.

    Attributes:
        paths: Project paths (results/m4_trials.csv, recall configs, flag files).
        grid: configs/hpo/d_phase.yaml (fold groups and extra metrics of the report).
        rule: configs/selection_rule.yaml (each channel's metric; nothing is selected).
        base: configs/base.yaml.
        contexts: The D folds.
        git_commit: HEAD when the run started.
        git_dirty: Uncommitted changes when the run started.
    """

    paths: ProjectPaths
    grid: PhaseGrid
    rule: SelectionRule
    base: BaseConfig
    contexts: Sequence[FoldContext]
    git_commit: str
    git_dirty: bool

    def evaluator_of(self, channel: str) -> EvaluateFn:
        """(step, raw config) -> record for one channel, backed by results/m4_trials.csv."""
        folds = [context.fold.name for context in self.contexts]
        groups, extra = list(self.grid.fold_groups), self.grid.extra_metrics
        columns = result_columns(folds, groups, extra)
        records = read_records(self.paths.m4_trials_file, columns, folds, groups, extra)
        finished = {record.trial_id: record for record in records if record.channel == channel}
        settings = TrialSettings(
            phase="D",
            metric=self.rule.selection_metric[channel],
            extra_metrics=extra,
            fold_groups=self.grid.fold_groups,
            base=self.base,
            git_commit=self.git_commit,
            git_dirty=self.git_dirty,
            flags_dir=partial(self.paths.hpo_trial_dir, "D"),
        )
        default_raw = load_yaml(self.paths.recall_config_file(channel))

        def run_trial(step: str, raw_config: RawConfig) -> TrialRecord:
            return evaluate_trial(channel, step, raw_config, default_raw, self.contexts, settings)

        save = partial(append_record, self.paths.m4_trials_file, columns=columns)
        return make_evaluate_fn(channel, finished, run_trial, save)


# ---------------------------------------------------------------- R6 from saved models


@dataclass(frozen=True)
class R6Source:
    """What is needed to predict a fold with the R6 model saved before that fold's fine-tune.

    Attributes:
        config: The selected R6 config.
        resources: Model (its weights are replaced fold by fold) and tables.
        chunks: Prediction chunk sizes.
        checkpoints: Fold -> checkpoint that predicted it in the D phase
            (see r6_fold_checkpoints).
    """

    config: TwoTowerConfig
    resources: TwoTowerResources
    chunks: TwoTowerChunks
    checkpoints: dict[str, Path]


def r6_fold_checkpoints(
    folds: Sequence[str], base_checkpoint: Path, chain_checkpoint: Callable[[str], Path]
) -> dict[str, Path]:
    """Fold -> the model that predicted it in the chain (proposal §6.4).

    The first fold was predicted by the base checkpoint; every later fold by the
    model saved after fine-tuning on the PREVIOUS fold. So no fold is predicted
    by a model that has seen it.

    Args:
        folds: Fold names in time order, e.g. ["D1", "D2", "D3", "D4"].
        base_checkpoint: artifacts/models/twotower/base/epoch_{E_base}.pt.
        chain_checkpoint: Fold -> artifacts/models/twotower/chain/{trial_id}/{fold}.pt.
    """
    checkpoints = {folds[0]: base_checkpoint}
    for previous, fold in zip(folds, folds[1:], strict=False):
        checkpoints[fold] = chain_checkpoint(previous)
    return checkpoints


def build_r6_source(
    paths: ProjectPaths,
    preprocess: PreprocessConfig,
    base: BaseConfig,
    choice: SelectedChannel,
    folds: Sequence[str],
) -> R6Source:
    """Load the R6 tables and model once and list the checkpoint of every fold.

    Args:
        paths: Project paths.
        preprocess: Current preprocess config.
        base: configs/base.yaml (chunk sizes).
        choice: The selected R6 trial (a chain config, e_update > 0).
        folds: Fold names in time order.
    """
    config = parse_twotower_config(choice.raw_config)
    if config.e_update <= 0:
        raise ValueError("the R6 selection is frozen (e_update = 0): it has no chain checkpoints")
    trial_id = choice.record.trial_id
    checkpoints = r6_fold_checkpoints(
        folds,
        paths.twotower_checkpoint_file(config.base_epoch),
        partial(paths.twotower_chain_checkpoint_file, trial_id),
    )
    missing = [str(path) for path in checkpoints.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"R6 checkpoints are missing: {missing}")
    resources = load_twotower_resources(paths, preprocess, config)
    return R6Source(config, resources, twotower_chunks(base), checkpoints)


def load_or_generate_r6(context: FoldContext, source: R6Source, path: Path) -> pl.DataFrame:
    """R6 candidates of one fold: read `path` if it exists, else predict with the saved model.

    Nothing is trained. An existing file is never overwritten.
    """
    fold = context.fold.name
    if path.exists():
        LOGGER.info("%s %s: reading existing %s", R6_CHANNEL, fold, path)
        candidates = pl.read_parquet(path)
        validate_candidates(candidates, TWOTOWER_SPEC, source.config.quota)
        return candidates
    checkpoint = source.checkpoints[fold]
    metadata = load_checkpoint(source.resources.model, checkpoint)
    check_model_identity(metadata["model_identity"], source.resources.identity, checkpoint)
    source.resources.model.eval()
    LOGGER.info("%s %s: predicting with %s", R6_CHANNEL, fold, checkpoint)
    channel = TwoTowerChannel(source.config, source.resources, source.chunks)
    candidates = channel.generate(context.queries, context.inputs)
    write_candidates(candidates, TWOTOWER_SPEC, source.config.quota, path, overwrite=False)
    return candidates


# ---------------------------------------------------------------- summary


POOL_COUNT_NAMES = ("rows_in", "rows_out", "hits_in", "hits_out", "rows_out_only", "hits_out_only")


def _ratio(numerator: int, denominator: int) -> float:
    """Return numerator / denominator, or 0.0 when there is nothing to divide by."""
    return numerator / denominator if denominator else 0.0


def pool_rows(
    channel: str,
    pools: Mapping[str, PoolDiagnostics],
    union_all: pl.DataFrame,
    scopes: Mapping[str, Sequence[str]],
) -> dict[str, dict[str, float]]:
    """Active-pool rows of one channel for the summary table.

    Counts are summed over the folds of each scope before dividing (so a hit
    rate is hits / rows of the whole scope, not a mean of per-fold rates).

    Args:
        channel: e.g. "r3_sibling".
        pools: Fold -> PoolDiagnostics of this channel.
        union_all: Per-query union recall of all channels, all folds.
        scopes: Scope name -> folds.
    """
    rows: dict[str, dict[str, float]] = {
        f"{channel}_out_of_pool_row_share": {},
        f"{channel}_in_pool_hit_rate": {},
        f"{channel}_out_of_pool_hit_rate": {},
        f"{channel}_out_of_pool_share_of_hits": {},
        f"{channel}_out_of_pool_only_row_share": {},
        f"{channel}_out_of_pool_only_hits": {},
    }
    for scope, folds in scopes.items():
        total = {
            name: sum(getattr(pools[fold], name) for fold in folds) for name in POOL_COUNT_NAMES
        }
        n_rows = total["rows_in"] + total["rows_out"]
        rows[f"{channel}_out_of_pool_row_share"][scope] = _ratio(total["rows_out"], n_rows)
        rows[f"{channel}_in_pool_hit_rate"][scope] = _ratio(total["hits_in"], total["rows_in"])
        rows[f"{channel}_out_of_pool_hit_rate"][scope] = _ratio(
            total["hits_out"], total["rows_out"]
        )
        rows[f"{channel}_out_of_pool_share_of_hits"][scope] = _ratio(
            total["hits_out"], total["hits_in"] + total["hits_out"]
        )
        rows[f"{channel}_out_of_pool_only_row_share"][scope] = _ratio(
            total["rows_out_only"], n_rows
        )
        rows[f"{channel}_out_of_pool_only_hits"][scope] = float(total["hits_out_only"])
    without = pl.concat([pools[fold].union_without_out for fold in pools])
    all_values = scope_values(union_all, scopes, "mean")
    without_values = scope_values(without, scopes, "mean")
    rows[f"union_recall_without_{channel}_out_of_pool"] = without_values
    # Drop = union recall (all) - union recall (channel limited to A(a)): §5.7 item 4.
    rows[f"union_recall_drop_without_{channel}_out_of_pool"] = {
        scope: all_values[scope] - without_values[scope] for scope in scopes
    }
    return rows


def r3_rule_switches(pooled_drop: float, max_drop: float) -> bool:
    """The pre-registered R3 rule (docs/decisions.md 2026-11-16).

    True = change R3 to output only articles in A(a): the 4-week pooled union
    recall drops by LESS than max_drop (0.001 = 0.1 percentage points, absolute)
    when R3's out-of-pool candidates are removed. A drop of exactly max_drop or
    more keeps R3 unchanged.
    """
    return pooled_drop < max_drop


def summarize_union_check(
    results: Sequence[FoldResult],
    scopes: Mapping[str, Sequence[str]],
    selected: Sequence[SelectedChannel],
    settings: UnionCheckConfig,
) -> pl.DataFrame:
    """The M4 table: one row per measure, one column per scope (fold, pooled, group).

    Rows: everything of union_precheck.summarize_precheck (union recall, leave-one-out,
    zero-hit share, candidates per query, each channel's metric with recorded_pooled),
    then union_recall_target, the active-pool rows of R2 and R3 (pool_rows), and, while
    R3 still has candidates outside A(a), the R3 rule: r3_rule_max_union_recall_drop
    and r3_rule_switch_to_active_pool_only (1 = switch, 0 = keep; decided on the pooled
    column, repeated in every column).
    """
    table = summarize_precheck([result.tables for result in results], scopes, selected)
    folds = [fold for fold, scope_folds in scopes.items() if tuple(scope_folds) == (fold,)]
    union_all = pl.concat([result.tables.union_recall[ALL_CHANNELS] for result in results])
    extra: dict[str, dict[str, float]] = {
        "union_recall_target": dict.fromkeys(scopes, settings.union_recall_target)
    }
    for channel in results[0].pools:
        pools = {fold: result.pools[channel] for fold, result in zip(folds, results, strict=True)}
        extra.update(pool_rows(channel, pools, union_all, scopes))
    r3_pools = [result.pools[R3_CHANNEL] for result in results if R3_CHANNEL in result.pools]
    if r3_pools and sum(pool.rows_out for pool in r3_pools) > 0:
        # The rule is about R3 while it still returns articles outside A(a); once R3
        # is "A(a) only" there is nothing left to decide, so the rows are left out.
        drop = extra[f"union_recall_drop_without_{R3_CHANNEL}_out_of_pool"][POOLED]
        switch = r3_rule_switches(drop, settings.r3_max_union_recall_drop)
        extra["r3_rule_max_union_recall_drop"] = dict.fromkeys(
            scopes, settings.r3_max_union_recall_drop
        )
        extra["r3_rule_switch_to_active_pool_only"] = dict.fromkeys(scopes, float(switch))
    extra_rows = [
        {MEASURE: name, **values, RECORDED_POOLED: None} for name, values in extra.items()
    ]
    return pl.concat([table, pl.DataFrame(extra_rows, schema=table.schema)])


# ---------------------------------------------------------------- run


@dataclass(frozen=True)
class CandidateFiles:
    """Where candidates are read from and written to.

    Attributes:
        reuse: (fold, channel, trial_id) -> union pre-check file; read if it exists.
        write: (fold, channel, trial_id) -> artifacts/m4/candidates/ file; read if
            it exists, else written.
    """

    reuse: CandidatesFile
    write: CandidatesFile


def _fold_candidates(
    selected: Sequence[SelectedChannel],
    context: FoldContext,
    base: BaseConfig,
    files: CandidateFiles,
    r6: R6Source | None,
    track: StageTracker,
) -> dict[str, pl.DataFrame]:
    """Per channel: query_id, article_idx, rank of one fold (R2 / R3 also in_active_pool)."""
    fold = context.fold.name
    candidates_by_channel: dict[str, pl.DataFrame] = {}
    for choice in selected:
        key = (fold, choice.channel, choice.record.trial_id)
        with track(f"{fold} | {choice.channel} candidates"):
            if choice.channel == R6_CHANNEL:
                if r6 is None:
                    raise ValueError("r6_twotower is selected but no R6Source was given")
                candidates = load_or_generate_r6(context, r6, files.write(*key))
            else:
                reuse_path = files.reuse(*key)
                path = reuse_path if reuse_path.exists() else files.write(*key)
                candidates = load_or_generate_candidates(choice, context, base, path)
            kept = candidates.select(QUERY_ID, ARTICLE_IDX, "rank")
            if choice.channel in POOL_FLAG_CHANNELS:
                kept = flag_active_pool(kept, context.inputs.active_pool)
            candidates_by_channel[choice.channel] = kept
    return candidates_by_channel


@dataclass(frozen=True)
class UnionCheckRun:
    """What every fold of one union-check run shares.

    Attributes:
        selected: The channels' selections.
        base: configs/base.yaml.
        files: Candidate file locations.
        r6: Model and checkpoints of R6 (None if R6 is not selected).
        track: Stage tracker (memory per stage).
    """

    selected: Sequence[SelectedChannel]
    base: BaseConfig
    files: CandidateFiles
    r6: R6Source | None
    track: StageTracker


def check_one_fold(run: UnionCheckRun, context: FoldContext) -> FoldResult:
    """Everything of ONE fold: candidates, record check, union tables.

    The fold's candidates exist only inside this function, so they are released
    when it returns; the result holds only small per-query tables.
    """
    metrics = {choice.channel: choice.record.metric for choice in run.selected}
    candidates = _fold_candidates(run.selected, context, run.base, run.files, run.r6, run.track)
    chunk_size = run.base.chunking.query_chunk_size
    result = fold_result(candidates, context, metrics, chunk_size, run.track)
    for choice in run.selected:
        per_query = result.tables.channel_metric[choice.channel]
        check_reproduces_record(choice, context.fold.name, per_query)
    return result


def run_union_check(
    run: UnionCheckRun,
    folds: Sequence[str],
    load_context: Callable[[str], FoldContext],
    settings: UnionCheckConfig,
    fold_groups: Mapping[str, Sequence[str]],
) -> pl.DataFrame:
    """Check the folds ONE AT A TIME and build the M4 table.

    Each fold's context (queries, labels, active pool) is loaded when the fold
    starts and released, together with its candidates, before the next fold
    starts. Every channel's fold score must equal its recorded score exactly
    (check_reproduces_record).

    Args:
        run: Selections, files, R6 source and stage tracker.
        folds: Fold names in time order.
        load_context: Fold name -> its FoldContext (loaded on demand).
        settings: configs/m4_union_check.yaml.
        fold_groups: Diagnostic groups, e.g. {"D1_D2": ("D1", "D2")}.
    """
    results: list[FoldResult] = []
    for fold in folds:
        with run.track(f"{fold} | whole fold"):
            results.append(check_one_fold(run, load_context(fold)))
        LOGGER.info("fold %s: union tables done", fold)
    scopes = precheck_scopes(list(folds), fold_groups)
    return summarize_union_check(results, scopes, run.selected, settings)
