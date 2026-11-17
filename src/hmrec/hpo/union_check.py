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

Memory (docs/decisions.md 2026-11-17, after the pre-check peaked at 6.71 GB):
unions are built from the HITS only. A candidate that is not a label can never
change union recall, and only about 1 in 400 candidates is a hit, so each
channel's candidates are first cut to their hits (one semi join with the
labels, inside eval/), and every union / leave-one-out union is a union of
these small tables. Candidates per query are counted in chunks of queries.
The results equal the pre-check's direct computation exactly (tests).
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import polars as pl

from hmrec.config import (
    BaseConfig,
    PreprocessConfig,
    TwoTowerConfig,
    UnionCheckConfig,
    parse_twotower_config,
)
from hmrec.eval.metrics import FOLD, QUERY_ID, VALUE, macro_mean
from hmrec.hpo.runner import IN_ACTIVE_POOL, POOL_FLAG_CHANNELS, FoldContext, twotower_chunks
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
from hmrec.paths import ProjectPaths
from hmrec.recall.base import ARTICLE_IDX, validate_candidates, write_candidates
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


def count_candidates_chunked(
    candidates_by_channel: Mapping[str, pl.DataFrame], queries: pl.DataFrame, query_chunk_size: int
) -> pl.DataFrame:
    """Distinct candidate articles per query after union + dedup, counted in chunks of queries.

    Only one chunk's pairs of all channels are in memory at a time
    (query_chunk_size queries x at most 480 candidates).

    Returns:
        One row per query in `queries`, sorted by query_id: query_id, fold,
        value (float64; 0 for a query without candidates). Same as
        union_precheck.count_candidates_per_query.
    """
    if query_chunk_size <= 0:
        raise ValueError(f"query_chunk_size must be > 0, got {query_chunk_size}")
    query_ids = queries[QUERY_ID].sort()
    counts: list[pl.DataFrame] = []
    for start in range(0, query_ids.len(), query_chunk_size):
        chunk_ids = query_ids.slice(start, query_chunk_size)
        # query_ids are sorted and unique, so chunks are disjoint id ranges.
        in_chunk = pl.col(QUERY_ID).is_between(chunk_ids[0], chunk_ids[-1])
        parts = [c.filter(in_chunk).select(PAIR) for c in candidates_by_channel.values()]
        # Dedup at (query_id, article_idx): an article from several channels counts once.
        distinct = pl.concat(parts).unique()
        counts.append(distinct.group_by(QUERY_ID).agg(pl.len().alias("n_candidates")))
    per_query = queries.select(QUERY_ID, FOLD).join(pl.concat(counts), on=QUERY_ID, how="left")
    value = pl.col("n_candidates").fill_null(0).cast(pl.Float64).alias(VALUE)
    return per_query.select(QUERY_ID, FOLD, value).sort(QUERY_ID)


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
) -> FoldResult:
    """Union tables and pool diagnostics of one fold.

    Args:
        candidates_by_channel: Channel -> candidates with query_id, article_idx,
            rank (R2 / R3 also in_active_pool, see flag_active_pool).
        context: The fold.
        metrics: Channel -> its selection metric.
        query_chunk_size: Queries per chunk when counting candidates.
    """
    hits = channel_hits(candidates_by_channel, context)
    channel_metric = {
        channel: context.evaluator.per_query(candidates, metrics[channel])
        for channel, candidates in candidates_by_channel.items()
    }
    tables = FoldTables(
        union_recall=union_recall_tables_from_hits(hits, context),
        candidate_counts=count_candidates_chunked(
            candidates_by_channel, context.queries, query_chunk_size
        ),
        channel_metric=channel_metric,
    )
    pools = {
        channel: pool_diagnostics(channel, candidates_by_channel, hits, context)
        for channel in POOL_FLAG_CHANNELS
        if channel in candidates_by_channel
    }
    return FoldResult(tables, pools)


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
    then union_recall_target, the active-pool rows of R2 and R3 (pool_rows), and the
    R3 rule: r3_rule_max_union_recall_drop and r3_rule_switch_to_active_pool_only
    (1 = switch, 0 = keep; decided on the pooled column, repeated in every column).
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
    if R3_CHANNEL in results[0].pools:
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
) -> dict[str, pl.DataFrame]:
    """Per channel: query_id, article_idx, rank of one fold (R2 / R3 also in_active_pool)."""
    fold = context.fold.name
    candidates_by_channel: dict[str, pl.DataFrame] = {}
    for choice in selected:
        key = (fold, choice.channel, choice.record.trial_id)
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


def run_union_check(
    selected: Sequence[SelectedChannel],
    contexts: Sequence[FoldContext],
    base: BaseConfig,
    files: CandidateFiles,
    r6: R6Source | None,
    settings: UnionCheckConfig,
    fold_groups: Mapping[str, Sequence[str]],
) -> pl.DataFrame:
    """Load or generate the candidates fold by fold, check them, and build the M4 table.

    One fold's candidates are in memory at a time. Every channel's fold score
    must equal the recorded D-phase score exactly (check_reproduces_record).

    Args:
        selected: The six channels' selections (select_channels with manual choices).
        contexts: The D folds in time order.
        base: configs/base.yaml.
        files: Candidate file locations.
        r6: Model and checkpoints of R6 (None if R6 is not in `selected`).
        settings: configs/m4_union_check.yaml.
        fold_groups: Diagnostic groups, e.g. {"D1_D2": ("D1", "D2")}.
    """
    metrics = {choice.channel: choice.record.metric for choice in selected}
    results: list[FoldResult] = []
    for context in contexts:
        candidates_by_channel = _fold_candidates(selected, context, base, files, r6)
        result = fold_result(
            candidates_by_channel, context, metrics, base.chunking.query_chunk_size
        )
        for choice in selected:
            per_query = result.tables.channel_metric[choice.channel]
            check_reproduces_record(choice, context.fold.name, per_query)
        results.append(result)
        LOGGER.info("fold %s: union tables done", context.fold.name)
    scopes = precheck_scopes([context.fold.name for context in contexts], fold_groups)
    return summarize_union_check(results, scopes, selected, settings)
