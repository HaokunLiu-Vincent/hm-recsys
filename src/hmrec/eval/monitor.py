"""Monitoring of the generated candidates of one fold (M6, proposal §6.6). Reports only.

Nothing here selects, tunes or changes anything: the recall configuration is
locked. The rows go to results/m6_candidates.csv, one per channel and one for
the union of the six channels:

    every fold      rows, candidates per query (mean / P50 / P90), share of
                    queries with no candidate. These need NO labels.
    F and V folds   also Recall@quota of each channel, union recall of the six
                    channels and the share of queries no channel hits, through
                    the fold's FoldEvaluator (labels stay inside eval/).
    T folds         NO label is read and no label-based number is computed.
                    T metrics belong to M8 (the final evaluation). The code
                    refuses to build an evaluator for a T fold
                    (require_label_phase), and tests/unit/test_monitor.py
                    checks that.

A channel's recall is also compared with the mean of the folds before it
(proposal §6.6: "more than 10% below the mean of the previous 4 weeks is
flagged, the configuration is not changed"). For F1 the folds before are
D1-D4, whose scores are recorded in the lock files. The flag is a column of
the report and has no other effect.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import polars as pl

from hmrec.config import CandidateRunConfig
from hmrec.eval.metrics import FOLD, QUERY_ID, VALUE, macro_mean
from hmrec.eval.report import FoldEvaluator
from hmrec.locks import LockedChannel
from hmrec.run_records import RunStamp

ARTICLE_IDX = "article_idx"
PAIR = [QUERY_ID, ARTICLE_IDX]
UNION = "union"
UNION_METRIC = "union_recall"
# The only phases whose labels M6 may read. D is finished (its scores are in the
# lock files); T is not evaluated before M8.
LABEL_PHASES = ("F", "V")

MONITOR_COLUMNS: tuple[str, ...] = (
    "fold",
    "phase",
    "scope",
    "quota",
    "metric",
    "n_queries",
    "n_rows",
    "candidates_per_query_mean",
    "candidates_per_query_p50",
    "candidates_per_query_p90",
    "empty_query_share",
    "recall",
    "previous_folds",
    "previous_mean_recall",
    "recall_vs_previous",
    "low_recall_flag",
    "zero_hit_query_share",
    "labels_read",
    "git_commit",
    "git_dirty",
    "finished_at",
)
NOT_COMPUTED = ""  # an empty cell: the number was not computed (e.g. recall on a T fold)


# ---------------------------------------------------------------- counts (no labels)


def count_candidates_chunked(
    candidates_by_channel: Mapping[str, pl.DataFrame], queries: pl.DataFrame, query_chunk_size: int
) -> pl.DataFrame:
    """Distinct candidate articles per query after union + dedup, counted in chunks of queries.

    Only one chunk's pairs of all channels are in memory at a time
    (query_chunk_size queries x at most 480 candidates).

    Args:
        candidates_by_channel: Channel -> candidates with query_id, article_idx.
            With one channel this is simply that channel's rows per query.
        queries: One row per query of the fold; columns query_id, fold.
        query_chunk_size: Queries per chunk (chunking.query_chunk_size).

    Returns:
        One row per query in `queries`, sorted by query_id: query_id, fold,
        value (float64; 0 for a query without candidates).
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


@dataclass(frozen=True)
class CandidateCounts:
    """How many candidates the queries of one fold have (one channel, or the union).

    Attributes:
        n_queries: Queries of the fold (also those without a candidate).
        n_rows: Candidate rows in total (distinct (query, article) pairs).
        mean: Mean candidates per query (queries without candidates count as 0).
        p50: Median candidates per query ("nearest" rule: an observed value).
        p90: 90th percentile of candidates per query.
        empty_query_share: Share of queries with no candidate at all.
    """

    n_queries: int
    n_rows: int
    mean: float
    p50: float
    p90: float
    empty_query_share: float


def _quantile(values: pl.Series, quantile: float) -> float:
    """A quantile of a non-empty float series with the "nearest" rule."""
    result = values.quantile(quantile, interpolation="nearest")
    if result is None:
        raise ValueError("cannot take a quantile of an empty table")
    return float(result)


def summarize_counts(per_query: pl.DataFrame) -> CandidateCounts:
    """Summarise a per-query count table (query_id, fold, value) from count_candidates_chunked."""
    if per_query.height == 0:
        raise ValueError("the fold has no queries")
    values = per_query[VALUE]
    return CandidateCounts(
        n_queries=per_query.height,
        n_rows=int(values.sum()),
        mean=macro_mean(per_query),
        p50=_quantile(values, 0.5),
        p90=_quantile(values, 0.9),
        empty_query_share=float((values == 0).sum()) / per_query.height,
    )


# ---------------------------------------------------------------- labels: F and V folds only


def require_label_phase(fold: str, phase: str) -> None:
    """Fail unless M6 may read the labels of this fold (phases F and V only).

    Raises:
        ValueError: For any other phase. In particular a T fold: its labels are
            not read and no label-based metric is computed before M8.
    """
    if phase not in LABEL_PHASES:
        raise ValueError(
            f"M6 must not read the labels of fold {fold} (phase {phase}): only phases "
            f"{LABEL_PHASES} are monitored with labels; T metrics are computed in M8"
        )


def load_label_evaluator(fold: str, phase: str, labels_file: Path) -> FoldEvaluator:
    """The fold's FoldEvaluator; the ONLY way M6 gets one. The phase is checked first.

    Raises:
        ValueError: If the fold is not an F or V fold (the file is not opened).
    """
    require_label_phase(fold, phase)
    return FoldEvaluator.from_file(fold, labels_file)


@dataclass(frozen=True)
class LabelMetrics:
    """Label-based numbers of one fold.

    Attributes:
        channel_recall: Channel -> mean over queries of its selection metric
            (Recall@quota), float64.
        union_recall: Mean over queries of the union recall of all channels.
        zero_hit_query_share: Share of queries for which no channel returned any label.
    """

    channel_recall: dict[str, float]
    union_recall: float
    zero_hit_query_share: float


def label_metrics(
    evaluator: FoldEvaluator,
    candidates_by_channel: Mapping[str, pl.DataFrame],
    metrics: Mapping[str, str],
) -> LabelMetrics:
    """Each channel's Recall@quota, the union recall and the share of queries with no hit.

    The union is built from the HITS only (a candidate that is not a label
    cannot change union recall), as in the M4 union check.

    Args:
        evaluator: The fold's evaluator (holds the labels).
        candidates_by_channel: Channel -> candidates with query_id, article_idx, rank.
        metrics: Channel -> its selection metric, e.g. "recall@100".
    """
    channel_recall = {
        channel: macro_mean(evaluator.per_query(candidates, metrics[channel]))
        for channel, candidates in candidates_by_channel.items()
    }
    hits = [evaluator.hits(c.select(PAIR)) for c in candidates_by_channel.values()]
    union = evaluator.union_recall_of_hits(pl.concat(hits))
    zero_hit = union.select(QUERY_ID, FOLD, (pl.col(VALUE) == 0).cast(pl.Float64).alias(VALUE))
    return LabelMetrics(channel_recall, macro_mean(union), macro_mean(zero_hit))


# ---------------------------------------------------------------- comparison with earlier folds


@dataclass(frozen=True)
class RecallComparison:
    """A fold's recall against the mean of the folds before it. Reported only.

    Attributes:
        previous_folds: The folds the mean is taken over, in time order.
        previous_mean: Their mean recall; None if one of them has no recall
            recorded (then nothing is compared).
        relative_change: recall / previous_mean - 1 (e.g. -0.12 = 12% lower); None likewise.
        is_low: True if the recall is more than the allowed share below the mean.
    """

    previous_folds: tuple[str, ...]
    previous_mean: float | None
    relative_change: float | None
    is_low: bool


def previous_fold_names(fold: str, fold_names: Sequence[str], n_previous: int) -> tuple[str, ...]:
    """The `n_previous` folds right before `fold` in time order (fewer if there are not enough)."""
    names = list(fold_names)
    if fold not in names:
        raise ValueError(f"unknown fold {fold!r}; expected one of {names}")
    position = names.index(fold)
    return tuple(names[max(0, position - n_previous) : position])


def compare_with_previous(
    recall: float,
    recall_by_fold: Mapping[str, float],
    previous_folds: Sequence[str],
    n_previous: int,
    max_relative_drop: float,
) -> RecallComparison:
    """Compare a recall with the mean recall of the folds before it (proposal §6.6).

    The fold is flagged if recall < mean x (1 - max_relative_drop), e.g. more
    than 10% below the mean of the previous 4 weeks.

    Args:
        recall: The fold's recall.
        recall_by_fold: Fold -> recall of earlier folds (D folds from the lock
            files, post-lock folds from results/m6_candidates.csv).
        previous_folds: The folds right before this one (previous_fold_names).
        n_previous: How many folds the comparison needs (4).
        max_relative_drop: Allowed share below the mean (0.10).

    Returns:
        The comparison. If there are fewer than `n_previous` folds before, or
        one of them has no recall, previous_mean is None and nothing is flagged.
    """
    names = tuple(previous_folds)
    values = [recall_by_fold[name] for name in names if name in recall_by_fold]
    if len(names) < n_previous or len(values) < len(names):
        return RecallComparison(names, None, None, False)
    previous_mean = math.fsum(values) / len(values)
    if previous_mean <= 0:
        return RecallComparison(names, previous_mean, None, False)
    relative_change = recall / previous_mean - 1.0
    return RecallComparison(
        names, previous_mean, relative_change, relative_change < -max_relative_drop
    )


def recall_history(
    locked: Mapping[str, LockedChannel], earlier_rows: Sequence[Mapping[str, str]]
) -> dict[str, dict[str, float]]:
    """Channel -> {fold -> recall} of every fold that already has a recall.

    D folds come from the lock files (d_phase_record.folds); post-lock folds
    from the rows already in results/m6_candidates.csv. T folds never appear:
    their rows have no recall.
    """
    history = {channel: dict(lock.d_phase_record.folds) for channel, lock in locked.items()}
    for row in earlier_rows:
        if row["scope"] in history and row["recall"] != NOT_COMPUTED:
            history[row["scope"]][row["fold"]] = float(row["recall"])
    return history


# ---------------------------------------------------------------- rows of results/m6_candidates.csv


@dataclass(frozen=True)
class FoldCandidates:
    """The generated candidates of one fold, as the monitor reads them.

    Attributes:
        fold: Fold name, e.g. "F1".
        phase: The fold's phase ("F", "V" or "T").
        queries: One row per query; columns query_id, fold.
        candidates_by_channel: Channel -> its candidates (query_id, article_idx,
            rank), in channel order R1 ... R6.
        locked: Channel -> its lock (quota, selection metric, D-phase scores).
        fold_names: Every fold in time order, D1 ... T4 (for the comparison).
    """

    fold: str
    phase: str
    queries: pl.DataFrame
    candidates_by_channel: Mapping[str, pl.DataFrame]
    locked: Mapping[str, LockedChannel]
    fold_names: tuple[str, ...]


def _text(value: float | None) -> str:
    """A float as CSV text (repr, so it reads back exactly); None as an empty cell."""
    return NOT_COMPUTED if value is None else repr(value)


def _row(
    data: FoldCandidates,
    scope: str,
    quota: int,
    metric: str,
    counts: CandidateCounts,
    recall: float | None,
    comparison: RecallComparison | None,
    zero_hit_query_share: float | None,
    stamp: RunStamp,
) -> dict[str, str]:
    """One row of results/m6_candidates.csv (a channel, or the union)."""
    return {
        "fold": data.fold,
        "phase": data.phase,
        "scope": scope,
        "quota": str(quota),
        "metric": metric,
        "n_queries": str(counts.n_queries),
        "n_rows": str(counts.n_rows),
        "candidates_per_query_mean": repr(counts.mean),
        "candidates_per_query_p50": repr(counts.p50),
        "candidates_per_query_p90": repr(counts.p90),
        "empty_query_share": repr(counts.empty_query_share),
        "recall": _text(recall),
        "previous_folds": " ".join(comparison.previous_folds) if comparison else NOT_COMPUTED,
        "previous_mean_recall": _text(comparison.previous_mean if comparison else None),
        "recall_vs_previous": _text(comparison.relative_change if comparison else None),
        "low_recall_flag": str(comparison.is_low) if comparison else NOT_COMPUTED,
        "zero_hit_query_share": _text(zero_hit_query_share),
        "labels_read": str(recall is not None),
        "git_commit": stamp.git_commit,
        "git_dirty": str(stamp.git_dirty),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }


def monitor_rows(
    data: FoldCandidates,
    evaluator: FoldEvaluator | None,
    earlier_rows: Sequence[Mapping[str, str]],
    config: CandidateRunConfig,
    query_chunk_size: int,
    stamp: RunStamp,
) -> list[dict[str, str]]:
    """The rows of one fold for results/m6_candidates.csv: one per channel, then the union.

    Args:
        data: The fold's candidates and locks.
        evaluator: The fold's evaluator from load_label_evaluator for an F or V
            fold; None for a T fold (then every label-based cell stays empty).
        earlier_rows: The rows already in results/m6_candidates.csv.
        config: configs/m6_candidates.yaml (comparison settings).
        query_chunk_size: Queries per chunk when counting candidates.
        stamp: The code state of the run.

    Raises:
        ValueError: If an evaluator is given for a fold whose labels M6 may not read.
    """
    if evaluator is not None:
        require_label_phase(data.fold, data.phase)
    metrics = {channel: lock.d_phase_record.metric for channel, lock in data.locked.items()}
    labels = (
        None if evaluator is None else label_metrics(evaluator, data.candidates_by_channel, metrics)
    )
    history = recall_history(data.locked, earlier_rows)
    previous = previous_fold_names(data.fold, data.fold_names, config.previous_folds)
    rows: list[dict[str, str]] = []
    for channel, candidates in data.candidates_by_channel.items():
        per_query = count_candidates_chunked({channel: candidates}, data.queries, query_chunk_size)
        recall = None if labels is None else labels.channel_recall[channel]
        comparison = None
        if recall is not None:
            comparison = compare_with_previous(
                recall,
                history[channel],
                previous,
                config.previous_folds,
                config.max_relative_recall_drop,
            )
        quota = data.locked[channel].quota
        counts = summarize_counts(per_query)
        rows.append(
            _row(data, channel, quota, metrics[channel], counts, recall, comparison, None, stamp)
        )
    union_counts = summarize_counts(
        count_candidates_chunked(data.candidates_by_channel, data.queries, query_chunk_size)
    )
    total_quota = sum(lock.quota for lock in data.locked.values())
    rows.append(
        _row(
            data,
            UNION,
            total_quota,
            UNION_METRIC,
            union_counts,
            None if labels is None else labels.union_recall,
            None,
            None if labels is None else labels.zero_hit_query_share,
            stamp,
        )
    )
    return rows


def flagged_channels(rows: Sequence[Mapping[str, str]]) -> list[str]:
    """Scopes of `rows` whose low_recall_flag is set (for the one-line fold summary)."""
    return [row["scope"] for row in rows if row["low_recall_flag"] == "True"]
