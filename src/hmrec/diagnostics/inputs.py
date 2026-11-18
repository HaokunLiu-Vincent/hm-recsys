"""Read the saved files the diagnostics describe (IO only; the computing is elsewhere).

    artifacts/queries/{fold}.parquet                      the query list with customer_idx
    artifacts/predictions/.../{fold}_per_query.parquet    the Ranker's saved per-query values
    artifacts/features/{fold}/part-*.parquet              the frozen wide table (verified first)
    artifacts/candidates/{fold}/{channel}.parquet         the six channels' candidates
    data/processed/transactions.parquet                   purchase lines BEFORE an anchor only

Nothing is written here, and no model is loaded. Labels are not read here
either: where a function needs them it takes the fold's FoldEvaluator, which the
caller got from hmrec.eval.diagnostic_labels.
"""

from dataclasses import dataclass

import polars as pl

from hmrec.data.asof import history_window
from hmrec.data.day_index import date_to_day_index
from hmrec.data.folds import Fold
from hmrec.data.id_maps import ARTICLE_IDX, CUSTOMER_IDX
from hmrec.data.metadata import scan_processed_parquet
from hmrec.diagnostics.context import DiagnosticsContext
from hmrec.diagnostics.query_groups import DAYS, query_days
from hmrec.diagnostics.recorded import area_of_phase
from hmrec.eval.metrics import FOLD, QUERY_ID, VALUE
from hmrec.eval.report import FoldEvaluator
from hmrec.feature_files import verify_feature_files
from hmrec.fusion.rrf import BEST_RANK, RRF_SCORE, rrf_baseline
from hmrec.locks import ALL_CHANNELS
from hmrec.ranker.ablation import FULL

NUM_TREES = "num_trees"


@dataclass(frozen=True)
class WideSummary:
    """What the diagnostics take from a fold's wide table.

    Attributes:
        baseline_top: The RRF baseline's Top-K of every query: query_id,
            article_idx, rank (int32, 1 = best), score; sorted by (query_id, rank).
        days: One row per query: query_id, user_days_since_last_purchase.
    """

    baseline_top: pl.DataFrame
    days: pl.DataFrame


def read_queries(context: DiagnosticsContext, fold: Fold) -> pl.DataFrame:
    """The fold's query list: query_id (int32), fold (str), customer_idx (int32)."""
    columns = [QUERY_ID, FOLD, CUSTOMER_IDX]
    return pl.read_parquet(context.paths.queries_file(fold.name), columns=columns)


def read_ranker_per_query(context: DiagnosticsContext, fold: Fold) -> pl.DataFrame:
    """The locked Ranker's saved per-query AP of a fold, at the locked number of trees.

    The file was written when the fold was scored (M7 for V folds, the formal run
    for T folds); it holds one row per (query, number of trees).

    Returns:
        One row per query: query_id, fold, value (float64, AP@K); sorted by query_id.
    """
    ranker = context.ranker
    area = area_of_phase(fold.phase)
    path = context.paths.ranker_per_query_file(area, FULL, ranker.config_id, fold.name)
    columns = [QUERY_ID, FOLD, NUM_TREES, context.map_metric]
    saved = pl.read_parquet(path, columns=columns)
    at_locked_trees = saved.filter(pl.col(NUM_TREES) == ranker.num_trees)
    if at_locked_trees.height == 0:
        raise ValueError(f"{path} has no rows with {ranker.num_trees} trees")
    values = at_locked_trees.select(QUERY_ID, FOLD, pl.col(context.map_metric).alias(VALUE))
    return values.sort(QUERY_ID)


def read_wide_summary(context: DiagnosticsContext, fold: Fold) -> WideSummary:
    """The RRF baseline's Top-K and every query's days since the last purchase.

    The part files are verified against features.json first
    (hmrec.feature_files.verify_feature_files), so these are the frozen files.
    One part is in memory at a time, and only five of its columns. A query is
    never split over two parts, so the Top-K of a part is final.
    """
    features_dir = context.paths.features_dir(fold.name)
    feature_files = verify_feature_files(features_dir, context.ranker.wide_columns)
    columns = [QUERY_ID, ARTICLE_IDX, RRF_SCORE, BEST_RANK, DAYS]
    tops: list[pl.DataFrame] = []
    days: list[pl.DataFrame] = []
    for part in feature_files.parts:
        table = pl.read_parquet(features_dir / part.name, columns=columns)
        tops.append(rrf_baseline(table, context.ranker.top_k))
        days.append(query_days(table.select(QUERY_ID, DAYS)))
    # query_days once more over all parts: a query in two parts would repeat here.
    return WideSummary(baseline_top=pl.concat(tops), days=query_days(pl.concat(days)))


def read_union_hits(
    context: DiagnosticsContext, fold: Fold, evaluator: FoldEvaluator
) -> pl.DataFrame:
    """The candidate pairs of the six channels that are labels (for union recall).

    Built from the hits only, channel by channel, exactly as the recall monitor
    does (hmrec.eval.monitor.label_metrics): a candidate that is not a label
    cannot change union recall.

    Returns:
        Rows query_id, article_idx; a pair several channels returned repeats.
    """
    pair_columns = [QUERY_ID, ARTICLE_IDX]
    hits = []
    for channel in ALL_CHANNELS:
        path = context.paths.candidates_file(fold.name, channel)
        hits.append(evaluator.hits(pl.read_parquet(path, columns=pair_columns)))
    return pl.concat(hits)


def read_lines_before_anchor(context: DiagnosticsContext, fold: Fold) -> pl.DataFrame:
    """Every purchase line before the fold's anchor: t_dat_day in [0, anchor_day).

    The window starts at day 0 (the first day of the data) and ends BEFORE the
    anchor: the anchor day belongs to the label week. The date filter is done
    by hmrec.data.asof.history_window, which also checks its result.

    Returns:
        One row per purchase line: article_idx (int32), t_dat_day (int16).
    """
    paths, day_zero = context.paths, context.preprocess.day_zero
    transactions = scan_processed_parquet(
        paths.transactions_file, paths.processed_metadata_file, context.preprocess
    )
    days_since_day_zero = date_to_day_index(fold.anchor, day_zero)
    return history_window(
        transactions, fold.anchor, day_zero, days_since_day_zero, columns=[ARTICLE_IDX]
    )
