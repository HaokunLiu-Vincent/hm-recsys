"""What a tick puts out: the checked Top-K, the output signals, the prediction set (O4, O5).

O4 check (docs/mlops.md §6.1): every customer of the week gets exactly K distinct,
valid article ids (K = the locked top_k, 12).

Output signals (docs/mlops.md §10; they need no labels):

    candidates_per_query     mean number of deduplicated candidates per query
    top_popular_share        share of the recommended (customer, article) slots
                             whose article is among the N most popular articles
                             of the week: the N articles with the most purchase
                             lines in the last days before the anchor
                             (configs/m12_pipeline.yaml: N = 100, 7 days)
    distinct_articles        number of distinct articles recommended to anyone

O5: the prediction set of the week, predictions/{anchor}/:

    top.parquet       one row per (customer, rank): customer_idx (int32), rank
                      (int32, 1 = best), article_idx (int32); sorted by
                      (customer_idx, rank). The indexes are those of
                      data/processed/ (customer_id_map.parquet,
                      article_id_map.parquet), named in the manifest with sha256.
    manifest.json     week, mode, number of customers, the file's sha256, and the
                      model files behind the predictions with their sha256.
    serving/          the same predictions in the form the recommendation service
                      loads: original customer and article ids, and the fallback
                      list for unknown customers (hmrec.serving_set,
                      hmrec.weekly.serving_build; M14).

The `current` pointer is switched by the driver after this step's folder is in
place (hmrec.weekly.driver), and only then.

This module imports neither torch nor lightgbm.
"""

import csv
import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from hmrec.data.asof import history_window
from hmrec.data.metadata import scan_processed_parquet
from hmrec.locks import file_sha256
from hmrec.logging_utils import get_logger
from hmrec.paths import MODEL_DIR, PREDICTION_FILE, PREDICTION_MANIFEST_FILE, SERVING_DIR
from hmrec.run_manifest import SERVE_MODE, MetricRecord
from hmrec.weekly.context import OutputFile, TickContext
from hmrec.weekly.serving_build import build_serving_set, serving_output_files
from hmrec.weekly.shard_features import N_CANDIDATES, QUERY_STATS_FILE
from hmrec.weekly.shards import number_of_shards
from hmrec.weekly.state import RANKER_ROLE, TWOTOWER_ROLE, ModelPointer

LOGGER = get_logger("weekly.outputs")

QUERY_ID = "query_id"
CUSTOMER_IDX = "customer_idx"
ARTICLE_IDX = "article_idx"
RANK = "rank"
SCORE = "score"
TOP_COLUMNS = [QUERY_ID, RANK, ARTICLE_IDX, SCORE]
PREDICTION_COLUMNS = [CUSTOMER_IDX, RANK, ARTICLE_IDX]
PREDICTION_SCHEMA_VERSION = 1
BOOTSTRAP_CHAIN_TABLE = "m6_twotower_chain.csv"

StepOutcome = tuple[str, list[MetricRecord], list[OutputFile]]


# ---------------------------------------------------------------- what C2 measured


@dataclass(frozen=True)
class WeekEvaluation:
    """What step C2 measured for a closed week (weeks/{anchor}/evaluation/metrics.json).

    Attributes:
        week: The closed week's name, e.g. "V1".
        anchor: Its anchor (ISO date).
        n_queries: Queries of the week (customers who bought a catalog article).
        n_label_pairs: (query, article) pairs bought in the week.
        channel_recall_metric: Channel -> its metric, e.g. "recall@100".
        channel_recall: Channel -> mean over queries of that metric (float64).
        union_recall: Mean over queries of the union recall of the six channels.
        zero_hit_query_share: Share of queries for which no channel returned a label.
        ranker: Metric -> mean over queries, for the published Top-K (e.g. "map@12").
        rrf_baseline: The same metrics for the RRF baseline of the same candidates.
        hit_value: Sum of the prices of recommended articles the customers bought
            (hmrec.eval.hit_value).
    """

    week: str
    anchor: str
    n_queries: int
    n_label_pairs: int
    channel_recall_metric: dict[str, str]
    channel_recall: dict[str, float]
    union_recall: float
    zero_hit_query_share: float
    ranker: dict[str, float]
    rrf_baseline: dict[str, float]
    hit_value: float


def write_week_evaluation(path: Path, evaluation: WeekEvaluation) -> None:
    """Write evaluation/metrics.json (floats as JSON numbers: they read back exactly)."""
    path.write_text(json.dumps(asdict(evaluation), indent=2) + "\n", encoding="utf-8")


def read_week_evaluation(path: Path) -> WeekEvaluation:
    """Read evaluation/metrics.json of a closed week."""
    return WeekEvaluation(**json.loads(path.read_text(encoding="utf-8")))


# ---------------------------------------------------------------- O4: the check


def check_top_k(top: pl.DataFrame, queries: pl.DataFrame, top_k: int, n_articles: int) -> None:
    """Fail unless every query has exactly `top_k` distinct, valid articles at ranks 1 .. top_k.

    Args:
        top: One row per (query_id, rank): query_id, rank, article_idx.
        queries: The queries that must be covered: query_id.
        top_k: Articles per query (12).
        n_articles: Size of the catalog; a valid article_idx is in 0 .. n_articles - 1.

    Raises:
        ValueError: With the first rule that is broken.
    """
    known = queries.select(QUERY_ID)
    if top.join(known, on=QUERY_ID, how="anti").height > 0:
        raise ValueError("the Top-K holds a query_id that is not a query of the week")
    per_query = top.group_by(QUERY_ID).agg(
        pl.len().alias("n_rows"),
        pl.col(ARTICLE_IDX).n_unique().alias("n_articles"),
        pl.col(RANK).min().alias("first_rank"),
        pl.col(RANK).max().alias("last_rank"),
        pl.col(RANK).n_unique().alias("n_ranks"),
    )
    if per_query.height != known.height:
        raise ValueError(
            f"{known.height - per_query.height} of {known.height} queries have no Top-{top_k}"
        )
    complete = (
        (pl.col("n_rows") == top_k)
        & (pl.col("n_articles") == top_k)
        & (pl.col("n_ranks") == top_k)
        & (pl.col("first_rank") == 1)
        & (pl.col("last_rank") == top_k)
    )
    incomplete = per_query.filter(~complete).height
    if incomplete > 0:
        raise ValueError(
            f"{incomplete} queries do not have exactly {top_k} distinct articles at ranks "
            f"1..{top_k}"
        )
    invalid = top.filter((pl.col(ARTICLE_IDX) < 0) | (pl.col(ARTICLE_IDX) >= n_articles)).height
    if invalid > 0:
        raise ValueError(f"{invalid} recommended article ids are outside 0..{n_articles - 1}")


# ---------------------------------------------------------------- output signals


@dataclass(frozen=True)
class OutputSignals:
    """The output signals of one week (weeks/{anchor}/scores/signals.json).

    Attributes:
        n_queries: Customers predicted for.
        candidates_per_query: Mean deduplicated candidates per query.
        top_popular_share: Share of recommended slots among the most popular articles.
        distinct_articles: Distinct articles recommended to anyone.
        popular_articles: How many "most popular" articles were counted (100).
        popular_window_days: Days before the anchor that define "most popular".
    """

    n_queries: int
    candidates_per_query: float
    top_popular_share: float
    distinct_articles: int
    popular_articles: int
    popular_window_days: int


def popular_articles(
    transactions: pl.LazyFrame, anchor: date, day_zero: date, window_days: int, n_articles: int
) -> pl.DataFrame:
    """The `n_articles` articles with the most purchase lines in [anchor - window_days, anchor).

    Ties are broken by the smaller article_idx. The window is cut with
    hmrec.data.asof.history_window; the anchor day is not in it.

    Returns:
        One row per article, most popular first. Column: article_idx (int32).
    """
    lines = history_window(transactions, anchor, day_zero, window_days, columns=[ARTICLE_IDX])
    counts = lines.group_by(ARTICLE_IDX).agg(pl.len().alias("n_lines"))
    return (
        counts.sort(["n_lines", ARTICLE_IDX], descending=[True, False])
        .head(n_articles)
        .select(ARTICLE_IDX)
    )


def output_signals(
    top: pl.DataFrame,
    popular: pl.DataFrame,
    n_queries: int,
    n_candidates: int,
    popular_window_days: int,
) -> OutputSignals:
    """Compute the three output signals from the week's Top-K.

    Args:
        top: One row per (query_id, rank) with article_idx.
        popular: The most popular articles (popular_articles).
        n_queries: Queries of the week.
        n_candidates: Deduplicated candidate rows of the week.
        popular_window_days: The window the popular articles were counted in.
    """
    in_popular = top.join(popular, on=ARTICLE_IDX, how="semi").height
    return OutputSignals(
        n_queries=n_queries,
        candidates_per_query=n_candidates / n_queries,
        top_popular_share=in_popular / top.height,
        distinct_articles=top[ARTICLE_IDX].n_unique(),
        popular_articles=popular.height,
        popular_window_days=popular_window_days,
    )


def write_checked_top(
    context: TickContext, top: pl.DataFrame, n_candidates: int
) -> tuple[OutputSignals, list[OutputFile]]:
    """Check the week's Top-K, then write it with its output signals (scores/).

    Args:
        context: The process's context.
        top: The Top-K of every query of the week: query_id, rank, article_idx, score.
        n_candidates: Deduplicated candidate rows of the week.

    Raises:
        ValueError: If a query does not have top_k distinct valid articles.
    """
    view, week, preprocess = context.view, context.tick, context.preprocess
    metadata = view.processed_metadata_file
    queries = pl.read_parquet(view.queries_file(week.name), columns=[QUERY_ID])
    articles = scan_processed_parquet(view.articles_file, metadata, preprocess)
    n_articles = int(articles.select(pl.len()).collect().item())
    top_k = int(top[RANK].to_numpy().max())
    check_top_k(top, queries, top_k, n_articles)
    pipeline = context.pipeline
    popular = popular_articles(
        scan_processed_parquet(view.transactions_file, metadata, preprocess),
        week.anchor,
        preprocess.day_zero,
        pipeline.popular_window_days,
        pipeline.popular_articles,
    )
    signals = output_signals(
        top, popular, queries.height, n_candidates, pipeline.popular_window_days
    )
    top_file = view.top_file(week.name)
    top_file.parent.mkdir(parents=True, exist_ok=True)
    top.select(TOP_COLUMNS).sort(QUERY_ID, RANK).write_parquet(top_file)
    signals_file = view.signals_file(week.name)
    signals_file.write_text(json.dumps(asdict(signals), indent=2) + "\n", encoding="utf-8")
    LOGGER.info(
        "O4: Top-%d of %d queries checked -> %s; %s", top_k, queries.height, top_file, signals
    )
    files = [OutputFile("top_k", top_file.name), OutputFile("output_signals", signals_file.name)]
    return signals, files


def signal_metrics(anchor: date, signals: OutputSignals) -> list[MetricRecord]:
    """The output signals as manifest metrics of the tick's week."""
    return [
        MetricRecord("output/candidates_per_query", anchor, signals.candidates_per_query),
        MetricRecord("output/top_popular_share", anchor, signals.top_popular_share),
        MetricRecord("output/distinct_articles", anchor, float(signals.distinct_articles)),
    ]


def read_signals(path: Path) -> OutputSignals:
    """Read weeks/{anchor}/scores/signals.json."""
    return OutputSignals(**json.loads(path.read_text(encoding="utf-8")))


# ---------------------------------------------------------------- serve: merge the shards


def run_merge_scores(context: TickContext) -> StepOutcome:
    """O4 in `serve` mode: put the shards' Top-K together, check it, write the signals.

    Reads shards/{k}/top/top.parquet and the query statistics the scoring step
    kept next to it. Every shard must be there.

    Raises:
        FileNotFoundError: If a shard was not scored.
        ValueError: If a customer does not have top_k distinct valid articles.
    """
    view, week = context.view, context.tick.name
    n_queries = int(pl.scan_parquet(view.queries_file(week)).select(pl.len()).collect().item())
    n_shards = number_of_shards(n_queries, context.pipeline.shard_customers)
    tops, n_candidates = [], 0
    for shard in range(n_shards):
        top_file = view.for_shard(shard).top_file(week)
        if not top_file.is_file():
            raise FileNotFoundError(f"{top_file} not found: shard {shard} is not scored")
        tops.append(pl.read_parquet(top_file, columns=TOP_COLUMNS))
        stats = pl.read_parquet(top_file.with_name(QUERY_STATS_FILE), columns=[N_CANDIDATES])
        n_candidates += int(stats[N_CANDIDATES].sum())
    signals, files = write_checked_top(context, pl.concat(tops), n_candidates)
    detail = (
        f"Top-K of {signals.n_queries} customers from {n_shards} shards; "
        f"{signals.candidates_per_query:.1f} candidates per query, "
        f"{signals.distinct_articles} distinct articles"
    )
    return detail, signal_metrics(context.tick.anchor, signals), files


# ---------------------------------------------------------------- O5: the prediction set


def chain_model_pointer(context: TickContext) -> ModelPointer:
    """The R6 chain model that predicted the tick's week: the model after the closed week.

    Read from the chain record (file name and sha256) with the csv module:
    hmrec.recall.chain_records would load torch, and this step is on no side.
    The file is hashed again here and must have the recorded sha256.

    Raises:
        ValueError: If the closed week has no chain record or the file differs.
    """
    closing, paths, view = context.closing, context.paths, context.view
    if closing.ordinal < context.first_week.ordinal:
        table = paths.bootstrap_monitoring_file(BOOTSTRAP_CHAIN_TABLE)
    else:
        table = view.chain_record_file(closing.name)
    with table.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["fold"] == closing.name]
    if len(rows) != 1:
        raise ValueError(f"{table} has {len(rows)} rows for week {closing.name}, expected 1")
    model_file = view.twotower_locked_chain_checkpoint_file(closing.name)
    sha256 = file_sha256(model_file)
    if sha256 != rows[0]["sha256"]:
        raise ValueError(f"{model_file}: sha256 {sha256} is not the recorded {rows[0]['sha256']}")
    return ModelPointer(TWOTOWER_ROLE, relative_to_root(context, model_file), sha256)


def relative_to_root(context: TickContext, path: Path) -> str:
    """A path relative to the state root if it lies in it, else to the repository root.

    A file of inputs that lie outside the repository (a mounted bucket) is named
    as the run manifest names it (TickContext.recorded_location).
    """
    for root in (context.layout.root, context.paths.project_root):
        if root == path or root in path.parents:
            return path.relative_to(root).as_posix()
    return context.recorded_location(path)[1]


def ranker_model_pointer(context: TickContext) -> ModelPointer:
    """The Ranker that scored the tick's week, with the sha256 its training recorded."""
    view, week = context.view, context.tick.name
    record = json.loads(view.model_record_file(week).read_text(encoding="utf-8"))
    model_file = view.week_step_dir(week, MODEL_DIR) / "model.txt"
    sha256 = file_sha256(model_file)
    if sha256 != record["sha256"]:
        raise ValueError(f"{model_file}: sha256 {sha256} is not the recorded {record['sha256']}")
    return ModelPointer(RANKER_ROLE, relative_to_root(context, model_file), sha256)


def prediction_table(top: pl.DataFrame, queries: pl.DataFrame) -> pl.DataFrame:
    """The Top-K by customer: customer_idx, rank, article_idx (int32), sorted.

    Raises:
        ValueError: If a query of the Top-K has no customer.
    """
    joined = top.select(QUERY_ID, RANK, ARTICLE_IDX).join(
        queries.select(QUERY_ID, CUSTOMER_IDX), on=QUERY_ID, how="left"
    )
    if joined[CUSTOMER_IDX].null_count() > 0:
        raise ValueError("the Top-K holds a query_id that is not a query of the week")
    return joined.select(
        pl.col(CUSTOMER_IDX).cast(pl.Int32),
        pl.col(RANK).cast(pl.Int32),
        pl.col(ARTICLE_IDX).cast(pl.Int32),
    ).sort(CUSTOMER_IDX, RANK)


def run_publish(context: TickContext) -> StepOutcome:
    """O5: write the prediction set of the week (the pointer is switched by the driver).

    The serving form of the set is written with it (hmrec.weekly.serving_build), so
    that the service can load a set as soon as the pointer names it.

    Raises:
        ValueError: If a model file does not have its recorded sha256, or the
            serving form cannot be built (for example: no fallback list).
    """
    view, week, paths = context.view, context.tick, context.paths
    top = pl.read_parquet(view.top_file(week.name))
    queries = pl.read_parquet(view.queries_file(week.name))
    predictions = prediction_table(top, queries)
    folder = context.layout.prediction_staging_dir(week.anchor)
    folder.mkdir(parents=True, exist_ok=True)
    predictions_file = folder / PREDICTION_FILE
    predictions.write_parquet(predictions_file)
    models = [ranker_model_pointer(context), chain_model_pointer(context)]
    id_maps = {
        "customer": paths.customer_id_map_file,
        "article": paths.article_id_map_file,
    }
    content: dict[str, Any] = {
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "anchor": week.anchor.isoformat(),
        "week": week.name,
        "mode": context.mode,
        "n_customers": queries.height,
        "top_k": int(predictions[RANK].to_numpy().max()),
        "predictions": {
            "file": PREDICTION_FILE,
            "columns": PREDICTION_COLUMNS,
            "n_rows": predictions.height,
            "sha256": file_sha256(predictions_file),
        },
        "models": [asdict(model) for model in models],
        "id_maps": {
            name: {"path": relative_to_root(context, path), "sha256": file_sha256(path)}
            for name, path in id_maps.items()
        },
        "run_id": context.run_id,
        "git_commit": context.stamp.git_commit,
    }
    (folder / PREDICTION_MANIFEST_FILE).write_text(
        json.dumps(content, indent=2) + "\n", encoding="utf-8"
    )
    build_serving_set(
        context,
        folder / SERVING_DIR,
        predictions,
        content["predictions"]["sha256"],
        models[0].sha256,
        context.run_id,
    )
    detail = f"prediction set of {queries.height} customers x {content['top_k']} articles"
    if context.mode == SERVE_MODE:
        detail += " (every customer of the customer table)"
    files = [
        OutputFile("prediction_set", PREDICTION_FILE),
        OutputFile("prediction_set_manifest", PREDICTION_MANIFEST_FILE),
        *serving_output_files(),
    ]
    return detail, [MetricRecord("output/customers", week.anchor, float(queries.height))], files
