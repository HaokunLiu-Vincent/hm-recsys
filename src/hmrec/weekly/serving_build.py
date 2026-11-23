"""Build the serving form of a prediction set (docs/mlops.md §9; M14).

Step O5 calls build_serving_set after it wrote the prediction set; the result is
predictions/{anchor}/serving/ (the format is hmrec.serving_set). Two things are
added to the dense indexes of the prediction set:

    the original identifiers    customer_idx -> customer_id and article_idx ->
                                article_id, from the two id maps of
                                data/processed/ (the files the prediction set's
                                manifest names with their sha256).
    the fallback list           what an UNKNOWN customer gets: the first top_k
                                articles of R1's popularity list for the "age
                                missing" bucket of the same week.

Where the fallback list comes from. R1 gives every query the list of its
customer's age bucket (proposal §5 R1: the bucket's Top-quota of S(g, i), filled
with global popularity), ordered by rank. So the R1 candidates of ANY query whose
customer has no age ARE that bucket's list, and its first top_k rows are the
fallback list. It is read from the tick's own R1 output:

    eval    weeks/{anchor}/recall_r1_popularity/candidates.parquet
    serve   weeks/{anchor}/recall_r1_popularity/shards/{k}/candidates.parquet
            (the first shard that holds a customer without an age)

Nothing is computed again, and every such query of the file must hold the same
first top_k articles, else the build stops. A week none of whose queries has a
customer without an age has no such list in the tick's output: the build stops
then, because the service could not answer an unknown customer.

backfill_serving_set builds the same files for a prediction set that was
published before M14, from the state as it is; no step of the tick runs again.

Only data from before the anchor is behind these files: the prediction set and
the R1 candidates were computed by the tick, and the id maps hold no purchase.

This module imports neither torch nor lightgbm.
"""

import json
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from hmrec.data.metadata import scan_processed_parquet
from hmrec.locks import file_sha256
from hmrec.logging_utils import get_logger
from hmrec.paths import R1_CHANNEL_NAME, SERVING_DIR, ProjectPaths, StateLayout
from hmrec.run_manifest import SERVE_MODE
from hmrec.run_records import RunStamp
from hmrec.serving_set import (
    ARRAY_FILES,
    SERVING_RECORD_FILE,
    Fallback,
    ServingDescription,
    ServingRecord,
    write_serving_set,
)
from hmrec.weekly.context import OutputFile, TickContext, load_tick_context
from hmrec.weekly.shards import number_of_shards, shard_queries
from hmrec.weekly.state import RANKER_ROLE

LOGGER = get_logger("weekly.serving_build")

QUERY_ID = "query_id"
CUSTOMER_IDX = "customer_idx"
CUSTOMER_ID = "customer_id"
ARTICLE_IDX = "article_idx"
ARTICLE_ID = "article_id"
AGE_BUCKET = "age_bucket"
RANK = "rank"
BACKFILL = "backfill"


# ---------------------------------------------------------------- the fallback list


def bucket_list_of_queries(
    candidates: pl.DataFrame, bucket_queries: pl.DataFrame, top_k: int
) -> list[int]:
    """The first top_k articles of R1's list, read from the candidates of some queries.

    Args:
        candidates: One row per (query_id, rank) of R1 with rank <= top_k.
            Columns: query_id, rank, article_idx.
        bucket_queries: The queries whose customers are all in ONE age bucket
            (at least one). Column: query_id.
        top_k: Length of the list.

    Returns:
        top_k article_idx, rank 1 first.

    Raises:
        ValueError: If a rank 1..top_k is missing, or two of the queries hold
            different articles at a rank (they would not share one list).
    """
    rows = candidates.join(bucket_queries.select(QUERY_ID), on=QUERY_ID, how="semi")
    per_rank = (
        rows.group_by(RANK)
        .agg(
            pl.col(ARTICLE_IDX).n_unique().alias("n_articles"),
            pl.col(ARTICLE_IDX).first(),
            pl.len().alias("n_queries"),
        )
        .sort(RANK)
    )
    if per_rank[RANK].to_list() != list(range(1, top_k + 1)):
        raise ValueError(
            f"the R1 candidates of the bucket's queries do not hold the ranks 1..{top_k}"
        )
    if per_rank["n_articles"].max() != 1 or per_rank["n_queries"].n_unique() != 1:
        raise ValueError(
            "queries of one age bucket hold different R1 articles at the same rank: "
            "they do not share one list"
        )
    return per_rank[ARTICLE_IDX].to_list()


def r1_candidate_files(
    context: TickContext, queries: pl.DataFrame
) -> list[tuple[Path, pl.DataFrame]]:
    """The R1 candidate files of the tick's week, each with the queries it holds.

    eval: one file with every query. serve: one file per shard of customers.
    """
    view, week = context.view, context.tick.name
    if context.mode != SERVE_MODE:
        return [(view.candidates_file(week, R1_CHANNEL_NAME), queries)]
    size = context.pipeline.shard_customers
    n_shards = number_of_shards(queries.height, size)
    return [
        (
            view.for_shard(shard).candidates_file(week, R1_CHANNEL_NAME),
            shard_queries(queries, shard, size),
        )
        for shard in range(n_shards)
    ]


def fallback_list(
    context: TickContext, queries: pl.DataFrame, top_k: int
) -> tuple[list[int], Path]:
    """The fallback list of the tick's week and the R1 file it was read from.

    Args:
        context: The tick's context.
        queries: The week's query list. Columns: query_id, customer_idx.
        top_k: Length of the list.

    Returns:
        (top_k article_idx with rank 1 first, the R1 candidates file).

    Raises:
        ValueError: If no query of the week has a customer in the "age missing"
            bucket, or the bucket's queries do not share one list.
    """
    view = context.view
    bucket = context.preprocess.unknown_age_bucket
    customers = scan_processed_parquet(
        view.customers_file, view.processed_metadata_file, context.preprocess
    )
    without_age = customers.filter(pl.col(AGE_BUCKET) == bucket).select(CUSTOMER_IDX).collect()
    for candidates_file, file_queries in r1_candidate_files(context, queries):
        bucket_queries = file_queries.join(without_age, on=CUSTOMER_IDX, how="semi")
        if bucket_queries.height == 0:
            continue
        candidates = (
            pl.scan_parquet(candidates_file)
            .filter(pl.col(RANK) <= top_k)
            .select(QUERY_ID, RANK, ARTICLE_IDX)
            .collect()
        )
        articles = bucket_list_of_queries(candidates, bucket_queries, top_k)
        LOGGER.info(
            "fallback list: bucket %s, %d queries of %s share it",
            bucket,
            bucket_queries.height,
            candidates_file,
        )
        return articles, candidates_file
    raise ValueError(
        f"no query of week {context.tick.name} has a customer in age bucket {bucket!r}: the "
        "tick's R1 output holds no list for unknown customers, so the fallback list of the "
        "service cannot be built"
    )


# ---------------------------------------------------------------- the arrays


def ids_as_bytes(ids: pl.Series, what: str) -> np.ndarray:
    """A column of text ids as a numpy array of bytes (one fixed width).

    Raises:
        ValueError: If an id is missing or is not ASCII text.
    """
    if ids.null_count() > 0:
        raise ValueError(f"{ids.null_count()} {what} ids are missing in the id map")
    try:
        return np.array(ids.to_list(), dtype=np.bytes_)
    except UnicodeEncodeError as error:
        raise ValueError(f"a {what} id is not ASCII text: {error}") from error


def serving_arrays(
    predictions: pl.DataFrame,
    customer_id_map: pl.DataFrame,
    article_id_map: pl.DataFrame,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The three arrays of a serving set (hmrec.serving_set) from a prediction table.

    Args:
        predictions: One row per (customer, rank), sorted by (customer_idx, rank),
            exactly top_k rows per customer. Columns: customer_idx, rank,
            article_idx (int32).
        customer_id_map: customer_idx -> customer_id (text).
        article_id_map: article_idx -> article_id (text); the indexes must be
            0 .. n_articles - 1.
        top_k: Articles per customer.

    Returns:
        (customer_ids, top_articles, article_ids): bytes [n_customers], sorted
        ascending; int32 [n_customers, top_k], row i for customer_ids[i]; bytes
        [n_articles], position = article_idx.

    Raises:
        ValueError: If the table is not top_k rows per customer at ranks
            1..top_k in order, or an id map does not cover the table.
    """
    if predictions.height == 0 or predictions.height % top_k != 0:
        raise ValueError(f"{predictions.height} prediction rows are not {top_k} per customer")
    n_customers = predictions.height // top_k
    ranks = predictions[RANK].to_numpy().reshape(n_customers, top_k)
    customers = predictions[CUSTOMER_IDX].to_numpy().reshape(n_customers, top_k)
    if not bool(np.all(ranks == np.arange(1, top_k + 1))):
        raise ValueError(f"the prediction rows are not ranks 1..{top_k} for every customer")
    if not bool(np.all(customers == customers[:, :1])):
        raise ValueError(f"a customer of the prediction table does not have {top_k} rows in a row")
    top_articles = predictions[ARTICLE_IDX].to_numpy().reshape(n_customers, top_k)
    customer_idx = pl.DataFrame({CUSTOMER_IDX: customers[:, 0]}, schema={CUSTOMER_IDX: pl.Int32})
    # A left join keeps the table's order, so row i of the ids belongs to row i of the Top-K.
    with_ids = customer_idx.join(
        customer_id_map.select(CUSTOMER_IDX, CUSTOMER_ID), on=CUSTOMER_IDX, how="left"
    )
    customer_ids = ids_as_bytes(with_ids[CUSTOMER_ID], "customer")
    articles = article_id_map.sort(ARTICLE_IDX)
    if articles[ARTICLE_IDX].to_list() != list(range(articles.height)):
        raise ValueError("the article id map does not hold the indexes 0 .. n_articles - 1")
    article_ids = ids_as_bytes(articles[ARTICLE_ID], "article")
    # Sorted by the id's bytes with numpy, the library that searches them later.
    order = np.argsort(customer_ids, kind="stable")
    return (
        customer_ids[order],
        np.ascontiguousarray(top_articles[order], dtype=np.int32),
        article_ids,
    )


# ---------------------------------------------------------------- build (O5) and backfill


def build_serving_set(
    context: TickContext,
    folder: Path,
    predictions: pl.DataFrame,
    prediction_sha256: str,
    ranker_sha256: str,
    built_by: str,
) -> ServingRecord:
    """Write the serving form of the tick's prediction set into `folder`.

    Args:
        context: The tick's context.
        folder: The serving folder to write (it must not hold a record).
        predictions: The prediction table of the week: customer_idx, rank,
            article_idx, sorted by (customer_idx, rank).
        prediction_sha256: sha256 of the prediction set's top.parquet.
        ranker_sha256: sha256 of the Ranker file behind the predictions.
        built_by: The run id of the tick, or "backfill".

    Raises:
        ValueError: If the fallback list or the arrays cannot be built.
        FileExistsError: If the folder already holds a serving record.
    """
    view, preprocess = context.view, context.preprocess
    metadata = view.processed_metadata_file
    queries = pl.read_parquet(
        view.queries_file(context.tick.name), columns=[QUERY_ID, CUSTOMER_IDX]
    ).sort(QUERY_ID)
    top_k = int(predictions[RANK].to_numpy().max())
    fallback_idx, r1_file = fallback_list(context, queries, top_k)
    customer_id_map = scan_processed_parquet(
        view.customer_id_map_file, metadata, preprocess
    ).collect()
    article_id_map = scan_processed_parquet(
        view.article_id_map_file, metadata, preprocess
    ).collect()
    customer_ids, top_articles, article_ids = serving_arrays(
        predictions, customer_id_map, article_id_map, top_k
    )
    fallback = Fallback(
        age_bucket=preprocess.unknown_age_bucket,
        article_ids=tuple(article_ids[index].decode("ascii") for index in fallback_idx),
        source=r1_file.relative_to(context.layout.root).as_posix(),
        source_sha256=file_sha256(r1_file),
    )
    description = ServingDescription(
        anchor=context.tick.anchor.isoformat(),
        week=context.tick.name,
        mode=context.mode,
        prediction_sha256=prediction_sha256,
        ranker_sha256=ranker_sha256,
        fallback=fallback,
        built_by=built_by,
        git_commit=context.stamp.git_commit,
    )
    record = write_serving_set(folder, description, customer_ids, top_articles, article_ids)
    LOGGER.info(
        "serving set %s: %d customers x %d articles -> %s",
        record.set_id,
        record.n_customers,
        record.top_k,
        folder,
    )
    return record


def serving_output_files() -> list[OutputFile]:
    """The files of a serving set as outputs of step O5 (paths inside the prediction set)."""
    names = [*ARRAY_FILES, SERVING_RECORD_FILE]
    return [OutputFile(f"serving_set/{Path(name).stem}", f"{SERVING_DIR}/{name}") for name in names]


def backfill_serving_set(paths: ProjectPaths, anchor: date, stamp: RunStamp) -> ServingRecord:
    """Build the serving form of a prediction set that is already published.

    For the sets of the replay, which were published before M14. Reads the
    prediction set, the R1 candidates of its tick and the id maps; runs no step
    of the tick again and changes no existing file.

    Args:
        paths: The project paths of the state (hmrec.paths.tick_root_paths).
        anchor: The anchor of the prediction set.
        stamp: The commit of the code and whether it is dirty.

    Raises:
        FileNotFoundError: If the prediction set does not exist.
        FileExistsError: If it already has a serving folder (complete or not:
            nothing is replaced, and a half-written one is for a person to look at).
        ValueError: If the prediction file does not have its recorded sha256.
    """
    layout = StateLayout(paths.pipeline_state_dir)
    manifest_file = layout.prediction_manifest_file(anchor)
    if not manifest_file.is_file():
        raise FileNotFoundError(f"{manifest_file} not found: no prediction set for {anchor}")
    folder = layout.serving_dir(anchor)
    if folder.exists():
        raise FileExistsError(f"{folder} already exists; a serving set is never replaced")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    predictions_file = layout.prediction_file(anchor)
    sha256 = file_sha256(predictions_file)
    if sha256 != manifest["predictions"]["sha256"]:
        raise ValueError(
            f"{predictions_file}: sha256 {sha256} is not the recorded "
            f"{manifest['predictions']['sha256']}"
        )
    rankers = [model["sha256"] for model in manifest["models"] if model["role"] == RANKER_ROLE]
    if len(rankers) != 1:
        raise ValueError(f"{manifest_file} names {len(rankers)} Ranker models, expected 1")
    context = load_tick_context(paths, anchor, manifest["mode"], BACKFILL, stamp)
    predictions = pl.read_parquet(predictions_file)
    return build_serving_set(context, folder, predictions, sha256, rankers[0], BACKFILL)
