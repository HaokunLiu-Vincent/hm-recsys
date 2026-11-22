"""The data steps of a tick: C1 ingest, the query list, the labels of a closed week.

C1 INGEST is where the simulated clock is enforced (docs/mlops.md §6.3). The
data end on 2020-09-22, so "today" is simulated: the tick at anchor `a` copies
from the full table data/processed/transactions.parquet (the SOURCE) every
purchase line with t_dat < a into the tick's own folder, in the source's row
order. Every later step of the tick reads THAT file (hmrec.paths.PipelinePaths
.transactions_file) and never the source, so it cannot use a later purchase:
its input does not hold one. The stage-1 filters (hmrec.data.asof) still run on
top of it, unchanged.

Two readers of the source exist, and no other:

    C1 ingest            cuts the source at the anchor (hmrec.data.asof).
    the query list in    `eval` mode predicts only for the customers who WILL buy
    `eval` mode          in the week (docs/mlops.md §6.2), so that the replay can
                         be compared with the stage-1 records. That needs the
                         week's own purchases: it reads WHO bought, nothing else.
                         A real system cannot know this; `serve` mode predicts
                         for every customer and reads no later row.

The labels of a week are built when the week is CLOSED, by the next tick, from
the transactions that tick ingested (the week is then before its anchor). The
query list written a week earlier must be the one those labels give.

This module imports neither torch nor lightgbm.
"""

import json
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from hmrec.config import load_monitoring_config, load_queries_config
from hmrec.data.asof import TRANSACTION_DAY, history_window
from hmrec.data.catalog import build_catalog
from hmrec.data.day_index import date_to_day_index, day_index_to_date
from hmrec.data.folds import Fold
from hmrec.data.id_maps import CUSTOMER_IDX
from hmrec.data.metadata import scan_processed_parquet
from hmrec.data.queries import (
    ANCHOR,
    FOLD,
    QUERIES_SCHEMA,
    QUERY_ID,
    QUERY_ID_DTYPE,
    FoldQueries,
    build_fold_queries,
    check_fold_queries,
    label_week_purchase_values,
)
from hmrec.logging_utils import get_logger
from hmrec.paths import ProjectPaths
from hmrec.run_manifest import EVAL_MODE, MetricRecord
from hmrec.weekly.calendar import DAYS_PER_WEEK, query_id_base
from hmrec.weekly.context import OutputFile, TickContext

LOGGER = get_logger("weekly.data_steps")

TRANSACTIONS_ROLE = "transactions_before_anchor"
QUERIES_ROLE = "queries"

# What a step function returns: (detail text, metrics, output files).
StepOutcome = tuple[str, list[MetricRecord], list[OutputFile]]


# ---------------------------------------------------------------- C1 ingest


@dataclass(frozen=True)
class IngestRecord:
    """What C1 wrote and saw (weeks/{anchor}/ingest/ingest.json).

    Attributes:
        anchor: The tick's anchor (ISO date).
        n_rows: Purchase lines in the ingested table (all with t_dat < anchor).
        first_date: Date of the oldest row (ISO).
        last_date: Date of the newest row (ISO); it is before the anchor.
        new_week_rows: Rows of the new week [anchor - 7, anchor).
        new_week_first_date: Date of its oldest row (ISO), "" if it has none.
        new_week_last_date: Date of its newest row (ISO), "" if it has none.
        expected_first_date: anchor - 7 days (ISO).
        expected_last_date: anchor - 1 day (ISO).
        previous_week_rows: Rows of the weeks right before the new week, the
            oldest first: [anchor - 7 (n + 1), anchor - 7 n), ..., [anchor - 14, anchor - 7).
    """

    anchor: str
    n_rows: int
    first_date: str
    last_date: str
    new_week_rows: int
    new_week_first_date: str
    new_week_last_date: str
    expected_first_date: str
    expected_last_date: str
    previous_week_rows: tuple[int, ...]


def ingest_table(source: pl.LazyFrame, anchor: date, day_zero: date) -> pl.DataFrame:
    """Every purchase line of the source with t_dat before the anchor, in the source's order.

    The cut is made by hmrec.data.asof.history_window (the only date filter of
    the project), with a window [day 0, anchor): t_dat_day 0 is day_zero, the
    first day of the data.

    Args:
        source: The full processed transactions (one row per purchase line).
        anchor: The tick's anchor.
        day_zero: Day 0 of t_dat_day.

    Returns:
        The rows with t_dat_day < anchor day, all columns, in the input row order.
    """
    anchor_day = date_to_day_index(anchor, day_zero)
    return history_window(source, anchor, day_zero, lookback_days=anchor_day)


def describe_ingested(
    ingested: pl.DataFrame, anchor: date, day_zero: date, n_previous_weeks: int
) -> IngestRecord:
    """Count the rows of the new week and of the weeks before it.

    Week k (k = 0 is the new week) holds the days [anchor - 7 (k + 1), anchor - 7 k).

    Args:
        ingested: The ingested table; column t_dat_day (int16).
        anchor: The tick's anchor.
        day_zero: Day 0 of t_dat_day.
        n_previous_weeks: How many weeks before the new week to count.

    Raises:
        ValueError: If the table is empty or holds a row on or after the anchor.
    """
    if ingested.height == 0:
        raise ValueError(f"no purchase line before the anchor {anchor}")
    anchor_day = date_to_day_index(anchor, day_zero)
    # int64 on purpose: the int16 day index would overflow in the arithmetic below.
    days = ingested[TRANSACTION_DAY].to_numpy().astype(np.int64)
    first_day, last_day = int(days.min()), int(days.max())
    if last_day >= anchor_day:
        raise ValueError(
            f"the ingested table holds t_dat_day {last_day}, which is not before the anchor "
            f"day {anchor_day}: the clock is broken"
        )
    # Days back from the anchor: yesterday is 1. Week k holds the days 7 k + 1 ... 7 k + 7 back.
    weeks_back = (anchor_day - 1 - days) // DAYS_PER_WEEK
    rows_by_week = np.bincount(weeks_back, minlength=n_previous_weeks + 1)
    new_week = days[weeks_back == 0]

    def iso_date(day: int) -> str:
        return day_index_to_date(day, day_zero).isoformat()

    return IngestRecord(
        anchor=anchor.isoformat(),
        n_rows=ingested.height,
        first_date=iso_date(first_day),
        last_date=iso_date(last_day),
        new_week_rows=int(rows_by_week[0]),
        new_week_first_date=iso_date(int(new_week.min())) if new_week.size > 0 else "",
        new_week_last_date=iso_date(int(new_week.max())) if new_week.size > 0 else "",
        expected_first_date=(anchor - timedelta(days=DAYS_PER_WEEK)).isoformat(),
        expected_last_date=(anchor - timedelta(days=1)).isoformat(),
        previous_week_rows=tuple(
            int(rows_by_week[back]) for back in range(n_previous_weeks, 0, -1)
        ),
    )


def read_ingest_record(path: Path) -> IngestRecord:
    """Read weeks/{anchor}/ingest/ingest.json."""
    content = json.loads(path.read_text(encoding="utf-8"))
    content["previous_week_rows"] = tuple(content["previous_week_rows"])
    return IngestRecord(**content)


def scan_source_transactions(context: TickContext) -> pl.LazyFrame:
    """The SOURCE transactions (data/processed/); only C1 and the `eval` query list call this."""
    paths = context.paths
    return scan_processed_parquet(
        paths.transactions_file, paths.processed_metadata_file, context.preprocess
    )


def scan_ingested_transactions(context: TickContext) -> pl.LazyFrame:
    """The tick's ingested transactions: every row is before the tick's anchor."""
    return scan_processed_parquet(
        context.view.transactions_file, context.paths.processed_metadata_file, context.preprocess
    )


def run_ingest(context: TickContext) -> StepOutcome:
    """C1: write the transactions before the anchor and count the new week.

    The written file is read back and checked: it must hold exactly the rows
    that were cut, and none on or after the anchor.

    Raises:
        ValueError: If the file read back differs, or holds a row on or after the anchor.
    """
    anchor, day_zero = context.tick.anchor, context.preprocess.day_zero
    ingested = ingest_table(scan_source_transactions(context), anchor, day_zero)
    target = context.view.transactions_file
    target.parent.mkdir(parents=True, exist_ok=True)
    ingested.write_parquet(target)
    written = scan_ingested_transactions(context)
    summary = written.select(pl.len().alias("n"), pl.col(TRANSACTION_DAY).max().alias("last"))
    n_written, last_written = summary.collect().row(0)
    anchor_day = date_to_day_index(anchor, day_zero)
    if n_written != ingested.height or last_written >= anchor_day:
        raise ValueError(
            f"{target}: read back {n_written} rows up to t_dat_day {last_written}; expected "
            f"{ingested.height} rows, all before the anchor day {anchor_day}"
        )
    history_weeks = load_monitoring_config(context.paths.monitoring_config_file).history_weeks
    record = describe_ingested(ingested, anchor, day_zero, history_weeks)
    context.view.ingest_record_file.write_text(
        json.dumps(asdict(record), indent=2) + "\n", encoding="utf-8"
    )
    LOGGER.info("C1: %d rows before %s -> %s; %s", record.n_rows, anchor, target, record)
    detail = (
        f"{record.n_rows} purchase lines before {anchor} (newest {record.last_date}); "
        f"new week {record.new_week_rows} rows"
    )
    metrics = [
        MetricRecord("input/transactions_rows_new_week", anchor, float(record.new_week_rows)),
        MetricRecord("input/transactions_rows_before_anchor", anchor, float(record.n_rows)),
    ]
    return detail, metrics, [OutputFile(TRANSACTIONS_ROLE, target.name)]


# ---------------------------------------------------------------- the query list


def build_serve_queries(customer_idx: np.ndarray, week: Fold, first_query_id: int) -> pl.DataFrame:
    """One query per customer for the week (`serve` mode).

    Customers are sorted by customer_idx and numbered 0 .. n - 1 (position);
    query_id = first_query_id + position, as for a fold (hmrec.data.queries).

    Args:
        customer_idx: int32 [n], distinct customers (any order).
        week: The predicted week (its name goes to the fold column).
        first_query_id: query_id of the customer with the smallest customer_idx.

    Returns:
        One row per customer, sorted by customer_idx (= by query_id).
        Columns: query_id (int32), fold (str), customer_idx (int32), anchor (date).

    Raises:
        ValueError: If a customer repeats or there is no customer.
    """
    customers = np.sort(customer_idx.astype(np.int32))
    if customers.size == 0:
        raise ValueError("a week needs at least one customer to predict for")
    if bool(np.any(np.diff(customers) == 0)):
        raise ValueError("a customer appears twice; a week has one query per customer")
    query_ids = first_query_id + np.arange(customers.size, dtype=np.int64)
    queries = pl.DataFrame(
        {QUERY_ID: query_ids, CUSTOMER_IDX: customers},
        schema={QUERY_ID: pl.Int64, CUSTOMER_IDX: pl.Int32},
    ).with_columns(
        pl.col(QUERY_ID).cast(QUERY_ID_DTYPE, strict=True),
        pl.lit(week.name).alias(FOLD),
        pl.lit(week.anchor, dtype=pl.Date).alias(ANCHOR),
    )
    return queries.select(QUERIES_SCHEMA.names())


def read_catalog(paths: ProjectPaths, context: TickContext) -> pl.DataFrame:
    """The article catalog C: one row per article (article_idx)."""
    articles = scan_processed_parquet(
        paths.articles_file, paths.processed_metadata_file, context.preprocess
    )
    return build_catalog(articles)


def eval_queries(context: TickContext) -> pl.DataFrame:
    """`eval` mode: the customers who buy a catalog article in the tick's week.

    Built by hmrec.data.queries.build_fold_queries, the function of stage 1, from
    the SOURCE transactions: the week is on and after the anchor, so the tick's
    ingested table does not hold it. Only the query list is kept; the label sets
    the function returns with it are dropped here. They are built again when the
    week is closed (closed_week_answers).

    Raises:
        ValueError: If the week is not a fold of the manifest: the data hold no
            purchases of a later week, so `eval` mode has nothing to compare with.
    """
    if not context.tick.in_manifest:
        raise ValueError(
            f"`eval` mode needs a week of the fold manifest; the data hold no purchases of "
            f"{context.tick.name} ({context.tick.anchor}). Use `serve` mode."
        )
    offset = load_queries_config(context.paths.queries_config_file).query_id_fold_offset
    built = build_fold_queries(
        scan_source_transactions(context),
        read_catalog(context.paths, context),
        context.tick.fold,
        context.manifest,
        context.preprocess.day_zero,
        offset,
    )
    return built.queries


def serve_queries(context: TickContext) -> pl.DataFrame:
    """`serve` mode: one query per customer of the customer table. No transaction is read."""
    paths = context.paths
    customers = (
        scan_processed_parquet(
            paths.customers_file, paths.processed_metadata_file, context.preprocess
        )
        .select(CUSTOMER_IDX)
        .collect()
    )
    offset = load_queries_config(paths.queries_config_file).query_id_fold_offset
    if customers.height >= offset:
        raise ValueError(f"{customers.height} customers is not below query_id_fold_offset={offset}")
    base = query_id_base(context.tick, offset)
    return build_serve_queries(customers[CUSTOMER_IDX].to_numpy(), context.tick.fold, base)


def run_queries(context: TickContext) -> StepOutcome:
    """O1 (first part): write the week's query list (never replaced)."""
    queries = eval_queries(context) if context.mode == EVAL_MODE else serve_queries(context)
    target = context.view.queries_file(context.tick.name)
    if target.exists():
        raise FileExistsError(f"{target} already exists; a query list is never replaced")
    target.parent.mkdir(parents=True, exist_ok=True)
    queries.write_parquet(target)
    first, last = queries[QUERY_ID][0], queries[QUERY_ID][-1]
    detail = f"{queries.height} queries ({context.mode} mode), query ids {first}..{last}"
    LOGGER.info("O1 queries: %s -> %s", detail, target)
    metrics = [MetricRecord("output/queries", context.tick.anchor, float(queries.height))]
    return detail, metrics, [OutputFile(QUERIES_ROLE, target.name)]


# ---------------------------------------------------------------- the answers of a closed week


@dataclass(frozen=True)
class ClosedWeekAnswers:
    """What is known about a week once it is over.

    Attributes:
        queries: The week's queries as they were written when it was opened.
        labels: One row per (query_id, article_idx) bought in the week
            (query_id, fold, article_idx), sorted.
        purchase_values: One row per (customer_idx, article_idx) bought in the
            week with purchase_value (hmrec.data.queries.label_week_purchase_values).
    """

    queries: pl.DataFrame
    labels: pl.DataFrame
    purchase_values: pl.DataFrame


def closed_week_answers(context: TickContext) -> ClosedWeekAnswers:
    """Build the label sets of the week the tick closes, from the tick's ingested transactions.

    The week [closing anchor, tick anchor) lies before the tick's anchor, so the
    ingested table holds it. The queries the labels give must be exactly the
    query list written when the week was opened; a difference means the week was
    opened with other data or another query definition, and stops the tick.

    Raises:
        ValueError: If the closed week is not a fold of the manifest (a `serve`
            week: see hmrec.weekly.driver), or the query lists differ.
    """
    closing = context.closing
    if not closing.in_manifest:
        raise ValueError(f"week {closing.name} is not a fold of the manifest; it has no labels")
    transactions = scan_ingested_transactions(context)
    catalog = read_catalog(context.paths, context)
    day_zero = context.preprocess.day_zero
    offset = load_queries_config(context.paths.queries_config_file).query_id_fold_offset
    built: FoldQueries = build_fold_queries(
        transactions, catalog, closing.fold, context.manifest, day_zero, offset
    )
    check_fold_queries(built)
    opened = pl.read_parquet(context.view.queries_file(closing.name))
    if not opened.equals(built.queries):
        raise ValueError(
            f"week {closing.name}: the query list written when the week was opened "
            f"({opened.height} queries) is not the one its purchases give "
            f"({built.queries.height} queries)"
        )
    values = label_week_purchase_values(transactions, catalog, closing.fold, day_zero)
    return ClosedWeekAnswers(opened, built.labels, values)
