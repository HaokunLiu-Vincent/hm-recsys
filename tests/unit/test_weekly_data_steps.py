"""Tests for weekly.data_steps, eval.hit_value and eval.pipeline_labels on tiny data.

Calendar: the real fold manifest and day zero (2018-09-20). Day indexes:
    V1 = [678, 685)   anchor 2020-07-29        V2 = [685, 692)   anchor 2020-08-05

Purchase lines (customer, article, day), price 0.5 each; the catalog is articles 0..5:

    before V1                  (0, 1, 640) (1, 2, 650) (2, 3, 671) (0, 1, 677)
    week before V1 [671, 678)  holds (2, 3, 671) and (0, 1, 677): its first and last day
    V1 week [678, 685)         (0, 1, 678) (0, 1, 678) (0, 2, 680) (1, 3, 684)
                               -> buyers 0 and 1; customer 0 bought article 1 TWICE
    V2 week [685, 692)         (2, 4, 685) (3, 5, 691)

The tick at V2 (anchor day 685) may know every line before day 685 and none of
the V2 week.
"""

from datetime import date
from pathlib import Path

import polars as pl
import pytest
from weekly_example import V1, V2, build_workspace, tick_context, write_processed

from hmrec.data.folds import Fold
from hmrec.eval.hit_value import hit_value
from hmrec.eval.pipeline_labels import load_closed_week_evaluator, require_week_is_over
from hmrec.paths import ProjectPaths
from hmrec.run_manifest import EVAL_MODE, SERVE_MODE
from hmrec.weekly.data_steps import (
    build_serve_queries,
    closed_week_answers,
    describe_ingested,
    ingest_table,
    read_ingest_record,
    run_ingest,
    run_queries,
    scan_ingested_transactions,
    scan_source_transactions,
)

BEFORE_V1 = [(0, 1, 640), (1, 2, 650), (2, 3, 671), (0, 1, 677)]
V1_WEEK = [(0, 1, 678), (0, 1, 678), (0, 2, 680), (1, 3, 684)]
V2_WEEK = [(2, 4, 685), (3, 5, 691)]
LINES = [*BEFORE_V1, *V1_WEEK, *V2_WEEK]
N_CUSTOMERS, N_ARTICLES = 5, 6
DAY_ZERO = date(2018, 9, 20)


@pytest.fixture
def paths(tmp_path: Path) -> ProjectPaths:
    workspace = build_workspace(tmp_path)
    write_processed(workspace, LINES, N_CUSTOMERS, N_ARTICLES)
    return workspace


def _days(table: pl.DataFrame) -> list[int]:
    return table["t_dat_day"].to_list()


# ---------------------------------------------------------------- C1: the clock


def test_ingest_keeps_every_row_before_the_anchor_and_none_later(paths: ProjectPaths) -> None:
    context = tick_context(paths, V2)
    ingested = ingest_table(scan_source_transactions(context), V2, DAY_ZERO)
    # Days 640 .. 684 are before the anchor day 685; the V2 week (685, 691) is not there.
    assert _days(ingested) == [640, 650, 671, 677, 678, 678, 680, 684]
    assert ingested.columns == [
        "customer_idx",
        "article_idx",
        "t_dat_day",
        "price",
        "sales_channel_id",
    ]
    # Day 0 (the first day of the data) is inside the window.
    assert _days(ingest_table(scan_source_transactions(context), date(2018, 9, 21), DAY_ZERO)) == []


def test_rows_on_and_after_the_anchor_cannot_change_what_a_tick_reads(tmp_path: Path) -> None:
    # The time cut-off test: poison on the anchor day and after it changes nothing.
    clean, poisoned = tmp_path / "clean", tmp_path / "poisoned"
    poison = [(4, 0, 685), (4, 5, 686), (0, 1, 700)]
    tables = []
    for root, lines in ((clean, [*BEFORE_V1, *V1_WEEK]), (poisoned, [*LINES, *poison])):
        root.mkdir()
        workspace = build_workspace(root)
        write_processed(workspace, lines, N_CUSTOMERS, N_ARTICLES)
        context = tick_context(workspace, V2)
        run_ingest(context)
        # Every later step reads THIS file (context.view), never the source.
        tables.append(scan_ingested_transactions(context).collect())
    assert tables[0].equals(tables[1])
    assert max(_days(tables[1])) == 684


def test_ingest_step_writes_the_table_and_counts_the_weeks(paths: ProjectPaths) -> None:
    context = tick_context(paths, V2)
    detail, metrics, files = run_ingest(context)
    assert context.view.transactions_file == (
        paths.pipeline_state_dir / "weeks" / "2020-08-05" / "ingest" / "transactions.parquet"
    )
    record = read_ingest_record(context.view.ingest_record_file)
    assert (record.n_rows, record.first_date, record.last_date) == (8, "2020-06-21", "2020-08-04")
    # New week = V1 week [678, 685): 4 lines, from its first day to its last.
    assert (record.new_week_rows, record.new_week_first_date, record.new_week_last_date) == (
        4,
        "2020-07-29",
        "2020-08-04",
    )
    assert (record.expected_first_date, record.expected_last_date) == ("2020-07-29", "2020-08-04")
    # The 4 weeks before it, oldest first: [650, 657) 1 line, [657, 664) 0, [664, 671) 0,
    # [671, 678) 2 lines.
    assert record.previous_week_rows == (1, 0, 0, 2)
    assert "8 purchase lines before 2020-08-05" in detail
    assert [file.name for file in files] == ["transactions.parquet"]
    assert {metric.name: metric.value for metric in metrics}[
        "input/transactions_rows_new_week"
    ] == 4.0


def test_a_row_on_the_anchor_day_breaks_the_clock() -> None:
    table = pl.DataFrame({"t_dat_day": [684, 685]}, schema={"t_dat_day": pl.Int16})
    with pytest.raises(ValueError, match="the clock is broken"):
        describe_ingested(table, V2, DAY_ZERO, 4)
    with pytest.raises(ValueError, match="no purchase line before the anchor"):
        describe_ingested(table.head(0), V2, DAY_ZERO, 4)


def test_a_week_with_a_missing_last_day_is_visible_in_the_record() -> None:
    table = pl.DataFrame({"t_dat_day": [678, 680]}, schema={"t_dat_day": pl.Int16})
    record = describe_ingested(table, V2, DAY_ZERO, 4)
    # The newest row is 2020-07-31, not the expected 2020-08-04: the driver's gate alerts.
    assert (record.new_week_last_date, record.expected_last_date) == ("2020-07-31", "2020-08-04")


# ---------------------------------------------------------------- the query list


def test_eval_queries_are_the_buyers_of_the_week_with_stage_1_ids(paths: ProjectPaths) -> None:
    context = tick_context(paths, V1)
    detail, _, files = run_queries(context)
    queries = pl.read_parquet(context.view.queries_file("V1"))
    # V1 is fold number 8: query ids start at 80,000,000, in customer order.
    assert queries.rows() == [(80_000_000, "V1", 0, V1), (80_000_001, "V1", 1, V1)]
    assert "2 queries (eval mode)" in detail and files[0].name == "queries.parquet"
    with pytest.raises(FileExistsError, match="never replaced"):
        run_queries(context)


def test_serve_queries_are_every_customer_and_read_no_transaction(paths: ProjectPaths) -> None:
    paths.transactions_file.unlink()  # serve mode must not need it
    context = tick_context(paths, V2, SERVE_MODE)
    run_queries(context)
    queries = pl.read_parquet(context.view.queries_file("V2"))
    assert queries["customer_idx"].to_list() == [0, 1, 2, 3, 4]
    assert queries["query_id"].to_list() == [90_000_000 + position for position in range(5)]


def test_eval_mode_needs_a_week_of_the_manifest(paths: ProjectPaths) -> None:
    after_the_data = tick_context(paths, date(2020, 9, 23), EVAL_MODE)
    with pytest.raises(ValueError, match="Use `serve` mode"):
        run_queries(after_the_data)


def test_serve_queries_are_checked() -> None:
    import numpy as np

    week = Fold("W20200923", "S", date(2020, 9, 23), date(2020, 9, 29))
    queries = build_serve_queries(np.array([7, 3, 5], dtype=np.int32), week, 160_000_000)
    assert queries["customer_idx"].to_list() == [3, 5, 7]
    assert queries["query_id"].to_list() == [160_000_000, 160_000_001, 160_000_002]
    assert queries["fold"].unique().to_list() == ["W20200923"]
    with pytest.raises(ValueError, match="appears twice"):
        build_serve_queries(np.array([1, 1], dtype=np.int32), week, 0)
    with pytest.raises(ValueError, match="at least one customer"):
        build_serve_queries(np.array([], dtype=np.int32), week, 0)


# ---------------------------------------------------------------- closing a week


def test_closed_week_labels_come_from_the_ticks_own_transactions(paths: ProjectPaths) -> None:
    run_queries(tick_context(paths, V1))  # the V1 tick opened the week
    context = tick_context(paths, V2)
    run_ingest(context)
    answers = closed_week_answers(context)
    assert answers.queries["customer_idx"].to_list() == [0, 1]
    # G_q is a set: customer 0's two lines of article 1 are one label.
    assert answers.labels.select("query_id", "article_idx").rows() == [
        (80_000_000, 1),
        (80_000_000, 2),
        (80_000_001, 3),
    ]
    # The purchase values keep both lines: 2 x 0.5 for (customer 0, article 1).
    assert answers.purchase_values.rows() == [(0, 1, 2, 1.0), (0, 2, 1, 0.5), (1, 3, 1, 0.5)]


def test_closing_stops_if_the_query_list_is_not_the_one_the_purchases_give(
    paths: ProjectPaths,
) -> None:
    opened = tick_context(paths, V1)
    run_queries(opened)
    file = opened.view.queries_file("V1")
    pl.read_parquet(file).head(1).write_parquet(file)  # another query list than at opening
    context = tick_context(paths, V2)
    run_ingest(context)
    with pytest.raises(ValueError, match="is not the one its purchases give"):
        closed_week_answers(context)


# ---------------------------------------------------------------- labels and the clock


def test_labels_of_a_week_are_read_only_after_the_week(paths: ProjectPaths) -> None:
    week = tick_context(paths, V1).tick.fold
    run_queries(tick_context(paths, V1))
    context = tick_context(paths, V2)
    run_ingest(context)
    labels_file = context.view.labels_file("V1")
    labels_file.parent.mkdir(parents=True)
    closed_week_answers(context).labels.write_parquet(labels_file)
    assert load_closed_week_evaluator(week, V2, labels_file).n_queries == 2
    # The tick AT the week's anchor, or inside the week, may not read them.
    for too_early in (V1, date(2020, 8, 4)):
        with pytest.raises(ValueError, match="the week is not over"):
            load_closed_week_evaluator(week, too_early, labels_file)
    require_week_is_over(week, date(2020, 8, 12))


# ---------------------------------------------------------------- hit value


def test_hit_value_sums_what_customers_paid_for_recommended_articles() -> None:
    queries = pl.DataFrame({"query_id": [10, 11], "customer_idx": [0, 1]})
    purchases = pl.DataFrame(
        {
            "customer_idx": [0, 0, 1, 1],
            "article_idx": [1, 2, 3, 4],
            "purchase_value": [1.0, 0.5, 0.25, 8.0],
        }
    )
    recommended = pl.DataFrame({"query_id": [10, 10, 11, 11], "article_idx": [1, 9, 3, 2]})
    # Hits: (customer 0, article 1) = 1.0 and (customer 1, article 3) = 0.25. Article 2 was
    # bought by customer 0 but recommended to customer 1; article 4 was not recommended.
    assert hit_value(recommended, queries, purchases) == 1.25
    assert hit_value(recommended.head(0), queries, purchases) == 0.0
    with pytest.raises(ValueError, match="repeats"):
        hit_value(pl.concat([recommended, recommended.head(1)]), queries, purchases)
    with pytest.raises(ValueError, match="not a query of the week"):
        hit_value(pl.DataFrame({"query_id": [99], "article_idx": [1]}), queries, purchases)
