"""Tests for weekly.outputs, weekly.shards and the pipeline's paths and config.

Top-K check (K = 2 here): every query needs exactly K distinct valid articles at
ranks 1..K. Output signals on a hand example:

    purchases before the anchor (day 100), last 7 days = days 93..99:
        article 1: 3 lines, article 2: 2 lines, article 3: 2 lines, article 4: 1 line
        article 9: 5 lines on day 92 (before the window) and 4 on day 100 (the anchor)
    the 2 most popular = articles 1 and 2 (2 beats 3 on the smaller article_idx)
    Top-2 of 3 queries: (1, 2) (1, 3) (4, 2) -> 4 of 6 slots are popular articles;
    4 distinct articles; 12 candidates / 3 queries = 4 per query.
"""

import dataclasses
from datetime import date
from pathlib import Path

import polars as pl
import pytest
from weekly_example import V1, V2, build_workspace, tick_context

from conftest import FIXTURE_DAY_ZERO, make_transactions
from hmrec.config import load_pipeline_config, load_yaml, parse_pipeline_config
from hmrec.data.day_index import day_index_to_date
from hmrec.paths import StateLayout, get_paths, pipeline_paths, shard_name
from hmrec.weekly.outputs import (
    check_top_k,
    output_signals,
    popular_articles,
    prediction_table,
)
from hmrec.weekly.shards import number_of_shards, shard_queries

REPO = Path(__file__).resolve().parents[2]
QUERIES = pl.DataFrame(
    {"query_id": [10, 11, 12], "customer_idx": [5, 6, 7]},
    schema={"query_id": pl.Int32, "customer_idx": pl.Int32},
)


def _top(rows: list[tuple[int, int, int]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows, schema={"query_id": pl.Int32, "rank": pl.Int32, "article_idx": pl.Int32}, orient="row"
    )


GOOD_TOP = _top([(10, 1, 1), (10, 2, 2), (11, 1, 1), (11, 2, 3), (12, 1, 4), (12, 2, 2)])


# ---------------------------------------------------------------- the Top-K check


def test_a_complete_top_k_passes() -> None:
    check_top_k(GOOD_TOP, QUERIES, top_k=2, n_articles=5)


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([(10, 1, 1), (10, 2, 2), (11, 1, 1), (11, 2, 3)], "1 of 3 queries have no Top-2"),
        ([*GOOD_TOP.rows()[:4], (12, 1, 4)], "do not have exactly 2 distinct articles"),
        ([*GOOD_TOP.rows()[:4], (12, 1, 4), (12, 2, 4)], "do not have exactly 2 distinct"),
        ([*GOOD_TOP.rows()[:4], (12, 2, 4), (12, 3, 2)], "do not have exactly 2 distinct"),
        ([*GOOD_TOP.rows()[:4], (12, 1, 4), (12, 2, 5)], "outside 0..4"),
        ([*GOOD_TOP.rows(), (99, 1, 1), (99, 2, 2)], "not a query of the week"),
    ],
)
def test_an_incomplete_or_invalid_top_k_is_refused(
    rows: list[tuple[int, int, int]], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        check_top_k(_top(rows), QUERIES, top_k=2, n_articles=5)


# ---------------------------------------------------------------- output signals


def test_popular_articles_use_only_the_days_before_the_anchor() -> None:
    lines = [(0, 1, 99), (1, 1, 98), (2, 1, 93), (0, 2, 99), (1, 2, 97), (0, 3, 96), (1, 3, 95)]
    lines += [(0, 4, 94)] + [(c, 9, 92) for c in range(5)] + [(c, 9, 100) for c in range(4)]
    transactions = make_transactions(lines).lazy()
    anchor = day_index_to_date(100, FIXTURE_DAY_ZERO)
    popular = popular_articles(transactions, anchor, FIXTURE_DAY_ZERO, window_days=7, n_articles=2)
    # Article 9 has the most lines of all, but none inside [93, 100).
    assert popular["article_idx"].to_list() == [1, 2]
    signals = output_signals(GOOD_TOP, popular, n_queries=3, n_candidates=12, popular_window_days=7)
    assert signals.top_popular_share == pytest.approx(4 / 6)
    assert (signals.distinct_articles, signals.candidates_per_query) == (4, 4.0)
    assert (signals.n_queries, signals.popular_articles) == (3, 2)


def test_prediction_set_is_keyed_by_customer() -> None:
    table = prediction_table(GOOD_TOP, QUERIES)
    assert table.columns == ["customer_idx", "rank", "article_idx"]
    assert table.rows()[:2] == [(5, 1, 1), (5, 2, 2)] and table["customer_idx"].is_sorted()
    assert set(table.schema.dtypes()) == {pl.Int32}
    with pytest.raises(ValueError, match="not a query of the week"):
        prediction_table(_top([(99, 1, 1)]), QUERIES)


# ---------------------------------------------------------------- shards


def test_shards_partition_the_queries_in_order() -> None:
    assert [number_of_shards(n, 2) for n in (1, 2, 3, 5)] == [1, 1, 2, 3]
    parts = [shard_queries(QUERIES, shard, 2) for shard in range(2)]
    assert [part["query_id"].to_list() for part in parts] == [[10, 11], [12]]
    with pytest.raises(ValueError, match="shard must be in 0..1"):
        shard_queries(QUERIES, 2, 2)
    with pytest.raises(ValueError, match="shard_customers must be > 0"):
        number_of_shards(3, 0)
    assert shard_name(7) == "007"


# ---------------------------------------------------------------- paths


def test_a_ticks_paths_lie_in_the_state_and_inputs_in_the_bootstrap_state(tmp_path: Path) -> None:
    paths = build_workspace(tmp_path)
    view = tick_context(paths, V2).view
    state = paths.pipeline_state_dir
    # The transactions a step reads are the ones THIS tick ingested.
    assert (
        view.transactions_file == state / "weeks" / "2020-08-05" / "ingest" / "transactions.parquet"
    )
    assert view.candidates_file("V2", "r4_itemcf") == (
        state / "weeks" / "2020-08-05" / "recall_r4_itemcf" / "candidates.parquet"
    )
    assert view.features_dir("V1") == state / "weeks" / "2020-07-29" / "features"
    assert view.labels_file("V1").parent.name == "evaluation"
    # A fold before the first tick keeps its place in the bootstrap state.
    bootstrap = paths.bootstrap_dir
    assert view.ranker_sample_file("full", "F3", 40) == (
        bootstrap / "ranker" / "samples" / "full" / "F3" / "r40.parquet"
    )
    assert view.twotower_locked_chain_checkpoint_file("F4") == (
        bootstrap / "models" / "twotower" / "chain_locked" / "F4.pt"
    )
    assert (
        view.twotower_checkpoint_file(4)
        == bootstrap / "models" / "twotower" / "base" / "epoch_4.pt"
    )
    # A pipeline week's sample and chain model are in the state.
    assert view.ranker_sample_file("full", "V1", 40) == (
        state / "weeks" / "2020-07-29" / "sample" / "r40.parquet"
    )
    assert view.twotower_locked_chain_checkpoint_file("V1").name == "V1.pt"
    # Nothing a tick writes goes to a stage-1 location.
    assert view.artifacts_root != paths.artifacts_root and view.results_root != paths.results_root
    assert view.category_codes_file == paths.category_codes_file
    with pytest.raises(ValueError, match="not a week of the pipeline"):
        view.queries_file("V3")  # the tick at V2 knows no later week


def test_staging_and_shards_change_only_the_folders_they_name(tmp_path: Path) -> None:
    paths = build_workspace(tmp_path)
    layout = StateLayout(paths.pipeline_state_dir)
    weeks = (("V1", V1), ("V2", V2))
    view = pipeline_paths(paths, layout, V2, weeks, staging=(V2, "features"))
    week = layout.week_dir(V2)
    assert view.features_dir("V2") == week / ".staging" / "features"
    assert view.candidates_file("V2", "r5_wals").parent == week / "recall_r5_wals"
    assert view.features_dir("V1") == layout.week_dir(V1) / "features"
    shard = dataclasses.replace(view, staging=(V2, "shards/003/top")).for_shard(3)
    assert shard.top_file("V2") == week / ".staging" / "shards" / "003" / "top" / "top.parquet"
    assert shard.features_dir("V2") == week / "shards" / "003" / "features"
    assert shard.wals_vectors_dir("V2") == week / "recall_r5_wals" / "shards" / "003" / "vectors"
    # R1's bucket score table is one per week, also for a shard.
    assert shard.r1_bucket_scores_file("V2") == (
        week / "recall_r1_popularity" / "r1_bucket_scores.parquet"
    )


def test_real_paths_and_config() -> None:
    paths = get_paths()
    assert paths.pipeline_state_dir == paths.artifacts_root / "pipeline"
    assert (
        paths.replay_table_file("monitoring")
        == paths.results_root / "m12_replay" / "monitoring.csv"
    )
    config = load_pipeline_config(paths.pipeline_config_file)
    assert (config.shard_customers, config.metrics[0]) == (70_000, "map@12")
    assert (config.popular_articles, config.popular_window_days) == (100, 7)
    assert config.replay_eval_folds == ("V1", "V2", "V3", "V4", "T1", "T2", "T3", "T4")
    assert config.replay_serve_anchor == date(2020, 9, 23)
    raw = load_yaml(paths.pipeline_config_file)
    with pytest.raises(ValueError, match="shard_customers"):
        parse_pipeline_config({**raw, "serve": {"shard_customers": 0}})
