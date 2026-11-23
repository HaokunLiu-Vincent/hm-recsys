"""Tests for weekly.serving_build: the serving form of a prediction set and its fallback list.

A week of 5 customers and 6 articles, top_k = 2 (checked by hand):

    customer_idx   customer_id (64 hex)   age bucket      query   R1 list (rank 1, 2, 3)
    0              ...0                   AGE_18_24       100     0, 1, 2
    1              ...1                   AGE_18_24       101     0, 1, 2
    2              ...2                   AGE_18_24       102     0, 1, 2
    3              ...3                   AGE_UNKNOWN     103     5, 2, 4
    4              ...4                   AGE_18_24       104     0, 1, 2

    prediction set (customer -> Top-2 article_idx): 0 -> (1, 0), 1 -> (2, 3), 2 -> (4, 5),
    3 -> (0, 5), 4 -> (3, 1)

The fallback list is the first 2 articles of the R1 list of the customer without
an age: articles 5 and 2, i.e. the ids 0000000005 and 0000000002.
"""

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from weekly_example import (
    STAMP,
    V1,
    build_workspace,
    tick_context,
    write_processed,
    write_r1_candidates,
)

from hmrec.locks import file_sha256
from hmrec.paths import ProjectPaths, StateLayout
from hmrec.run_manifest import SERVE_MODE
from hmrec.serving_set import SOURCE_FALLBACK, SOURCE_RANKER, load_serving_set
from hmrec.weekly.serving_build import (
    backfill_serving_set,
    bucket_list_of_queries,
    build_serving_set,
    serving_arrays,
    serving_output_files,
)

TOP = {0: (1, 0), 1: (2, 3), 2: (4, 5), 3: (0, 5), 4: (3, 1)}
R1_LISTS = {100: [0, 1, 2], 101: [0, 1, 2], 102: [0, 1, 2], 103: [5, 2, 4], 104: [0, 1, 2]}
FALLBACK = ("0000000005", "0000000002")
RANKER_SHA256 = "ab" * 32


def _customer_id(customer: int) -> str:
    return f"{customer:064x}"


def _predictions(top: dict[int, tuple[int, ...]] = TOP) -> pl.DataFrame:
    rows = [(c, rank, a) for c, articles in top.items() for rank, a in enumerate(articles, 1)]
    schema = {"customer_idx": pl.Int32, "rank": pl.Int32, "article_idx": pl.Int32}
    return pl.DataFrame(rows, schema=schema, orient="row")


def _queries() -> pl.DataFrame:
    return pl.DataFrame(
        {"query_id": [100, 101, 102, 103, 104], "customer_idx": [0, 1, 2, 3, 4]},
        schema={"query_id": pl.Int32, "customer_idx": pl.Int32},
    )


@pytest.fixture
def paths(tmp_path: Path) -> ProjectPaths:
    """A workspace with the week above: tables, queries, R1 candidates, a published set."""
    workspace = build_workspace(tmp_path)
    write_processed(
        workspace, [(0, 0, 670)], n_customers=5, n_articles=6, customers_without_age=(3,)
    )
    view = tick_context(workspace, V1).view
    queries_file = view.queries_file("V1")
    queries_file.parent.mkdir(parents=True)
    _queries().write_parquet(queries_file)
    write_r1_candidates(view.candidates_file("V1", "r1_popularity"), R1_LISTS)
    layout = StateLayout(workspace.pipeline_state_dir)
    layout.prediction_set_dir(V1).mkdir(parents=True)
    _predictions().write_parquet(layout.prediction_file(V1))
    manifest = {
        "mode": "eval",
        "predictions": {"sha256": file_sha256(layout.prediction_file(V1))},
        "models": [
            {
                "role": "ranker",
                "path": "weeks/2020-07-29/ranker_model/model.txt",
                "sha256": RANKER_SHA256,
            },
            {"role": "twotower_chain", "path": "models/F4.pt", "sha256": "cd" * 32},
        ],
    }
    layout.prediction_manifest_file(V1).write_text(json.dumps(manifest), encoding="utf-8")
    return workspace


# ---------------------------------------------------------------- the fallback list


def test_the_bucket_list_is_read_from_the_candidates_of_its_queries() -> None:
    schema = {"query_id": pl.Int32, "rank": pl.Int32, "article_idx": pl.Int32}
    rows = [(q, r, a) for q, articles in R1_LISTS.items() for r, a in enumerate(articles, 1)]
    candidates = pl.DataFrame(rows, schema=schema, orient="row").filter(pl.col("rank") <= 2)
    one = pl.DataFrame({"query_id": [103]}, schema={"query_id": pl.Int32})
    assert bucket_list_of_queries(candidates, one, top_k=2) == [5, 2]
    # Three queries of one bucket share one list.
    same = pl.DataFrame({"query_id": [100, 101, 104]}, schema={"query_id": pl.Int32})
    assert bucket_list_of_queries(candidates, same, top_k=2) == [0, 1]
    # Queries with different lists are not one bucket: refused, not averaged.
    mixed = pl.DataFrame({"query_id": [100, 103]}, schema={"query_id": pl.Int32})
    with pytest.raises(ValueError, match="do not share one list"):
        bucket_list_of_queries(candidates, mixed, top_k=2)
    # The list is shorter than top_k.
    with pytest.raises(ValueError, match=r"ranks 1\.\.3"):
        bucket_list_of_queries(candidates, one, top_k=3)
    # One query of the bucket lacks a rank.
    holed = candidates.filter(~((pl.col("query_id") == 101) & (pl.col("rank") == 2)))
    with pytest.raises(ValueError, match="do not share one list"):
        bucket_list_of_queries(holed, same, top_k=2)


# ---------------------------------------------------------------- the arrays


def test_arrays_are_sorted_by_customer_id_and_keep_each_customers_row() -> None:
    # The ids are given in the REVERSE order of customer_idx, so sorting by id must
    # reorder the rows: customer 4 has the smallest id.
    customer_map = pl.DataFrame(
        {"customer_idx": [0, 1, 2, 3, 4], "customer_id": [_customer_id(9 - c) for c in range(5)]},
        schema={"customer_idx": pl.Int32, "customer_id": pl.String},
    )
    article_map = pl.DataFrame(
        {
            "article_idx": [3, 0, 1, 2, 5, 4],
            "article_id": [f"{a:010d}" for a in (3, 0, 1, 2, 5, 4)],
        },
        schema={"article_idx": pl.Int32, "article_id": pl.String},
    )
    customer_ids, top_articles, article_ids = serving_arrays(
        _predictions(), customer_map, article_map, top_k=2
    )
    assert customer_ids.tolist() == [_customer_id(c).encode() for c in (5, 6, 7, 8, 9)]
    assert top_articles.dtype == np.int32
    assert top_articles.tolist() == [list(TOP[c]) for c in (4, 3, 2, 1, 0)]
    assert article_ids.tolist() == [f"{a:010d}".encode() for a in range(6)]


def test_a_prediction_table_that_is_not_top_k_rows_per_customer_is_refused() -> None:
    customer_map = pl.DataFrame(
        {"customer_idx": list(range(5)), "customer_id": [_customer_id(c) for c in range(5)]},
        schema={"customer_idx": pl.Int32, "customer_id": pl.String},
    )
    article_map = pl.DataFrame(
        {"article_idx": list(range(6)), "article_id": [f"{a:010d}" for a in range(6)]},
        schema={"article_idx": pl.Int32, "article_id": pl.String},
    )
    good = _predictions()
    with pytest.raises(ValueError, match="not 2 per customer"):
        serving_arrays(good.head(9), customer_map, article_map, top_k=2)
    swapped = good.with_columns(pl.when(pl.col("rank") == 1).then(2).otherwise(1).alias("rank"))
    with pytest.raises(ValueError, match=r"not ranks 1\.\.2"):
        serving_arrays(swapped.cast({"rank": pl.Int32}), customer_map, article_map, top_k=2)
    shifted = pl.concat([good.slice(1), good.head(1)]).with_columns(
        pl.Series("rank", [1, 2] * 5, dtype=pl.Int32)
    )
    with pytest.raises(ValueError, match="does not have 2 rows in a row"):
        serving_arrays(shifted, customer_map, article_map, top_k=2)
    with pytest.raises(ValueError, match="1 customer ids are missing"):
        serving_arrays(good, customer_map.head(4), article_map, top_k=2)
    with pytest.raises(ValueError, match="does not hold the indexes"):
        serving_arrays(good, customer_map, article_map.filter(pl.col("article_idx") != 2), top_k=2)
    accent = customer_map.with_columns(
        pl.when(pl.col("customer_idx") == 0)
        .then(pl.lit("é"))
        .otherwise("customer_id")
        .alias("customer_id")
    )
    with pytest.raises(ValueError, match="not ASCII"):
        serving_arrays(good, accent, article_map, top_k=2)


# ---------------------------------------------------------------- a whole serving set


def test_backfill_builds_the_serving_set_of_a_published_prediction_set(
    paths: ProjectPaths,
) -> None:
    layout = StateLayout(paths.pipeline_state_dir)
    before = sorted(path.name for path in layout.prediction_set_dir(V1).iterdir())
    record = backfill_serving_set(paths, V1, STAMP)
    assert (record.n_customers, record.top_k, record.n_articles) == (5, 2, 6)
    description = record.description
    assert (description.anchor, description.week, description.mode) == ("2020-07-29", "V1", "eval")
    assert description.built_by == "backfill" and description.git_commit == STAMP.git_commit
    assert description.prediction_sha256 == file_sha256(layout.prediction_file(V1))
    assert record.model_version == RANKER_SHA256[:12]
    # The fallback list: the first 2 articles of the R1 list of the customer without an age.
    fallback = description.fallback
    assert fallback.age_bucket == "AGE_UNKNOWN" and fallback.article_ids == FALLBACK
    r1_file = "weeks/2020-07-29/recall_r1_popularity/candidates.parquet"
    assert fallback.source == r1_file
    assert fallback.source_sha256 == file_sha256(layout.root / r1_file)
    # What the service will answer, in original ids.
    serving = load_serving_set(layout.serving_dir(V1))
    for customer, (first, second) in TOP.items():
        expected = (f"{first:010d}", f"{second:010d}")
        assert serving.recommend(_customer_id(customer)) == (expected, SOURCE_RANKER)
    assert serving.recommend(_customer_id(77)) == (FALLBACK, SOURCE_FALLBACK)
    # Nothing that existed was changed, and the set is never built twice.
    after = sorted(path.name for path in layout.prediction_set_dir(V1).iterdir())
    assert after == sorted([*before, "serving"])
    with pytest.raises(FileExistsError, match="never replaced"):
        backfill_serving_set(paths, V1, STAMP)


def test_backfill_refuses_a_missing_set_and_a_changed_prediction_file(paths: ProjectPaths) -> None:
    from weekly_example import V2

    with pytest.raises(FileNotFoundError, match="no prediction set for 2020-08-05"):
        backfill_serving_set(paths, V2, STAMP)
    layout = StateLayout(paths.pipeline_state_dir)
    _predictions({**TOP, 4: (3, 2)}).write_parquet(layout.prediction_file(V1))
    with pytest.raises(ValueError, match="is not the recorded"):
        backfill_serving_set(paths, V1, STAMP)
    assert not layout.serving_dir(V1).exists()


def test_a_week_without_a_customer_without_age_has_no_fallback_list(
    paths: ProjectPaths, tmp_path: Path
) -> None:
    # The same week, but every customer has an age: R1's output holds no list for
    # unknown customers, and the build stops instead of inventing one.
    write_processed(paths, [(0, 0, 670)], n_customers=5, n_articles=6)
    with pytest.raises(ValueError, match="no query of week V1 has a customer in age bucket"):
        backfill_serving_set(paths, V1, STAMP)
    assert not (StateLayout(paths.pipeline_state_dir).serving_dir(V1) / "serving.json").exists()


def test_in_serve_mode_the_list_comes_from_the_first_shard_with_such_a_customer(
    paths: ProjectPaths, tmp_path: Path
) -> None:
    # 5 customers in shards of 2: (0, 1), (2, 3), (4). The customer without an age is
    # in shard 1; shard 0 has none and its file is not even opened (it does not exist).
    context = tick_context(paths, V1, SERVE_MODE)
    view = context.view
    shard_lists = {query: R1_LISTS[query] for query in (102, 103)}
    write_r1_candidates(view.for_shard(1).candidates_file("V1", "r1_popularity"), shard_lists)
    folder = tmp_path / "serving"
    record = build_serving_set(
        context, folder, _predictions(), "12" * 32, RANKER_SHA256, "2020-07-29-serve-a1"
    )
    fallback = record.description.fallback
    assert fallback.article_ids == FALLBACK
    assert fallback.source == "weeks/2020-07-29/recall_r1_popularity/shards/001/candidates.parquet"
    assert record.description.mode == "serve"
    assert record.description.built_by == "2020-07-29-serve-a1"


def test_the_serving_files_are_outputs_of_the_publish_step() -> None:
    files = serving_output_files()
    assert [file.name for file in files] == [
        "serving/customer_ids.npy",
        "serving/top_articles.npy",
        "serving/article_ids.npy",
        "serving/serving.json",
    ]
    assert len({file.role for file in files}) == 4
