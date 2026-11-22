"""Tests for weekly.shard_recall: fit once, generate shard by shard; shards never change the output.

The main property: however the customers are cut into shards, the candidates of
a query are exactly the ones the locked channel gives when it is called ONCE with
all queries (the M6 way). Every channel is checked with several shard sizes on
the examples of the channel tests, with the real lock files (R5 with a smaller
model, R6 with the tiny model of test_twotower_chain).

Ported from branch kaggle-late-submission (tests/unit/test_kaggle_recall.py).
"""

from datetime import timedelta
from pathlib import Path

import polars as pl
import pytest
from test_generate import _small_wals_lock
from test_r1_popularity import ACTIVE_POOL, CUSTOMERS
from test_r1_popularity import ANCHOR_DAY as R1_ANCHOR_DAY
from test_r1_popularity import QUERIES as R1_QUERIES
from test_r1_popularity import ROWS as R1_ROWS
from test_r3_sibling import ANCHOR_DAY as R3_ANCHOR_DAY
from test_r3_sibling import PRODUCT_CODES
from test_r3_sibling import ROWS as R3_ROWS
from test_r4_itemcf import ANCHOR_DAY as R4_ANCHOR_DAY
from test_r4_itemcf import RANDOM_ACTIVE as R4_ACTIVE
from test_r4_itemcf import RANDOM_CATALOG, RANDOM_ROWS
from test_r4_itemcf import RANDOM_QUERIES as R4_QUERIES
from test_r5_wals import ANCHOR_DAY as R5_ANCHOR_DAY
from test_r5_wals import RANDOM_ACTIVE as R5_ACTIVE
from test_r5_wals import RANDOM_QUERIES as R5_QUERIES
from test_r5_wals import RANDOM_ROWS as R5_ROWS
from test_twotower_chain import ALARM, LINES, MEMORY
from test_twotower_chain import CONFIG as CHAIN_CONFIG
from test_twotower_infer import CHUNKS, _resources

from conftest import FIXTURE_DAY_ZERO, make_queries, make_recall_inputs
from hmrec.config import load_base_config, load_preprocess_config
from hmrec.data.day_index import day_index_to_date
from hmrec.data.folds import Fold
from hmrec.paths import StateLayout, build_paths, get_paths, pipeline_paths
from hmrec.recall.base import RecallInputs, add_age_days
from hmrec.recall.locked import LockedRecall, build_locked_channel, load_locked_recall
from hmrec.recall.r1_popularity import PopularityChannel
from hmrec.recall.twotower.chain import TwoTowerChainChannel
from hmrec.recall.twotower.infer import TwoTowerChannel
from hmrec.vectors import load_vectors
from hmrec.weekly.shard_recall import (
    RecallJob,
    generate_channel,
    popularity_lists,
    query_age_buckets,
)
from hmrec.weekly.shards import number_of_shards

REPO = Path(__file__).resolve().parents[2]
BASE = load_base_config(REPO / "configs" / "base.yaml")
PREPROCESS = load_preprocess_config(REPO / "configs" / "preprocess.yaml")
LOCKED = load_locked_recall(get_paths())
WEEK = "W1"
R1, R2, R3, R4, R5, R6 = (
    "r1_popularity",
    "r2_repurchase",
    "r3_sibling",
    "r4_itemcf",
    "r5_wals",
    "r6_twotower",
)
KEYS = ["query_id", "rank"]
CHAIN_POOL = [0, 1, 3, 4]  # articles 2 and 5 are outside A(a)


def _job(
    tmp_path: Path,
    anchor_day: int,
    queries: pl.DataFrame,
    inputs: RecallInputs,
    shard_customers: int,
    locked: LockedRecall = LOCKED,
) -> RecallJob:
    """A recall job on tiny data whose files go under tmp_path, with the given shard size."""
    anchor = day_index_to_date(anchor_day, FIXTURE_DAY_ZERO)
    week = Fold(name=WEEK, phase="S", anchor=anchor, label_end=anchor + timedelta(days=6))
    project = build_paths(BASE, tmp_path)
    layout = StateLayout(project.pipeline_state_dir)
    view = pipeline_paths(project, layout, anchor, ((WEEK, anchor),), None)
    return RecallJob(
        view, BASE, PREPROCESS, locked, week, shard_customers, queries.sort("query_id"), inputs
    )


def _no_chain() -> TwoTowerChainChannel:
    raise AssertionError("only R6 loads the chain model")


def _written(job: RecallJob, channel: str) -> pl.DataFrame:
    """All shards' candidates of a channel, in (query_id, rank) order."""
    n_shards = number_of_shards(job.queries.height, job.shard_customers)
    files = [job.paths.for_shard(shard).candidates_file(WEEK, channel) for shard in range(n_shards)]
    return pl.concat([pl.read_parquet(file) for file in files]).sort(KEYS)


def _unsharded(job: RecallJob, channel: str) -> pl.DataFrame:
    """What the locked channel gives when called once with ALL queries (the M6 way)."""
    built = build_locked_channel(channel, job.locked, job.base, job.paths, PREPROCESS)
    return built.generate(job.queries, job.inputs).sort(KEYS)


# ---------------------------------------------------------------- R1


def _r1_inputs() -> RecallInputs:
    return make_recall_inputs(R1_ROWS, R1_ANCHOR_DAY, customers=CUSTOMERS, active_pool=ACTIVE_POOL)


@pytest.mark.parametrize("shard_customers", [1, 2, 3])
def test_r1_shards_give_the_unsharded_candidates_and_one_table(
    tmp_path: Path, shard_customers: int
) -> None:
    job = _job(tmp_path, R1_ANCHOR_DAY, R1_QUERIES, _r1_inputs(), shard_customers)
    result = generate_channel(job, R1, _no_chain)
    channel = build_locked_channel(R1, LOCKED, BASE, job.paths, PREPROCESS)
    assert isinstance(channel, PopularityChannel)
    expected, expected_table = channel.generate_with_bucket_scores(job.queries, job.inputs)
    assert _written(job, R1).equals(expected.sort(KEYS))
    # ONE S(g, i) table for the week, whatever the number of shards.
    table = pl.read_parquet(job.paths.r1_bucket_scores_file(WEEK))
    assert table.equals(expected_table)
    assert job.paths.for_shard(1).r1_bucket_scores_file(WEEK) == job.paths.r1_bucket_scores_file(
        WEEK
    )
    # 12 candidate rows; the 5 non-fallback ones were compared with the table (test_generate).
    assert (result.n_rows, result.checked_rows) == (12, 5)
    assert result.n_shards == -(-3 // shard_customers)


def test_r1_lists_do_not_depend_on_which_buckets_are_asked_for() -> None:
    inputs = _r1_inputs()
    channel = build_locked_channel(R1, LOCKED, BASE, get_paths(), PREPROCESS)
    assert isinstance(channel, PopularityChannel)
    lines = inputs.transactions.collect().filter(pl.col("t_dat_day") < R1_ANCHOR_DAY)
    history = add_age_days(lines, R1_ANCHOR_DAY)
    arguments = (channel.config, history, inputs.customers, inputs.active_pool)
    all_lists, table = popularity_lists(*arguments, ["AGE_18_24", "AGE_25_34", "AGE_UNKNOWN"])
    one_list, same_table = popularity_lists(*arguments, ["AGE_25_34"])
    # A bucket's list is the same whether it is built alone or with the others.
    is_bucket = pl.col("age_bucket") == "AGE_25_34"
    only = all_lists.filter(is_bucket).sort("article_idx")
    assert only.equals(one_list.filter(is_bucket).sort("article_idx"))
    assert table.equals(same_table)


def test_r1_refuses_a_query_customer_without_an_age_bucket() -> None:
    queries = make_queries([(7, 999)])
    with pytest.raises(ValueError, match="no age_bucket"):
        query_age_buckets(queries, _r1_inputs().customers)


# ---------------------------------------------------------------- R2 - R5


def _r3_job(tmp_path: Path, shard_customers: int) -> RecallJob:
    inputs = make_recall_inputs(R3_ROWS, R3_ANCHOR_DAY, product_codes=PRODUCT_CODES)
    customers = sorted({row[0] for row in R3_ROWS})
    queries = make_queries([(900 + customer, customer) for customer in customers])
    return _job(tmp_path, R3_ANCHOR_DAY, queries, inputs, shard_customers)


def _r4_job(tmp_path: Path, shard_customers: int) -> RecallJob:
    inputs = make_recall_inputs(
        RANDOM_ROWS, R4_ANCHOR_DAY, product_codes=RANDOM_CATALOG, active_pool=R4_ACTIVE
    )
    return _job(tmp_path, R4_ANCHOR_DAY, R4_QUERIES, inputs, shard_customers)


def _r5_job(tmp_path: Path, shard_customers: int) -> RecallJob:
    inputs = make_recall_inputs(R5_ROWS, R5_ANCHOR_DAY, active_pool=R5_ACTIVE)
    return _job(tmp_path, R5_ANCHOR_DAY, R5_QUERIES, inputs, shard_customers, _small_wals_lock())


@pytest.mark.parametrize("shard_customers", [1, 2, 1000])
def test_r2_and_r3_shards_give_the_unsharded_candidates(
    tmp_path: Path, shard_customers: int
) -> None:
    # R2 on R4's random purchases (every customer has some); R3 on its own example.
    r2_job = _r4_job(tmp_path / "r2", shard_customers)
    generate_channel(r2_job, R2, _no_chain)
    assert _written(r2_job, R2).equals(_unsharded(r2_job, R2))
    assert _written(r2_job, R2).height > 0
    r3_job = _r3_job(tmp_path / "r3", shard_customers)
    generate_channel(r3_job, R3, _no_chain)
    assert _written(r3_job, R3).equals(_unsharded(r3_job, R3))


@pytest.mark.parametrize("shard_customers", [1, 2, 5, 1000])
def test_r4_neighbours_fitted_once_give_the_unsharded_candidates(
    tmp_path: Path, shard_customers: int
) -> None:
    job = _r4_job(tmp_path, shard_customers)
    result = generate_channel(job, R4, _no_chain)
    written = _written(job, R4)
    # Bit for bit the same scores as one call with all queries.
    assert written.equals(_unsharded(job, R4))
    assert written.height > 0 and result.n_rows == written.height


@pytest.mark.parametrize("shard_customers", [1, 4, 1000])
def test_r5_one_fit_gives_the_unsharded_candidates_and_per_shard_vectors(
    tmp_path: Path, shard_customers: int
) -> None:
    job = _r5_job(tmp_path, shard_customers)
    result = generate_channel(job, R5, _no_chain)
    written = _written(job, R5)
    assert written.equals(_unsharded(job, R5))
    # Every score was compared with the dot product of the vectors saved for its shard.
    assert result.checked_rows == written.height > 0
    first_shard = job.queries.slice(0, shard_customers)["customer_idx"].to_list()
    saved = load_vectors(job.paths.for_shard(0).wals_vectors_dir(WEEK), "wals", WEEK)
    # Users: only the shard's customers that have a vector; articles: all with a vector.
    assert set(saved.user_customer_idx.tolist()) <= set(first_shard)
    assert set(saved.item_article_idx.tolist()) == {article for _, article, _ in R5_ROWS}


def test_candidates_are_never_replaced(tmp_path: Path) -> None:
    job = _r4_job(tmp_path, 5)
    generate_channel(job, R4, _no_chain)
    with pytest.raises(FileExistsError):
        generate_channel(job, R4, _no_chain)


def test_recall_ignores_purchases_on_and_after_the_anchor(tmp_path: Path) -> None:
    clean = _r4_job(tmp_path / "clean", 5)
    poison = [(customer, 1, R4_ANCHOR_DAY + offset) for customer in range(40) for offset in (0, 3)]
    inputs = make_recall_inputs(
        RANDOM_ROWS + poison, R4_ANCHOR_DAY, product_codes=RANDOM_CATALOG, active_pool=R4_ACTIVE
    )
    poisoned = _job(tmp_path / "poison", R4_ANCHOR_DAY, R4_QUERIES, inputs, 5)
    for channel in (R2, R4):
        generate_channel(clean, channel, _no_chain)
        generate_channel(poisoned, channel, _no_chain)
        assert _written(poisoned, channel).equals(_written(clean, channel))


def test_unknown_channel_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown channel"):
        generate_channel(_r4_job(tmp_path, 5), "r7_new", _no_chain)


# ---------------------------------------------------------------- R6


def _tiny_chain() -> TwoTowerChainChannel:
    """The tiny chain channel with a freshly built tiny model (see test_twotower_chain)."""
    return TwoTowerChainChannel(
        CHAIN_CONFIG, _resources(), CHUNKS, lambda fold: Path(f"{fold}.pt"), ALARM, MEMORY, {}
    )


@pytest.mark.parametrize("shard_customers", [1, 2, 3])
def test_r6_shards_give_the_unsharded_candidates(tmp_path: Path, shard_customers: int) -> None:
    inputs = make_recall_inputs(LINES, 47, active_pool=CHAIN_POOL)
    queries = make_queries([(1, 0), (2, 1), (3, 2)], fold=WEEK)
    job = _job(tmp_path, 47, queries, inputs, shard_customers)
    loaded = []

    def load_chain() -> TwoTowerChainChannel:
        loaded.append(True)
        return _tiny_chain()

    result = generate_channel(job, R6, load_chain)
    written = _written(job, R6)
    frozen = TwoTowerChannel(CHAIN_CONFIG, _resources(), CHUNKS)
    assert written.equals(frozen.generate(job.queries, job.inputs).sort(KEYS))
    assert result.checked_rows == written.height == 6
    assert loaded == [True]  # the model is loaded once, not per shard
    # A shard's vectors: its own customers, and EVERY catalog article (for tt_dot).
    saved = load_vectors(job.paths.for_shard(0).twotower_vectors_dir(WEEK), "twotower", WEEK)
    assert saved.user_customer_idx.tolist() == [0, 1, 2][:shard_customers]
    assert saved.item_article_idx.tolist() == [0, 1, 2, 3, 4, 5]
