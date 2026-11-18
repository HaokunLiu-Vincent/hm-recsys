"""Tests for kaggle.recall: fit once, generate shard by shard; the output never depends on shards.

The main property (docs/decisions.md 2026-11-19): however the customers are cut
into shards, the candidates of a query are exactly the ones the locked channel
gives when it is called ONCE with all queries (the M6 way). Every channel is
checked with shard sizes 1, 2 and "everything in one shard" on the examples of
the channel tests, with the real lock files (R5 with a smaller model, R6 with
the tiny model and chain of test_twotower_chain).

R6 of the real run: the locked chain's last model is fine-tuned once more on the
last fold's week. Tiny version of that here (FIXTURE_DAY_ZERO):
    folds D4 [33, 40), F1 [40, 47), T4 [47, 54); the submission week starts day 54
    the "locked chain" ends with F1.pt (the model that predicts T4)
    -> fine-tune on the T4 week [47, 54): new rows (0,4,48) (1,5,49) (2,0,50) of the
       non-alarm users, 3 replay rows; saved as T4.pt in the run's own folder
    -> the week of day 54 is predicted with T4.pt.
"""

import dataclasses
from datetime import timedelta
from pathlib import Path

import numpy as np
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
from hmrec.config import load_base_config, load_kaggle_config, load_preprocess_config
from hmrec.data.day_index import day_index_to_date
from hmrec.data.folds import Fold, FoldManifest
from hmrec.kaggle import recall as recall_module
from hmrec.kaggle.context import KaggleContext
from hmrec.kaggle.plan import RUN_REHEARSAL, RUN_SUBMISSION
from hmrec.kaggle.recall import (
    RecallJob,
    generate_channel,
    popularity_lists,
    prepare_twotower_chain,
    query_age_buckets,
)
from hmrec.locks import check_file_sha256, file_sha256
from hmrec.paths import build_paths, get_paths
from hmrec.recall.base import RecallInputs, add_age_days
from hmrec.recall.chain_records import ChainRecord, append_chain_record, read_chain_records
from hmrec.recall.inputs import SharedTables
from hmrec.recall.locked import LockedRecall, ModelFile, build_locked_channel, load_locked_recall
from hmrec.recall.r1_popularity import PopularityChannel
from hmrec.recall.twotower.chain import TwoTowerChainChannel
from hmrec.recall.twotower.infer import TwoTowerChannel
from hmrec.recall.twotower.train import load_checkpoint, save_checkpoint
from hmrec.run_records import RunStamp
from hmrec.vectors import load_vectors

REPO = Path(__file__).resolve().parents[2]
BASE = load_base_config(REPO / "configs" / "base.yaml")
PREPROCESS = load_preprocess_config(REPO / "configs" / "preprocess.yaml")
KAGGLE = load_kaggle_config(REPO / "configs" / "kaggle_submission.yaml")
LOCKED = load_locked_recall(get_paths())
STAMP = RunStamp("abc123", False)
R1, R2, R3, R4, R5, R6 = (
    "r1_popularity",
    "r2_repurchase",
    "r3_sibling",
    "r4_itemcf",
    "r5_wals",
    "r6_twotower",
)
KEYS = ["query_id", "rank"]


def _week(name: str, anchor_day: int, phase: str = "S") -> Fold:
    anchor = day_index_to_date(anchor_day, FIXTURE_DAY_ZERO)
    return Fold(name=name, phase=phase, anchor=anchor, label_end=anchor + timedelta(days=6))


def _job(
    tmp_path: Path,
    anchor_day: int,
    queries: pl.DataFrame,
    inputs: RecallInputs,
    shard_customers: int,
    locked: LockedRecall = LOCKED,
    run: str = RUN_REHEARSAL,
    manifest: FoldManifest | None = None,
) -> RecallJob:
    """A recall job on tiny data whose files go under tmp_path, with the given shard size."""
    paths = build_paths(BASE, tmp_path)
    context = KaggleContext(
        run=run,
        paths=paths,
        base=BASE,
        config=dataclasses.replace(KAGGLE, shard_customers=shard_customers),
        manifest=manifest or FoldManifest(folds=(_week("T4", anchor_day - 7, "T"),)),
        preprocess=dataclasses.replace(PREPROCESS, day_zero=FIXTURE_DAY_ZERO),
        target=_week("S", anchor_day),
        stamp=STAMP,
    )
    tables = SharedTables(inputs.transactions, inputs.customers, inputs.articles)
    return RecallJob(context, locked, tables, queries.sort("query_id"), inputs)


def _written(job: RecallJob, channel: str) -> pl.DataFrame:
    """All shards' candidates of a channel, in (query_id, rank) order."""
    context = job.context
    n_shards = -(-job.queries.height // context.config.shard_customers)
    files = [
        context.paths.kaggle_candidates_file(context.run, shard, channel)
        for shard in range(n_shards)
    ]
    return pl.concat([pl.read_parquet(file) for file in files]).sort(KEYS)


def _unsharded(job: RecallJob, channel: str) -> pl.DataFrame:
    """What the locked channel gives when called once with ALL queries (the M6 way)."""
    context = job.context
    built = build_locked_channel(channel, job.locked, context.base, context.paths, PREPROCESS)
    return built.generate(job.queries, job.inputs).sort(KEYS)


# ---------------------------------------------------------------- R1


def _r1_inputs() -> RecallInputs:
    return make_recall_inputs(R1_ROWS, R1_ANCHOR_DAY, customers=CUSTOMERS, active_pool=ACTIVE_POOL)


@pytest.mark.parametrize("shard_customers", [1, 2, 3])
def test_r1_shards_give_the_unsharded_candidates_and_one_table(
    tmp_path: Path, shard_customers: int
) -> None:
    job = _job(tmp_path, R1_ANCHOR_DAY, R1_QUERIES, _r1_inputs(), shard_customers)
    result = generate_channel(job, R1, STAMP)
    channel = build_locked_channel(R1, LOCKED, BASE, job.context.paths, PREPROCESS)
    assert isinstance(channel, PopularityChannel)
    expected, expected_table = channel.generate_with_bucket_scores(job.queries, job.inputs)
    assert _written(job, R1).equals(expected.sort(KEYS))
    # ONE S(g, i) table for the run, whatever the number of shards.
    table = pl.read_parquet(job.context.paths.kaggle_r1_bucket_scores_file(RUN_REHEARSAL))
    assert table.equals(expected_table)
    # 12 candidate rows; the 5 non-fallback ones were compared with the table (test_generate).
    assert (result.n_rows, result.checked_rows) == (12, 5)
    assert (result.n_shards, result.n_generated) == (-(-3 // shard_customers), result.n_shards)


def test_r1_lists_do_not_depend_on_which_buckets_are_asked_for() -> None:
    inputs = _r1_inputs()
    config = build_locked_channel(R1, LOCKED, BASE, get_paths(), PREPROCESS)
    assert isinstance(config, PopularityChannel)
    lines = inputs.transactions.collect().filter(pl.col("t_dat_day") < R1_ANCHOR_DAY)
    history = add_age_days(lines, R1_ANCHOR_DAY)
    arguments = (config.config, history, inputs.customers, inputs.active_pool)
    all_lists, table = popularity_lists(*arguments, ["AGE_18_24", "AGE_25_34", "AGE_UNKNOWN"])
    one_list, same_table = popularity_lists(*arguments, ["AGE_25_34"])
    # A bucket's list is the same whether it is built alone or with the others.
    # (The own rows of the other buckets are in both results; only the fill-in is per request.)
    is_bucket = pl.col("age_bucket") == "AGE_25_34"
    only = all_lists.filter(is_bucket).sort("article_idx")
    assert only.equals(one_list.filter(is_bucket).sort("article_idx"))
    assert table.equals(same_table)


CUSTOMERS_TABLE = _r1_inputs().customers


def test_r1_refuses_a_query_customer_without_an_age_bucket() -> None:
    queries = make_queries([(7, 999)])
    with pytest.raises(ValueError, match="no age_bucket"):
        query_age_buckets(queries, CUSTOMERS_TABLE)


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
    generate_channel(r2_job, R2, STAMP)
    assert _written(r2_job, R2).equals(_unsharded(r2_job, R2))
    assert _written(r2_job, R2).height > 0
    r3_job = _r3_job(tmp_path / "r3", shard_customers)
    generate_channel(r3_job, R3, STAMP)
    assert _written(r3_job, R3).equals(_unsharded(r3_job, R3))


@pytest.mark.parametrize("shard_customers", [1, 2, 5, 1000])
def test_r4_neighbours_fitted_once_give_the_unsharded_candidates(
    tmp_path: Path, shard_customers: int
) -> None:
    job = _r4_job(tmp_path, shard_customers)
    result = generate_channel(job, R4, STAMP)
    written = _written(job, R4)
    # Bit for bit the same scores as one call with all queries.
    assert written.equals(_unsharded(job, R4))
    assert written.height > 0 and result.n_rows == written.height


@pytest.mark.parametrize("shard_customers", [1, 4, 1000])
def test_r5_one_fit_gives_the_unsharded_candidates_and_per_shard_vectors(
    tmp_path: Path, shard_customers: int
) -> None:
    job = _r5_job(tmp_path, shard_customers)
    result = generate_channel(job, R5, STAMP)
    written = _written(job, R5)
    assert written.equals(_unsharded(job, R5))
    # Every score was compared with the dot product of the vectors saved for its shard.
    assert result.checked_rows == written.height > 0
    paths, shard_size = job.context.paths, job.context.config.shard_customers
    first_shard = job.queries.slice(0, shard_size)["customer_idx"].to_list()
    saved = load_vectors(paths.kaggle_vectors_dir(RUN_REHEARSAL, 0, "wals"), "wals", "S")
    # Users: only the shard's customers that have a vector; articles: all with a vector.
    assert set(saved.user_customer_idx.tolist()) <= set(first_shard)
    assert set(saved.item_article_idx.tolist()) == {article for _, article, _ in R5_ROWS}


def test_finished_shards_are_skipped_and_never_replaced(tmp_path: Path) -> None:
    job = _r4_job(tmp_path, 5)
    first = generate_channel(job, R4, STAMP)
    files = sorted((job.context.paths.kaggle_run_dir(RUN_REHEARSAL) / "shards").rglob("*.parquet"))
    written_at = [file.stat().st_mtime_ns for file in files]
    again = generate_channel(job, R4, STAMP)
    assert (first.n_generated, again.n_generated) == (first.n_shards, 0)
    assert again.n_rows == first.n_rows
    assert [file.stat().st_mtime_ns for file in files] == written_at
    # One shard removed: only that one is written again, and it is the same file.
    removed = pl.read_parquet(files[1])
    files[1].unlink()
    assert generate_channel(job, R4, STAMP).n_generated == 1
    assert pl.read_parquet(files[1]).equals(removed)


def test_vectors_without_candidates_are_a_half_finished_shard(tmp_path: Path) -> None:
    job = _r5_job(tmp_path, 4)
    generate_channel(job, R5, STAMP)
    job.context.paths.kaggle_candidates_file(RUN_REHEARSAL, 1, R5).unlink()
    with pytest.raises(FileExistsError, match="half-finished"):
        generate_channel(job, R5, STAMP)


def test_recall_ignores_purchases_on_and_after_the_anchor(tmp_path: Path) -> None:
    clean = _r4_job(tmp_path / "clean", 5)
    poison = [(customer, 1, R4_ANCHOR_DAY + offset) for customer in range(40) for offset in (0, 3)]
    inputs = make_recall_inputs(
        RANDOM_ROWS + poison, R4_ANCHOR_DAY, product_codes=RANDOM_CATALOG, active_pool=R4_ACTIVE
    )
    poisoned = _job(tmp_path / "poison", R4_ANCHOR_DAY, R4_QUERIES, inputs, 5)
    for channel in (R2, R4):
        generate_channel(clean, channel, STAMP)
        generate_channel(poisoned, channel, STAMP)
        assert _written(poisoned, channel).equals(_written(clean, channel))


def test_unknown_channel_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown channel"):
        generate_channel(_r4_job(tmp_path, 5), "r7_new", STAMP)


# ---------------------------------------------------------------- R6

CHAIN_POOL = [0, 1, 3, 4]  # articles 2 and 5 are outside A(a)
D4 = _week("D4", 33, "D")
F1 = _week("F1", 40, "F")
T4 = _week("T4", 47, "T")
TINY_MANIFEST = FoldManifest(folds=(D4, F1, T4))
SUBMISSION_DAY = 54
CHAIN_QUERIES = make_queries([(1, 0), (2, 1), (3, 2)], fold="S")


def _tiny_chain(paths, start: ModelFile, checkpoint_file=None) -> TwoTowerChainChannel:
    """The tiny chain, loaded from a start model whose sha256 is checked first."""
    resources = _resources()
    check_file_sha256(start.path, start.sha256)
    load_checkpoint(resources.model, start.path)
    return TwoTowerChainChannel(
        CHAIN_CONFIG,
        resources,
        CHUNKS,
        checkpoint_file or paths.twotower_locked_chain_checkpoint_file,
        ALARM,
        MEMORY,
        {"model_identity": {}},
    )


@pytest.fixture
def starts(monkeypatch: pytest.MonkeyPatch) -> list[ModelFile]:
    """Replace the loader of the real locked chain by one that builds the tiny chain.

    Returns the list of start models it was asked for (in order).
    """
    asked: list[ModelFile] = []

    def load_tiny_chain(locked, base, paths, preprocess, start, checkpoint_file=None):
        asked.append(start)
        return _tiny_chain(paths, start, checkpoint_file)

    monkeypatch.setattr(recall_module, "load_locked_twotower_chain", load_tiny_chain)
    return asked


def _chain_job(tmp_path: Path, run: str, shard_customers: int = 2) -> RecallJob:
    """A job whose "locked chain" ends with a recorded F1.pt (the model that predicts T4)."""
    anchor_day = SUBMISSION_DAY if run == RUN_SUBMISSION else 47
    inputs = make_recall_inputs(LINES, anchor_day, active_pool=CHAIN_POOL)
    job = _job(
        tmp_path,
        anchor_day,
        CHAIN_QUERIES,
        inputs,
        shard_customers,
        run=run,
        manifest=TINY_MANIFEST,
    )
    paths = job.context.paths
    f1_model = paths.twotower_locked_chain_checkpoint_file("F1")
    if not f1_model.exists():
        save_checkpoint(_resources().model, f1_model, {"model_identity": {}})
        record = ChainRecord(
            "F1", "2020-02-10", "D4.pt", "0" * 64, "F1.pt", file_sha256(f1_model),
            3, 3, 1, 1, 6, 1, 5.0, 5.0, 1.0, 0.1, "abc123", False, "2026-11-19T00:00:00",
        )  # fmt: skip
        append_chain_record(paths.m6_twotower_chain_results_file, record)
    for name in ("catalog.yaml", "m6_candidates.yaml"):
        target = paths.configs_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text((REPO / "configs" / name).read_text())
    return job


def test_rehearsal_predicts_with_the_chains_last_model_and_trains_nothing(
    tmp_path: Path, starts: list[ModelFile]
) -> None:
    job = _chain_job(tmp_path, RUN_REHEARSAL)
    chain, detail = prepare_twotower_chain(job, STAMP)
    f1_model = job.context.paths.twotower_locked_chain_checkpoint_file("F1")
    assert starts == [ModelFile(f1_model, file_sha256(f1_model))]
    assert "no fine-tune" in detail
    # Nothing was written: no new model, no record.
    assert not job.context.paths.kaggle_twotower_results_file.exists()
    assert not (job.context.paths.kaggle_run_dir(RUN_REHEARSAL) / "twotower").exists()
    assert isinstance(chain, TwoTowerChainChannel)


@pytest.mark.parametrize("shard_customers", [1, 2, 3])
def test_r6_shards_give_the_unsharded_candidates(
    tmp_path: Path, starts: list[ModelFile], shard_customers: int
) -> None:
    job = _chain_job(tmp_path, RUN_REHEARSAL, shard_customers)
    result = generate_channel(job, R6, STAMP)
    written = _written(job, R6)
    frozen = TwoTowerChannel(CHAIN_CONFIG, _resources(), CHUNKS)
    assert written.equals(frozen.generate(job.queries, job.inputs).sort(KEYS))
    assert result.checked_rows == written.height == 6
    # A shard's vectors: its own customers, and EVERY catalog article (for tt_dot).
    saved = load_vectors(
        job.context.paths.kaggle_vectors_dir(RUN_REHEARSAL, 0, "twotower"), "twotower", "S"
    )
    assert saved.user_customer_idx.tolist() == [0, 1, 2][:shard_customers]
    assert saved.item_article_idx.tolist() == [0, 1, 2, 3, 4, 5]


def test_submission_fine_tunes_once_on_the_last_fold_and_records_the_model(
    tmp_path: Path, starts: list[ModelFile]
) -> None:
    job = _chain_job(tmp_path, RUN_SUBMISSION)
    paths = job.context.paths
    chain, detail = prepare_twotower_chain(job, STAMP)
    new_model = paths.kaggle_twotower_checkpoint_file(RUN_SUBMISSION, "T4")
    f1_model = paths.twotower_locked_chain_checkpoint_file("F1")
    record = read_chain_records(paths.kaggle_twotower_results_file)["T4"]
    # Started from the chain's last model; saved in the run's OWN folder with its sha256.
    assert (record.start_model, record.start_sha256) == ("F1.pt", file_sha256(f1_model))
    assert record.sha256 == file_sha256(new_model)
    assert new_model.parent == paths.kaggle_run_dir(RUN_SUBMISSION) / "twotower"
    # The T4 week [47, 54): 3 new rows of the non-alarm users, 1 : 1 replay.
    assert (record.n_new_rows, record.n_replay_rows, record.anchor) == (3, 3, "2020-02-17")
    assert "fine-tuned on T4" in detail and record.sha256 in detail
    # The locked chain's folder and record are untouched.
    assert sorted(file.name for file in f1_model.parent.iterdir()) == ["F1.pt"]
    assert list(read_chain_records(paths.m6_twotower_chain_results_file)) == ["F1"]
    # The model that predicts is the SAVED one, loaded again after its sha256 check.
    assert starts[-1] == ModelFile(new_model, record.sha256)
    saved = _resources()
    load_checkpoint(saved.model, new_model)
    for name, tensor in chain.resources.model.state_dict().items():
        assert np.array_equal(tensor.numpy(), saved.model.state_dict()[name].numpy())
    # The fine-tune changed the model (otherwise the test would prove nothing).
    start = _resources()
    load_checkpoint(start.model, f1_model)
    weights = start.model.state_dict()
    changed = [
        name
        for name, tensor in saved.model.state_dict().items()
        if not np.array_equal(tensor.numpy(), weights[name].numpy())
    ]
    assert changed


def test_submission_never_fine_tunes_twice(tmp_path: Path, starts: list[ModelFile]) -> None:
    job = _chain_job(tmp_path, RUN_SUBMISSION)
    paths = job.context.paths
    prepare_twotower_chain(job, STAMP)
    new_model = paths.kaggle_twotower_checkpoint_file(RUN_SUBMISSION, "T4")
    sha256, n_loads = file_sha256(new_model), len(starts)
    prepare_twotower_chain(job, STAMP)  # a resumed process
    assert file_sha256(new_model) == sha256
    assert list(read_chain_records(paths.kaggle_twotower_results_file)) == ["T4"]
    # Only the saved model was loaded the second time: no start from F1.pt, no training.
    assert [start.path.name for start in starts[n_loads:]] == ["T4.pt"]


def test_submission_refuses_a_model_without_its_record(
    tmp_path: Path, starts: list[ModelFile]
) -> None:
    job = _chain_job(tmp_path, RUN_SUBMISSION)
    prepare_twotower_chain(job, STAMP)
    job.context.paths.kaggle_twotower_results_file.unlink()
    with pytest.raises(ValueError, match="they must agree"):
        prepare_twotower_chain(job, STAMP)


def test_chain_refuses_a_last_model_whose_sha256_changed(
    tmp_path: Path, starts: list[ModelFile]
) -> None:
    job = _chain_job(tmp_path, RUN_SUBMISSION)
    f1_model = job.context.paths.twotower_locked_chain_checkpoint_file("F1")
    f1_model.write_bytes(f1_model.read_bytes() + b"x")
    with pytest.raises(ValueError, match="sha256"):
        prepare_twotower_chain(job, STAMP)
    assert starts == []  # nothing was loaded


def test_submission_week_is_predicted_with_the_fine_tuned_model(
    tmp_path: Path, starts: list[ModelFile]
) -> None:
    job = _chain_job(tmp_path, RUN_SUBMISSION, shard_customers=2)
    generate_channel(job, R6, STAMP)
    fine_tuned = _resources()
    new_model = job.context.paths.kaggle_twotower_checkpoint_file(RUN_SUBMISSION, "T4")
    load_checkpoint(fine_tuned.model, new_model)
    expected = TwoTowerChannel(CHAIN_CONFIG, fine_tuned, CHUNKS).generate(job.queries, job.inputs)
    assert _written(job, R6).equals(expected.sort(KEYS))
