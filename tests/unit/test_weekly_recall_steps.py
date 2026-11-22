"""Tests for weekly.recall_steps: C4 advances the R6 chain, O1 predicts with the new model.

The tiny chain example of test_twotower_chain, moved onto the real calendar (day
zero 2018-09-20; the example's day 40 becomes the V1 anchor, day 678):

    replay window [650, 678)   (0,1) (1,2) (2,3) (3,4) (0,5) (1,0)
    V1 week [678, 685)         (0,2,679) (1,3,680) (2,4,681) (3,1,682) (0,2,679) <- repeat line
    V2 week [685, 692)         (0,4,686) (1,5,687) (2,0,688) (3,2,689)

Customer 3 is the alarm user. The bootstrap state's "model after F4" is the
freshly built tiny model. The loader of the real locked chain is replaced by one
that builds the tiny chain (and checks the start model's sha256 like the real one).

    tick V2:  C4 fine-tunes on the V1 week from F4.pt -> weeks/2020-07-29/twotower/V1.pt
              new rows of the non-alarm users: (0,2) (1,3) (2,4) -> 3 new + 3 replay rows
              O1 predicts the V2 week with V1.pt
"""

import dataclasses
from pathlib import Path

import polars as pl
import pytest
from test_twotower_chain import ALARM, LINES, MEMORY
from test_twotower_chain import CONFIG as CHAIN_CONFIG
from test_twotower_infer import CHUNKS, _resources
from weekly_example import V2, V3, build_workspace, tick_context, write_processed

from hmrec.locks import check_file_sha256, file_sha256
from hmrec.paths import ProjectPaths
from hmrec.recall.chain_records import ChainRecord, append_chain_record, read_chain_records
from hmrec.recall.locked import ModelFile
from hmrec.recall.twotower.chain import TwoTowerChainChannel
from hmrec.recall.twotower.infer import TwoTowerChannel
from hmrec.recall.twotower.train import load_checkpoint, save_checkpoint
from hmrec.run_manifest import EVAL_MODE, SERVE_MODE
from hmrec.vectors import load_vectors
from hmrec.weekly import recall_steps
from hmrec.weekly.calendar import previous_week
from hmrec.weekly.context import TickContext
from hmrec.weekly.data_steps import run_ingest, scan_ingested_transactions
from hmrec.weekly.recall_steps import chain_model_of, run_fine_tune, run_recall
from hmrec.weekly.steps import step_by_name

DAY_SHIFT = 638  # day 40 of the tiny example is the V1 anchor day 678
SHIFTED = [(customer, article, day + DAY_SHIFT) for customer, article, day in LINES]
CHAIN_POOL = [0, 1, 3, 4]  # articles 2 and 5 are outside A(a)
R2, R6 = "r2_repurchase", "r6_twotower"


@pytest.fixture
def paths(tmp_path: Path) -> ProjectPaths:
    """A workspace whose bootstrap chain ends with a real (tiny) model and a full record."""
    workspace = build_workspace(tmp_path)
    write_processed(workspace, SHIFTED, n_customers=4, n_articles=6)
    model = workspace.bootstrap_dir / "models" / "twotower" / "chain_locked" / "F4.pt"
    model.unlink()
    save_checkpoint(_resources().model, model, {"model_identity": {}})
    table = workspace.bootstrap_monitoring_file("m6_twotower_chain.csv")
    table.unlink()
    record = ChainRecord(
        "F4", "2020-07-22", "F3.pt", "0" * 64, "F4.pt", file_sha256(model),
        3, 3, 1, 1, 6, 1, 5.0, 5.0, 1.0, 0.1, "abc123", False, "2026-11-22T00:00:00",
    )  # fmt: skip
    append_chain_record(table, record)
    return workspace


@pytest.fixture
def starts(monkeypatch: pytest.MonkeyPatch) -> list[ModelFile]:
    """Replace the real locked chain by the tiny one; give every week the example's active pool.

    Returns the list of start models the loader was asked for (in order).
    """
    asked: list[ModelFile] = []

    def load_tiny_chain(locked, base, paths, preprocess, start: ModelFile) -> TwoTowerChainChannel:
        asked.append(start)
        resources = _resources()
        check_file_sha256(start.path, start.sha256)
        load_checkpoint(resources.model, start.path)
        return TwoTowerChainChannel(
            CHAIN_CONFIG,
            resources,
            CHUNKS,
            paths.twotower_locked_chain_checkpoint_file,
            ALARM,
            MEMORY,
            {"model_identity": {}},
        )

    real_inputs = recall_steps.build_recall_inputs

    def inputs_with_the_example_pool(tables, fold, preprocess, catalog):
        pool = pl.DataFrame({"article_idx": CHAIN_POOL}, schema={"article_idx": pl.Int32})
        return dataclasses.replace(real_inputs(tables, fold, preprocess, catalog), active_pool=pool)

    monkeypatch.setattr(recall_steps, "load_locked_twotower_chain", load_tiny_chain)
    monkeypatch.setattr(recall_steps, "build_recall_inputs", inputs_with_the_example_pool)
    return asked


def _write_queries(context: TickContext, customers: list[int]) -> pl.DataFrame:
    week = context.tick
    queries = pl.DataFrame(
        {
            "query_id": [100 + customer for customer in customers],
            "fold": [week.name] * len(customers),
            "customer_idx": customers,
            "anchor": [week.anchor] * len(customers),
        },
        schema={
            "query_id": pl.Int32,
            "fold": pl.String,
            "customer_idx": pl.Int32,
            "anchor": pl.Date,
        },
    )
    target = context.view.queries_file(week.name)
    target.parent.mkdir(parents=True, exist_ok=True)
    queries.write_parquet(target)
    return queries


def _tick(paths: ProjectPaths, mode: str = EVAL_MODE) -> TickContext:
    context = tick_context(paths, V2, mode)
    run_ingest(context)
    return context


def _recall_step(context: TickContext, channel: str):
    name = f"O1_recall_{channel}"
    return step_by_name(name, context.tick, context.closing, context.mode)


# ---------------------------------------------------------------- C4


def test_fine_tune_starts_from_the_model_that_predicted_the_closed_week(
    paths: ProjectPaths, starts: list[ModelFile]
) -> None:
    context = _tick(paths)
    detail, metrics, files = run_fine_tune(context)
    f4_model = paths.bootstrap_dir / "models" / "twotower" / "chain_locked" / "F4.pt"
    assert starts == [ModelFile(f4_model, file_sha256(f4_model))]
    # The new model and its record are in the CLOSED week's folder of the state.
    folder = paths.pipeline_state_dir / "weeks" / "2020-07-29" / "twotower"
    record = read_chain_records(folder / "chain_record.csv")["V1"]
    assert (record.anchor, record.start_model, record.model) == ("2020-07-29", "F4.pt", "V1.pt")
    assert record.sha256 == file_sha256(folder / "V1.pt") != record.start_sha256
    # 3 new rows of the non-alarm users in the V1 week, and 3 replay rows (1 : 1).
    assert (record.n_new_rows, record.n_replay_rows, record.n_alarm_samples) == (3, 3, 1)
    assert "fine-tuned on V1: 3 new + 3 replay rows" in detail
    assert {metric.name for metric in metrics} == {
        "r6_chain/train_loss",
        "r6_chain/alarm_loss",
        "r6_chain/fine_tune_seconds",
    }
    assert {metric.week for metric in metrics} == {context.closing.anchor}
    assert [file.name for file in files] == ["V1.pt", "chain_record.csv"]
    # The bootstrap state was only read.
    assert list(read_chain_records(paths.bootstrap_monitoring_file("m6_twotower_chain.csv"))) == [
        "F4"
    ]


def test_fine_tune_reads_no_purchase_on_or_after_the_ticks_anchor(
    paths: ProjectPaths, starts: list[ModelFile]
) -> None:
    context = _tick(paths)
    # The V2 week (days 686 .. 689) is in the source but not in what the tick reads.
    days = scan_ingested_transactions(context).collect()["t_dat_day"]
    assert days.max() == 682
    run_fine_tune(context)
    record = read_chain_records(context.view.chain_record_file("V1"))["V1"]
    assert record.n_new_rows == 3


def test_a_model_is_never_trained_twice_and_the_stop_limit_holds(
    paths: ProjectPaths, starts: list[ModelFile], monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _tick(paths)
    run_fine_tune(context)
    with pytest.raises((FileExistsError, ValueError)):
        run_fine_tune(context)  # V1.pt and its record exist
    # A loss above the limit of configs/m6_candidates.yaml stops the chain.
    other = build_workspace(paths.project_root / "other")
    write_processed(other, SHIFTED, n_customers=4, n_articles=6)
    for name in ("models/twotower/chain_locked/F4.pt", "monitoring/m6_twotower_chain.csv"):
        (other.bootstrap_dir / name).write_bytes((paths.bootstrap_dir / name).read_bytes())
    limit_file = other.m6_config_file
    limit_file.write_text(
        limit_file.read_text().replace("max_fine_tune_loss: 7.0", "max_fine_tune_loss: 0.001")
    )
    strict = tick_context(other, V2)
    run_ingest(strict)
    with pytest.raises(ValueError, match="the chain stops for a human decision"):
        run_fine_tune(strict)


# ---------------------------------------------------------------- O1


def test_the_week_is_predicted_with_the_model_fine_tuned_on_the_week_before(
    paths: ProjectPaths, starts: list[ModelFile]
) -> None:
    context = _tick(paths)
    run_fine_tune(context)
    queries = _write_queries(context, [0, 1, 2])
    detail, _, files = run_recall(context, _recall_step(context, R6))
    v1_model = context.view.twotower_locked_chain_checkpoint_file("V1")
    assert starts[-1] == ModelFile(v1_model, file_sha256(v1_model))
    written = pl.read_parquet(context.view.candidates_file("V2", R6))
    # What the frozen channel gives with V1.pt's weights as of the V2 anchor.
    resources = _resources()
    load_checkpoint(resources.model, v1_model)
    setup = recall_steps.load_recall_setup(context)
    inputs = recall_steps.week_inputs(context, setup, context.tick)
    expected = TwoTowerChannel(CHAIN_CONFIG, resources, CHUNKS).generate(queries, inputs)
    assert written.sort("query_id", "rank").equals(expected.sort("query_id", "rank"))
    assert written.height == 6 and "V1.pt" in detail
    # Vectors: the query customers and every catalog article (for the tt_dot feature).
    saved = load_vectors(context.view.twotower_vectors_dir("V2"), "twotower", "V2")
    assert saved.user_customer_idx.tolist() == [0, 1, 2]
    assert saved.item_article_idx.tolist() == [0, 1, 2, 3, 4, 5]
    assert {file.role for file in files} == {"candidates/r6_twotower", "vectors/r6_twotower"}


def test_the_chain_cannot_skip_a_week(paths: ProjectPaths, starts: list[ModelFile]) -> None:
    context = _tick(paths)
    _write_queries(context, [0, 1, 2])
    # V1 was not fine-tuned: the V2 week cannot be predicted.
    with pytest.raises(ValueError, match="The chain may not skip a week"):
        run_recall(context, _recall_step(context, R6))
    later = tick_context(paths, V3)
    with pytest.raises(ValueError, match="no chain record"):
        chain_model_of(later, previous_week(later.tick, later.manifest))


def test_a_chain_model_that_changed_on_disk_is_refused(
    paths: ProjectPaths, starts: list[ModelFile]
) -> None:
    context = _tick(paths)
    run_fine_tune(context)
    _write_queries(context, [0, 1, 2])
    model = context.view.twotower_locked_chain_checkpoint_file("V1")
    model.write_bytes(model.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="sha256"):
        run_recall(context, _recall_step(context, R6))


def test_a_statistical_channel_writes_the_weeks_candidates(
    paths: ProjectPaths, starts: list[ModelFile]
) -> None:
    context = _tick(paths)
    _write_queries(context, [0, 1, 2, 3])
    detail, metrics, files = run_recall(context, _recall_step(context, R2))
    written = pl.read_parquet(context.view.candidates_file("V2", R2))
    # R2 returns what each customer bought before the anchor: nothing of the V2 week.
    bought_before = {(c, a) for c, a, day in SHIFTED if day < 685}
    assert {
        (q - 100, a) for q, a in written.select("query_id", "article_idx").rows()
    } == bought_before
    assert metrics[0].name == "candidates_rows/r2_repurchase" and metrics[0].value == written.height
    assert [file.role for file in files] == ["candidates/r2_repurchase"]
    assert starts == []  # R1-R5 load no chain model


def test_serve_mode_writes_the_candidates_shard_by_shard(
    paths: ProjectPaths, starts: list[ModelFile]
) -> None:
    context = _tick(paths, SERVE_MODE)
    _write_queries(context, [0, 1, 2, 3])
    detail, _, files = run_recall(context, _recall_step(context, R2))
    # 4 customers in shards of 2 (the workspace's shard size).
    assert "in 2 shards" in detail and files == []
    shards = [context.view.for_shard(shard).candidates_file("V2", R2) for shard in range(2)]
    assert all(file.is_file() for file in shards)
    assert pl.read_parquet(shards[0])["query_id"].unique().sort().to_list() == [100, 101]


def test_a_step_without_a_channel_is_refused(paths: ProjectPaths) -> None:
    context = _tick(paths)
    step = dataclasses.replace(_recall_step(context, R2), channel="r7_new")
    with pytest.raises(ValueError, match="names no channel"):
        run_recall(context, step)
