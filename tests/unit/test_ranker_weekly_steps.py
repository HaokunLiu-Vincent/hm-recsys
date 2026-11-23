"""Tests for weekly.ranker_steps (LightGBM side): train, score, publish, then close a week.

One synthetic week through the Ranker steps of two ticks, with a tiny Ranker
configuration (20 trees, ratio 2) and the "learnable" folds of ranker_example.py:
every query has 12 candidates, the one with the largest `ui_decay_ptype` is its
only label, and rrf_score knows nothing about it.

    bootstrap state     samples of F1 F2 F3 F4 (drawn with the real sampling code)
    tick V1             features of V1 (60 queries) -> O3 dataset on F1-F4, O3 train,
                        O4 score (Top-12 = all 12 candidates), O5 prediction set
    tick V2             C2 scores the PUBLISHED V1 predictions against V1's purchases
                        (built from the tick's own transactions), C3 samples V1;
                        the V2 model then trains on F2 F3 F4 V1.

Purchases: customer c bought the label article of query c once in the V1 week at
price 0.5, so the hit value of a Top-12 that holds every candidate is 60 x 0.5.
"""

import json
import sys
from pathlib import Path

import polars as pl
import pytest
from final_example import write_candidates
from ranker_example import (
    LOCK,
    METRICS,
    learnable_fold,
    tiny_base_config,
    tiny_ranker_config,
    write_features,
)
from ranker_example import (
    build_workspace as build_ranker_workspace,
)
from weekly_example import (
    V1,
    V2,
    build_workspace,
    tick_context,
    write_processed,
    write_r1_candidates,
)

from hmrec.hpo.v_trial import TrialInputs
from hmrec.locks import file_sha256, load_ranker_lock
from hmrec.paths import ProjectPaths, tick_root_paths
from hmrec.ranker.identity import model_config, ranker_config_id
from hmrec.ranker.locked import LockedRanker
from hmrec.ranker.platform_entry import (
    MODE_ACCEPTED,
    MODE_CALIBRATION,
    PlatformLockedRanker,
    PlatformNotAccepted,
)
from hmrec.ranker.runtime import read_lightgbm_runtime
from hmrec.ranker.sampling import read_sample_record, record_file
from hmrec.serving_set import load_serving_set
from hmrec.weekly import ranker_steps
from hmrec.weekly.context import TickContext
from hmrec.weekly.data_steps import run_ingest, run_queries
from hmrec.weekly.outputs import read_signals, read_week_evaluation, run_publish
from hmrec.weekly.ranker_steps import (
    RankerSetup,
    check_sample_is_plausible,
    dataset_plan,
    load_locked_for_this_platform,
    locked_channels,
    run_dataset,
    run_evaluate,
    run_sample,
    run_score,
    run_train,
)

N_QUERIES = 60
V1_FIRST_QUERY = 80_000_000  # V1 is fold number 8 of the manifest
V1_WEEK_DAY = 680  # inside [678, 685)
WIDE, LABELS = learnable_fold(V1_FIRST_QUERY, N_QUERIES, seed=7)


@pytest.fixture
def paths(tmp_path: Path) -> ProjectPaths:
    """A workspace with tiny processed data and the samples of F1-F4 in its bootstrap state."""
    workspace = build_workspace(tmp_path)
    # Customer c bought its query's label article in the V1 week; some history before it.
    lines = [(query - V1_FIRST_QUERY, article, V1_WEEK_DAY) for query, article in LABELS]
    lines += [(customer, customer % 12, 670 + customer % 8) for customer in range(N_QUERIES)]
    # Customer 0 has no age: its R1 list is the fallback list of the serving set (M14).
    write_processed(
        workspace, lines, n_customers=N_QUERIES, n_articles=12, customers_without_age=(0,)
    )
    # ranker_example writes the samples under an artifacts root; the bootstrap state has
    # the layout of artifacts/, so its folder is that root.
    build_ranker_workspace(workspace.bootstrap_dir.parent.parent / "scratch")
    scratch = workspace.bootstrap_dir.parent.parent / "scratch" / "artifacts" / "ranker"
    scratch.rename(workspace.bootstrap_dir / "ranker")
    return workspace


def _setup(context: TickContext) -> RankerSetup:
    """A tiny Ranker configuration in the shape the locked entry returns."""
    config = tiny_ranker_config()
    tuned = config.tuned_defaults
    config_id = ranker_config_id(model_config(config, tuned, "full", "col_wise"))
    locked = LockedRanker(
        lock=load_ranker_lock(context.paths.ranker_lock_file),
        config=config,
        tuned=tuned,
        num_trees=config.num_trees,
        histogram_mode="col_wise",
        num_threads=2,
        config_id=config_id,
    )
    inputs = TrialInputs(context.view, tiny_base_config(), config, context.manifest, LOCK, METRICS)
    return RankerSetup(locked, inputs, read_lightgbm_runtime())


def _open_v1(paths: ProjectPaths) -> TickContext:
    """The V1 tick up to its features: ingest, the query list, features and candidates."""
    context = tick_context(paths, V1)
    run_ingest(context)
    run_queries(context)
    half = V1_FIRST_QUERY + N_QUERIES // 2
    parts = [WIDE.filter(pl.col("query_id") < half), WIDE.filter(pl.col("query_id") >= half)]
    write_features(context.view.features_dir("V1"), "V1", parts)
    return context


def _publish_v1(paths: ProjectPaths) -> TickContext:
    """The whole V1 tick: train, score, and the prediction set in its final place."""
    context = _open_v1(paths)
    setup = _setup(context)
    run_dataset(context, setup)
    run_train(context, setup)
    run_score(context, setup)
    # O5 reads the fallback list of the serving set from the tick's R1 candidates. The
    # features of this example are synthetic, so an R1 file is written here: every
    # query holds the list 11, 10, ..., 0.
    queries = pl.read_parquet(context.view.queries_file("V1"))
    r1_lists = {query: list(range(11, -1, -1)) for query in queries["query_id"].to_list()}
    write_r1_candidates(context.view.candidates_file("V1", "r1_popularity"), r1_lists)
    run_publish(context)
    layout = context.layout
    layout.prediction_staging_dir(V1).rename(layout.prediction_set_dir(V1))
    return context


def _close_v1(paths: ProjectPaths) -> tuple[TickContext, RankerSetup]:
    """The V2 tick up to C1, with V1's candidates in place for the recall metrics."""
    opened = _publish_v1(paths)
    queries = pl.read_parquet(opened.view.queries_file("V1"))
    write_candidates(opened.view, "V1", WIDE)  # six candidate files from the rank columns
    queries.write_parquet(opened.view.queries_file("V1"))  # keep the real query list
    context = tick_context(paths, V2)
    run_ingest(context)
    return context, _setup(context)


# ---------------------------------------------------------------- the locked entry


def test_the_locked_configuration_comes_through_this_platforms_entry(paths: ProjectPaths) -> None:
    context = tick_context(paths, V1)
    runtime = read_lightgbm_runtime()
    if sys.platform == "darwin":
        # The laptop: the strict entry, exactly as in stage 1.
        locked = load_locked_for_this_platform(context, runtime)
        assert locked.config_id == load_ranker_lock(paths.ranker_lock_file).selection.config_id
    elif (paths.platform_record_file(f"linux-{_architecture()}")).is_file():
        # An accepted platform (the arm64 image): the tolerance-accepted entry, mode
        # "accepted"; the tick's outputs are not a stage-1 location.
        locked = load_locked_for_this_platform(context, runtime)
        assert locked.num_trees == load_ranker_lock(paths.ranker_lock_file).model.num_trees
    else:
        # A platform without a record: refused, nothing is trained.
        with pytest.raises(PlatformNotAccepted, match="no platform record"):
            load_locked_for_this_platform(context, runtime)


def test_a_calibration_tick_enters_in_calibration_mode_on_linux(
    paths: ProjectPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The entry is replaced by a recorder, so both ticks are tried on every platform:
    # on Linux a calibration tick asks for mode "calibration", any other tick for
    # "accepted"; the output location is the tick's own view both times.
    asked: list[tuple[str, Path]] = []
    strict = LockedRanker.__new__(LockedRanker)

    def recorder(paths_, base, runtime, system, machine, mode, output_paths):
        asked.append((mode, output_paths.layout.root))
        return PlatformLockedRanker(locked=strict, mode=mode, runtime=runtime, record=None)

    monkeypatch.setattr(ranker_steps, "load_locked_ranker_for_platform", recorder)
    monkeypatch.setattr(ranker_steps.sys, "platform", "linux")
    runtime = read_lightgbm_runtime()
    plain = tick_context(paths, V1)
    calibration = tick_context(tick_root_paths(paths, {}, calibration=True), V1)
    assert load_locked_for_this_platform(plain, runtime) is strict
    assert load_locked_for_this_platform(calibration, runtime) is strict
    assert asked == [
        (MODE_ACCEPTED, paths.pipeline_state_dir),
        (MODE_CALIBRATION, paths.artifacts_root / "pipeline-calibration"),
    ]


def _architecture() -> str:
    import platform

    from hmrec.platform_fields import ARCHITECTURE_NAMES

    return ARCHITECTURE_NAMES.get(platform.machine(), platform.machine())


def test_channel_locks_are_read_without_the_recall_code(paths: ProjectPaths) -> None:
    channels = locked_channels(paths)
    assert list(channels) == [
        "r1_popularity",
        "r2_repurchase",
        "r3_sibling",
        "r4_itemcf",
        "r5_wals",
        "r6_twotower",
    ]
    assert channels["r1_popularity"].d_phase_record.metric == "recall@100"
    assert "torch" not in sys.modules


# ---------------------------------------------------------------- O3, O4, O5


def test_the_model_of_the_first_week_trains_on_the_bootstrap_samples(paths: ProjectPaths) -> None:
    context = _open_v1(paths)
    setup = _setup(context)
    plan = dataset_plan(context, setup)
    assert plan.record.train_folds == ("F1", "F2", "F3", "F4")
    assert all(paths.bootstrap_dir in source.sample_file.parents for source in plan.sources)
    detail, metrics, files = run_dataset(context, setup)
    week = paths.pipeline_state_dir / "weeks" / "2020-07-29"
    assert (week / "ranker_dataset" / "dataset.json").is_file() and "F1 F2 F3 F4" in detail
    detail, _, files = run_train(context, setup)
    record = json.loads((week / "ranker_model" / "model.json").read_text())
    assert record["sha256"] == file_sha256(week / "ranker_model" / "model.txt")
    assert record["train_folds"] == ["F1", "F2", "F3", "F4"] and record["num_trees"] == 20
    assert record["config_id"] == setup.locked.config_id and record["lightgbm_version"]
    assert [file.role for file in files] == ["ranker_model", "ranker_model_record"]
    with pytest.raises(FileExistsError, match="never replaced"):
        run_train(context, setup)


def test_score_writes_a_checked_top_k_and_the_output_signals(paths: ProjectPaths) -> None:
    context = _open_v1(paths)
    setup = _setup(context)
    run_dataset(context, setup)
    run_train(context, setup)
    detail, metrics, files = run_score(context, setup)
    top = pl.read_parquet(context.view.top_file("V1"))
    assert top.columns == ["query_id", "rank", "article_idx", "score"]
    assert top.height == N_QUERIES * 12 and top["query_id"].n_unique() == N_QUERIES
    # The model learned the signal: the label is at rank 1 for most queries.
    first = top.filter(pl.col("rank") == 1).select("query_id", "article_idx")
    hits = first.join(
        pl.DataFrame(LABELS, schema=first.schema, orient="row"), on=["query_id", "article_idx"]
    )
    assert hits.height > 0.8 * N_QUERIES
    signals = read_signals(context.view.signals_file("V1"))
    assert (signals.n_queries, signals.candidates_per_query, signals.distinct_articles) == (
        N_QUERIES,
        12.0,
        12,
    )
    assert {metric.name for metric in metrics} == {
        "output/candidates_per_query",
        "output/top_popular_share",
        "output/distinct_articles",
    }
    assert f"scored {N_QUERIES * 12} candidates" in detail


def test_a_model_that_changed_on_disk_is_not_used_for_scoring(paths: ProjectPaths) -> None:
    context = _open_v1(paths)
    setup = _setup(context)
    run_dataset(context, setup)
    run_train(context, setup)
    model = context.view.ranker_model_file("weekly", "full", setup.locked.config_id, "V1")
    model.write_text(model.read_text() + "\n")
    with pytest.raises(ValueError, match="sha256"):
        run_score(context, setup)


def test_the_prediction_set_names_the_models_behind_it(paths: ProjectPaths) -> None:
    context = _publish_v1(paths)
    folder = context.layout.prediction_set_dir(V1)
    predictions = pl.read_parquet(folder / "top.parquet")
    assert predictions.columns == ["customer_idx", "rank", "article_idx"]
    assert (
        predictions.height == N_QUERIES * 12 and predictions["customer_idx"].n_unique() == N_QUERIES
    )
    manifest = json.loads((folder / "manifest.json").read_text())
    assert (manifest["week"], manifest["mode"], manifest["n_customers"], manifest["top_k"]) == (
        "V1",
        "eval",
        N_QUERIES,
        12,
    )
    assert manifest["predictions"]["sha256"] == file_sha256(folder / "top.parquet")
    models = {model["role"]: model for model in manifest["models"]}
    assert models["ranker"]["path"] == "weeks/2020-07-29/ranker_model/model.txt"
    # The first week is predicted by the bootstrap state's chain model.
    assert models["twotower_chain"]["path"].endswith("chain_locked/F4.pt")


def test_the_publish_step_writes_the_serving_form_with_the_prediction_set(
    paths: ProjectPaths,
) -> None:
    context = _publish_v1(paths)
    layout = context.layout
    serving = load_serving_set(layout.serving_dir(V1))
    record, description = serving.record, serving.record.description
    assert (record.n_customers, record.top_k, record.n_articles) == (N_QUERIES, 12, 12)
    # It belongs to exactly the published prediction file and names its Ranker.
    manifest = json.loads(layout.prediction_manifest_file(V1).read_text())
    assert description.prediction_sha256 == manifest["predictions"]["sha256"]
    ranker = next(model for model in manifest["models"] if model["role"] == "ranker")
    assert description.ranker_sha256 == ranker["sha256"]
    assert record.model_version == ranker["sha256"][:12]
    assert description.built_by == manifest["run_id"]
    # The fallback list is the R1 list of the customer without an age: articles 11 ... 0.
    assert description.fallback.article_ids == tuple(f"{a:010d}" for a in range(11, -1, -1))
    assert description.fallback.age_bucket == "AGE_UNKNOWN"
    # A customer's answer is its row of the prediction set, in original ids.
    predictions = pl.read_parquet(layout.prediction_file(V1))
    for customer in (0, 17, N_QUERIES - 1):
        rows = predictions.filter(pl.col("customer_idx") == customer).sort("rank")
        expected = tuple(f"{article:010d}" for article in rows["article_idx"].to_list())
        assert serving.recommend(f"{customer:064x}") == (expected, "ranker")
    assert serving.recommend(f"{N_QUERIES:064x}")[1] == "fallback"


# ---------------------------------------------------------------- C2, C3


def test_closing_scores_the_published_predictions_against_the_weeks_purchases(
    paths: ProjectPaths,
) -> None:
    context, setup = _close_v1(paths)
    detail, metrics, files = run_evaluate(context, setup)
    evaluation = read_week_evaluation(context.view.evaluation_file("V1"))
    assert (evaluation.week, evaluation.n_queries, evaluation.n_label_pairs) == ("V1", 60, 60)
    # Every label is a candidate of at least one channel, and the Top-12 holds all 12.
    assert evaluation.union_recall == 1.0 and evaluation.ranker["recall@12"] == 1.0
    assert evaluation.hit_value == pytest.approx(N_QUERIES * 0.5)
    # The Ranker learned the signal; RRF knows nothing about it.
    assert evaluation.ranker["map@12"] > evaluation.rrf_baseline["map@12"]
    assert set(evaluation.channel_recall) == set(locked_channels(paths))
    by_name = {metric.name: metric for metric in metrics}
    assert by_name["map@12"].value == evaluation.ranker["map@12"]
    assert by_name["rrf_baseline/map@12"].value == evaluation.rrf_baseline["map@12"]
    # A labelled metric names the week it describes: the CLOSED one.
    assert {metric.week for metric in metrics} == {V1}
    labels = pl.read_parquet(context.view.labels_file("V1"))
    assert labels.height == 60 and labels["fold"].unique().to_list() == ["V1"]
    assert "V1: 60 queries" in detail and len(files) == 2


def test_closing_refuses_a_week_that_is_not_over(paths: ProjectPaths) -> None:
    opened = _publish_v1(paths)
    # A tick AT the V1 anchor would close F4, not V1: asking it for V1's labels is refused.
    import dataclasses

    wrong = dataclasses.replace(tick_context(paths, V2), tick=opened.tick)
    with pytest.raises(ValueError, match="the week is not over"):
        run_evaluate(wrong, _setup(wrong))
    with pytest.raises(ValueError, match="the week is not over"):
        run_sample(wrong, _setup(wrong))


def test_the_closed_weeks_sample_trains_the_next_model(paths: ProjectPaths) -> None:
    context, setup = _close_v1(paths)
    run_evaluate(context, setup)
    detail, metrics, files = run_sample(context, setup)
    sample = context.view.ranker_sample_file("full", "V1", 2)
    assert sample == paths.pipeline_state_dir / "weeks" / "2020-07-29" / "sample" / "r2.parquet"
    record = read_sample_record(record_file(sample))
    # One positive per query and 2 negatives per positive (the tiny ratio).
    assert (record.n_queries, record.n_positives, record.n_rows, record.seed) == (60, 60, 180, 42)
    assert record.sha256 == file_sha256(sample)
    assert {metric.name: metric.value for metric in metrics}["sample/rows"] == 180.0
    # The model that predicts V2 trains on F2 F3 F4 (bootstrap) and V1 (the state).
    plan = dataset_plan(context, setup)
    assert plan.record.train_folds == ("F2", "F3", "F4", "V1")
    assert plan.sources[-1].sample_file == sample


def test_a_week_that_is_not_closed_cannot_train_a_model(paths: ProjectPaths) -> None:
    context, setup = _close_v1(paths)
    with pytest.raises(FileNotFoundError, match="the sample is not finished"):
        dataset_plan(context, setup)  # V1 has no sample yet


def test_sample_counts_must_be_plausible(paths: ProjectPaths) -> None:
    context, setup = _close_v1(paths)
    run_evaluate(context, setup)
    run_sample(context, setup)
    record = read_sample_record(record_file(context.view.ranker_sample_file("full", "V1", 2)))
    check_sample_is_plausible(record, 2, 60)
    import dataclasses

    with pytest.raises(ValueError, match="70 sampled queries of 60 queries"):
        check_sample_is_plausible(dataclasses.replace(record, n_queries=70), 2, 60)
    with pytest.raises(ValueError, match="rows for 60 positives at ratio 2"):
        check_sample_is_plausible(dataclasses.replace(record, n_rows=500), 2, 60)
