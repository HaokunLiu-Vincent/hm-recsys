"""Tests for weekly.history and weekly.feature_steps: where the rules' numbers come from.

Workspace of weekly_example.py: the bootstrap state holds F1-F4 (recall 0.10 for
every channel, 300 candidates per query, features pop_rank and user_age with
missing rate 0.2 and mean 10); the fake steps write the same numbers for the
weeks of the pipeline.
"""

from pathlib import Path

import polars as pl
import pytest
from weekly_example import (
    STAMP,
    V1,
    V2,
    V3,
    FakeSteps,
    build_workspace,
    feature_row,
    patch_driver_environment,
    tick_context,
)

from hmrec.config import load_monitoring_config
from hmrec.paths import ProjectPaths
from hmrec.run_manifest import EVAL_MODE, SERVE_MODE
from hmrec.weekly.calendar import week_named
from hmrec.weekly.driver import run_tick
from hmrec.weekly.feature_steps import (
    load_locked_features_config,
    require_structure,
    written_keys,
)
from hmrec.weekly.history import (
    earlier_labelled_flags,
    feature_history,
    feature_summaries,
    merged_feature_summaries,
    output_history,
    read_findings,
    recall_history,
    week_channel_recall,
    week_feature_summaries,
    week_output_signals,
    weeks_back,
)
from hmrec.weekly.monitoring import FeatureSummary

REPO = Path(__file__).resolve().parents[2]
CONFIG = load_monitoring_config(REPO / "configs" / "m12_monitoring.yaml")


@pytest.fixture
def paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ProjectPaths:
    patch_driver_environment(monkeypatch)
    return build_workspace(tmp_path)


def test_the_first_tick_gets_its_history_from_the_bootstrap_state(paths: ProjectPaths) -> None:
    context = tick_context(paths, V1)
    features = feature_history(context, CONFIG)
    assert len(features) == 4 and features[0]["pop_rank"] == FeatureSummary(0.2, 10.0)
    signals = output_history(context, CONFIG)
    # Stage 1 ranked no F week: only candidates per query has a history before V1.
    assert signals["candidates_per_query"] == [300.0] * 4
    assert signals["top_popular_share"] == [] and signals["distinct_articles"] == []


def test_history_mixes_bootstrap_weeks_and_pipeline_weeks(paths: ProjectPaths) -> None:
    run_tick(paths, V1, EVAL_MODE, FakeSteps(paths), STAMP)
    run_tick(paths, V2, EVAL_MODE, FakeSteps(paths), STAMP)
    context = tick_context(paths, V3)
    # The 4 weeks before V3: F3, F4 (bootstrap), V1, V2 (the state).
    assert len(feature_history(context, CONFIG)) == 4
    assert output_history(context, CONFIG)["distinct_articles"] == [40.0, 40.0]
    # The closed week of the V3 tick is V2; before it F2 F3 F4 (bootstrap) and V1 (closed).
    recall = recall_history(context, CONFIG, ["r4_itemcf"])
    assert recall == {"r4_itemcf": [0.10] * 4}
    v1 = week_named("V1", context.manifest)
    assert week_channel_recall(context, v1)["r6_twotower"] == 0.10
    # V2 is not closed yet: it has no recall.
    assert week_channel_recall(context, context.closing) == {}


def test_a_gap_ends_the_history(paths: ProjectPaths) -> None:
    context = tick_context(paths, V3)
    # V2 and V1 never ran: the weeks before V3 have no value, so the lookup stops at once
    # and does not jump back to the bootstrap weeks.
    assert feature_history(context, CONFIG) == []
    values = {"V2": 1.0, "V1": None, "F4": 3.0}
    found = weeks_back(context, context.tick, 4, lambda week: values.get(week.name))
    assert found == [1.0]


def test_serve_and_eval_weeks_are_never_each_others_history(paths: ProjectPaths) -> None:
    run_tick(paths, V1, EVAL_MODE, FakeSteps(paths), STAMP)
    serve = tick_context(paths, V2, SERVE_MODE)
    v1 = week_named("V1", serve.manifest)
    f4 = week_named("F4", serve.manifest)
    # For a serve tick neither the eval week V1 nor the bootstrap weeks are comparable.
    assert week_feature_summaries(serve, v1) is None and week_feature_summaries(serve, f4) is None
    assert week_output_signals(serve, v1) == {} and feature_history(serve, CONFIG) == []
    # The labelled history does not depend on the tick's mode: V1's recall is V1's recall.
    assert week_channel_recall(serve, f4)["r1_popularity"] == 0.10
    eval_tick = tick_context(paths, V2, EVAL_MODE)
    assert week_feature_summaries(eval_tick, v1) is not None


def test_labelled_flags_are_read_from_the_tick_that_closed_the_week(paths: ProjectPaths) -> None:
    for anchor in (V1, V2, V3):
        run_tick(paths, anchor, EVAL_MODE, FakeSteps(paths, union_recall=0.2), STAMP)
    from datetime import date

    context = tick_context(paths, date(2020, 8, 19))  # closes V3
    flags = earlier_labelled_flags(context, CONFIG)
    # Newest first: V2 (closed by the V3 tick), then V1 (closed by the V2 tick). F4 is a
    # bootstrap week and has no record, so the list ends there.
    assert flags == [frozenset({("recall/union", "")})] * 2
    findings = read_findings(context.layout.monitoring_file(V3))
    assert any(finding.rule == "recall/union" and finding.week == V2 for finding in findings)


def test_feature_summaries_skip_features_without_rows_and_merge_shards() -> None:
    empty = {**feature_row("V1", "unused"), "missing_rate": "", "mean": ""}
    category = {**feature_row("V1", "code", 0.5), "mean": ""}
    summaries = feature_summaries([feature_row("V1", "f", 0.25, 8.0), empty, category])
    assert summaries == {"f": FeatureSummary(0.25, 8.0), "code": FeatureSummary(0.5, None)}
    # Two shards of 100 rows: missing 20 and 40 -> 0.30; means 10 (80 present) and 20 (60
    # present) -> (800 + 1200) / 140.
    shards = [[feature_row("W", "f", 0.2, 10.0)], [feature_row("W", "f", 0.4, 20.0)]]
    merged = merged_feature_summaries(shards)["f"]
    assert merged.missing_rate == pytest.approx(0.30)
    assert merged.mean == pytest.approx(2000 / 140)


# ---------------------------------------------------------------- feature steps


def test_structural_checks_stop_on_a_wrong_row_count_or_a_repeated_key() -> None:
    require_structure("V1", written=9, distinct=9, expected=9)
    with pytest.raises(ValueError, match="9 feature rows, but the run's own candidates hold 10"):
        require_structure("V1", written=9, distinct=9, expected=10)
    with pytest.raises(ValueError, match="2 repeated"):
        require_structure("V1", written=9, distinct=7, expected=9)


def test_written_keys_are_read_back_from_the_part_files(paths: ProjectPaths) -> None:
    view = tick_context(paths, V1).view
    folder = view.features_dir("V1")
    folder.mkdir(parents=True)
    for part, rows in enumerate(([(1, 10), (1, 11)], [(2, 10), (1, 10)])):
        pl.DataFrame(
            rows, schema={"query_id": pl.Int32, "article_idx": pl.Int32}, orient="row"
        ).write_parquet(folder / f"part-{part:05d}.parquet")
    # 4 rows, 3 distinct pairs: (1, 10) is in both parts.
    assert written_keys(view, "V1") == (4, 3)


def test_the_feature_config_is_the_locked_one(paths: ProjectPaths) -> None:
    config = load_locked_features_config(paths)
    assert len(config.names) == 71
    features_file = paths.features_config_file
    features_file.write_text(features_file.read_text().replace("rrf_k: 60", "rrf_k: 61"))
    with pytest.raises(ValueError):
        load_locked_features_config(paths)
