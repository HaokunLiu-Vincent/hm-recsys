"""Tests for weekly.calendar and weekly.steps: weeks on the fold grid, and the plan of a tick."""

from datetime import date
from pathlib import Path

import pytest

from hmrec.data.folds import load_fold_manifest
from hmrec.paths import StateLayout
from hmrec.run_manifest import EVAL_MODE, SERVE_MODE
from hmrec.weekly.calendar import (
    LATER_PHASE,
    previous_week,
    query_id_base,
    week_at,
    week_named,
    weeks_before,
    weeks_from,
)
from hmrec.weekly.steps import (
    SIDE_LIGHTGBM,
    SIDE_NONE,
    SIDE_TORCH,
    final_dir,
    staging_dir,
    step_by_name,
    steps_before_shards,
    steps_of_shards,
)

REPO = Path(__file__).resolve().parents[2]
MANIFEST = load_fold_manifest(REPO / "configs" / "fold_manifest.yaml")
OFFSET = 10_000_000


def test_a_manifest_anchor_is_its_fold() -> None:
    week = week_at(date(2020, 7, 29), MANIFEST)
    assert (week.name, week.fold.phase, week.ordinal, week.in_manifest) == ("V1", "V", 8, True)
    assert week.fold == MANIFEST.get("V1") and week.end_exclusive == date(2020, 8, 5)
    assert week_named("T4", MANIFEST).ordinal == 15


def test_a_week_after_the_last_fold_has_no_fold_and_another_phase() -> None:
    week = week_at(date(2020, 9, 23), MANIFEST)
    assert (week.name, week.ordinal, week.in_manifest) == ("W20200923", 16, False)
    # "S" is none of D, F, V, T: every stage-1 phase guard refuses the week.
    assert week.fold.phase == LATER_PHASE and LATER_PHASE not in ("D", "F", "V", "T")
    assert (week.fold.anchor, week.fold.label_end) == (date(2020, 9, 23), date(2020, 9, 29))
    assert week.name not in MANIFEST.names


@pytest.mark.parametrize("anchor", [date(2020, 7, 30), date(2020, 5, 27), date(2019, 1, 2)])
def test_an_anchor_off_the_weekly_grid_is_refused(anchor: date) -> None:
    with pytest.raises(ValueError, match="not an anchor of the weekly grid"):
        week_at(anchor, MANIFEST)


def test_previous_weeks_are_the_rankers_training_window() -> None:
    v2 = week_named("V2", MANIFEST)
    assert previous_week(v2, MANIFEST).name == "V1"
    assert [week.name for week in weeks_before(v2, MANIFEST, 4)] == ["F2", "F3", "F4", "V1"]
    serve = week_at(date(2020, 9, 23), MANIFEST)
    assert [week.name for week in weeks_before(serve, MANIFEST, 4)] == ["T1", "T2", "T3", "T4"]
    assert [w.name for w in weeks_from(week_named("T3", MANIFEST), serve, MANIFEST)] == [
        "T3",
        "T4",
        "W20200923",
    ]
    with pytest.raises(ValueError, match="only 2 weeks before it"):
        weeks_before(week_named("D3", MANIFEST), MANIFEST, 4)


def test_query_ids_are_those_of_stage_1_and_go_on_after_the_manifest() -> None:
    # V1 is fold number 8: its first query_id is 80,000,000 (artifacts/queries/V1.parquet).
    assert query_id_base(week_named("V1", MANIFEST), OFFSET) == 80_000_000
    # The week after T4 continues the numbering: 16 x offset.
    assert query_id_base(week_at(date(2020, 9, 23), MANIFEST), OFFSET) == 160_000_000
    far = week_at(date(2025, 1, 1), MANIFEST)
    with pytest.raises(ValueError, match="does not fit in int32"):
        query_id_base(far, OFFSET)


# ---------------------------------------------------------------- the plan of a tick


def test_eval_plan_follows_the_order_of_the_design() -> None:
    tick, closing = week_named("V2", MANIFEST), week_named("V1", MANIFEST)
    steps = steps_before_shards(tick, closing, EVAL_MODE)
    names = [step.name for step in steps]
    assert names[:5] == ["C1_ingest", "C2_evaluate", "C3_sample", "C4_fine_tune_r6", "O1_queries"]
    assert names[-5:] == ["O2_features", "O3_dataset", "O3_train", "O4_score", "O5_publish"]
    by_name = {step.name: step for step in steps}
    # C2-C4 write into the closed week's folder; C1 and the opening steps into the tick's.
    assert {by_name[name].week.name for name in names[1:4]} == {"V1"}
    assert by_name["C1_ingest"].week.name == by_name["O2_features"].week.name == "V2"
    # torch and LightGBM never share a process: a step is on one side or on none.
    assert by_name["C4_fine_tune_r6"].side == by_name["O1_recall_r5_wals"].side == SIDE_TORCH
    assert by_name["C2_evaluate"].side == by_name["O3_train"].side == SIDE_LIGHTGBM
    assert by_name["O2_features"].side == by_name["C1_ingest"].side == SIDE_NONE


def test_serve_plan_builds_and_scores_one_shard_at_a_time() -> None:
    tick, closing = week_at(date(2020, 9, 23), MANIFEST), week_named("T4", MANIFEST)
    before = [step.name for step in steps_before_shards(tick, closing, SERVE_MODE)]
    assert before[-3:] == ["O2_features_shared", "O3_dataset", "O3_train"]
    assert "O2_features" not in before and "O5_publish" not in before
    shards = [step.name for step in steps_of_shards(tick, 2)]
    assert shards == [
        "O2_features_shard_000",
        "O4_score_shard_000",
        "O2_features_shard_001",
        "O4_score_shard_001",
        "O4_score",
        "O5_publish",
    ]
    with pytest.raises(ValueError, match="at least one shard"):
        steps_of_shards(tick, 0)


def test_a_step_is_found_again_by_its_name() -> None:
    tick, closing = week_at(date(2020, 9, 23), MANIFEST), week_named("T4", MANIFEST)
    shard = step_by_name("O4_score_shard_017", tick, closing, SERVE_MODE)
    assert (shard.shard, shard.step_dir, shard.side) == (17, "shards/017/top", SIDE_LIGHTGBM)
    assert step_by_name("O4_score", tick, closing, SERVE_MODE).side == SIDE_NONE  # the merge
    assert step_by_name("O4_score", tick, closing, EVAL_MODE).side == SIDE_LIGHTGBM
    assert step_by_name("O1_recall_r6_twotower", tick, closing, EVAL_MODE).channel == "r6_twotower"
    with pytest.raises(ValueError, match="has no step"):
        step_by_name("O2_features_shard_000", tick, closing, EVAL_MODE)
    with pytest.raises(ValueError, match="mode must be one of"):
        step_by_name("C1_ingest", tick, closing, "train")


def test_a_step_writes_under_staging_until_it_is_renamed(tmp_path: Path) -> None:
    layout = StateLayout(tmp_path)
    tick, closing = week_named("V2", MANIFEST), week_named("V1", MANIFEST)
    sample = step_by_name("C3_sample", tick, closing, EVAL_MODE)
    assert final_dir(layout, sample) == tmp_path / "weeks" / "2020-07-29" / "sample"
    assert staging_dir(layout, sample) == tmp_path / "weeks" / "2020-07-29" / ".staging" / "sample"
    publish = step_by_name("O5_publish", tick, closing, EVAL_MODE)
    assert final_dir(layout, publish) == tmp_path / "predictions" / "2020-08-05"
    assert staging_dir(layout, publish) == tmp_path / "predictions" / ".staging" / "2020-08-05"
