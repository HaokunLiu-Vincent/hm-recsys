"""Tests for weekly.storage and weekly.state: staging, never overwriting, the current pointer."""

from datetime import date
from pathlib import Path

import pytest

from hmrec.data.folds import load_fold_manifest
from hmrec.paths import StateLayout
from hmrec.run_manifest import EVAL_MODE, SERVE_MODE
from hmrec.weekly.calendar import week_named
from hmrec.weekly.state import (
    CurrentPointer,
    ModelPointer,
    is_closed,
    is_published,
    next_run_id,
    pointer_from_json,
    pointer_to_json,
    read_current,
    require_same_mode,
    require_tick_may_start,
    switch_current,
    write_marker,
)
from hmrec.weekly.storage import LocalStateStore

REPO = Path(__file__).resolve().parents[2]
MANIFEST = load_fold_manifest(REPO / "configs" / "fold_manifest.yaml")
F4, V1, V2, V3 = (week_named(name, MANIFEST) for name in ("F4", "V1", "V2", "V3"))
POINTER = CurrentPointer(
    anchor=date(2020, 7, 29),
    mode=EVAL_MODE,
    closed_week=date(2020, 7, 22),
    prediction_set="predictions/2020-07-29",
    models=(ModelPointer("ranker", "weeks/2020-07-29/ranker_model/model.txt", "a" * 64),),
    run_id="2020-07-29-eval-a1",
)


@pytest.fixture
def store(tmp_path: Path) -> LocalStateStore:
    return LocalStateStore(tmp_path / "pipeline")


def _layout(store: LocalStateStore) -> StateLayout:
    return StateLayout(store.root)


# ---------------------------------------------------------------- storage


def test_a_staging_folder_becomes_the_output_by_one_rename(store: LocalStateStore) -> None:
    layout = _layout(store)
    staging = layout.staging_dir(V1.anchor, "features")
    staging.mkdir(parents=True)
    (staging / "part-00000.parquet").write_text("rows")
    final = layout.step_dir(V1.anchor, "features")
    store.commit_dir(staging, final)
    assert (final / "part-00000.parquet").read_text() == "rows" and not staging.exists()


def test_an_existing_output_is_never_overwritten(store: LocalStateStore) -> None:
    layout = _layout(store)
    final = layout.step_dir(V1.anchor, "features")
    final.mkdir(parents=True)
    (final / "features.json").write_text("first")
    staging = layout.staging_dir(V1.anchor, "features")
    staging.mkdir(parents=True)
    (staging / "features.json").write_text("second")
    with pytest.raises(FileExistsError, match="never overwritten"):
        store.commit_dir(staging, final)
    assert (final / "features.json").read_text() == "first"
    with pytest.raises(FileExistsError, match="never replaced"):
        store.write_text_once(final / "features.json", "third")
    with pytest.raises(FileNotFoundError, match="does not exist"):
        store.commit_dir(
            layout.staging_dir(V1.anchor, "scores"), layout.step_dir(V1.anchor, "scores")
        )


def test_failed_attempts_are_kept_aside_under_new_names(store: LocalStateStore) -> None:
    layout = _layout(store)
    failed = layout.failed_dir(V1.anchor)
    moved = []
    for attempt in ("first", "second"):
        staging = layout.staging_dir(V1.anchor, "shards/003/features")
        staging.mkdir(parents=True)
        (staging / "note.txt").write_text(attempt)
        moved.append(store.set_aside(staging, failed, "O2_features_shard_003"))
    assert [path.name for path in moved] == ["O2_features_shard_003.1", "O2_features_shard_003.2"]
    assert [(path / "note.txt").read_text() for path in moved] == ["first", "second"]


def test_the_store_touches_nothing_outside_its_root(store: LocalStateStore, tmp_path: Path) -> None:
    outside = tmp_path / "artifacts" / "features" / "V1"
    outside.mkdir(parents=True)
    with pytest.raises(ValueError, match="not inside the pipeline state"):
        store.discard_dir(outside)
    with pytest.raises(ValueError, match="not inside the pipeline state"):
        store.write_text_once(outside / "x.json", "{}")
    with pytest.raises(ValueError, match="never removed"):
        store.discard_dir(store.root)
    assert outside.is_dir()


def test_discard_removes_an_intermediate_folder(store: LocalStateStore) -> None:
    folder = _layout(store).step_dir(V1.anchor, "shards/000/features")
    folder.mkdir(parents=True)
    (folder / "part-00000.parquet").write_text("rows")
    store.discard_dir(folder)
    assert not folder.exists()
    store.discard_dir(folder)  # a missing folder is not an error


# ---------------------------------------------------------------- the current pointer


def test_current_pointer_round_trips_and_is_replaced_in_one_step(store: LocalStateStore) -> None:
    layout = _layout(store)
    assert read_current(store, layout) is None
    switch_current(store, layout, POINTER)
    assert read_current(store, layout) == POINTER
    later = CurrentPointer(
        date(2020, 8, 5), EVAL_MODE, date(2020, 7, 29), "predictions/2020-08-05", (), "r2"
    )
    switch_current(store, layout, later)
    assert read_current(store, layout) == later
    # No temporary file is left next to the pointer.
    assert [path.name for path in layout.current_file.parent.iterdir()] == ["current.json"]


def test_current_pointer_refuses_another_schema() -> None:
    raw = pointer_to_json(POINTER)
    assert pointer_from_json(raw) == POINTER
    with pytest.raises(ValueError, match="schema_version"):
        pointer_from_json({**raw, "schema_version": 2})
    with pytest.raises(ValueError, match="exactly the keys"):
        pointer_from_json({**raw, "note": "extra"})


# ---------------------------------------------------------------- may a tick start?


def test_the_first_tick_may_start_from_the_bootstrap_state(store: LocalStateStore) -> None:
    layout = _layout(store)
    # F4 has no marker: it is closed in the bootstrap state.
    assert is_closed(store, layout, F4, V1) and not is_closed(store, layout, V1, V1)
    require_tick_may_start(store, layout, V1, F4, V1)


def test_a_later_tick_needs_the_tick_before_it(store: LocalStateStore) -> None:
    layout = _layout(store)
    with pytest.raises(RuntimeError, match="A week cannot be skipped"):
        require_tick_may_start(store, layout, V2, V1, V1)
    write_marker(store, layout.published_file(V1.anchor), {"week": "V1"})
    assert is_published(store, layout, V1)
    require_tick_may_start(store, layout, V2, V1, V1)
    # V3 still cannot start: V2 was not published.
    with pytest.raises(RuntimeError, match="V2"):
        require_tick_may_start(store, layout, V3, V2, V1)
    with pytest.raises(RuntimeError, match="before the first tick"):
        require_tick_may_start(store, layout, F4, week_named("F3", MANIFEST), V1)


def test_a_week_keeps_the_mode_of_its_first_run(store: LocalStateStore) -> None:
    layout = _layout(store)
    require_same_mode(store, layout, V1, EVAL_MODE, "run-1")
    require_same_mode(store, layout, V1, EVAL_MODE, "run-2")
    with pytest.raises(RuntimeError, match="cannot be continued in mode 'serve'"):
        require_same_mode(store, layout, V1, SERVE_MODE, "run-3")


def test_run_ids_count_the_attempts_of_a_tick(store: LocalStateStore) -> None:
    layout = _layout(store)
    assert next_run_id(store, layout, V1.anchor, EVAL_MODE) == "2020-07-29-eval-a1"
    (layout.runs_dir / "2020-07-29-eval-a1").mkdir(parents=True)
    assert next_run_id(store, layout, V1.anchor, EVAL_MODE) == "2020-07-29-eval-a2"
    assert next_run_id(store, layout, V2.anchor, EVAL_MODE) == "2020-08-05-eval-a1"
