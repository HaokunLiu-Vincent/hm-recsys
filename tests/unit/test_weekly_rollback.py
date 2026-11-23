"""Tests for weekly.rollback: moving the `current` pointer between published prediction sets.

A state with three published sets (serving_example.write_state_set), anchors
2020-09-09, 2020-09-16 and 2020-09-23; `current` names the newest one, exactly as
step O5 wrote it.
"""

import dataclasses
import json
from datetime import date
from pathlib import Path

import pytest
from serving_example import pointer_text, write_pointer, write_state_set
from weekly_example import STAMP, build_workspace

from hmrec.csv_records import read_rows
from hmrec.paths import ProjectPaths, StateLayout
from hmrec.serving_set import CUSTOMER_IDS_FILE, SERVING_RECORD_FILE
from hmrec.weekly.rollback import (
    POINTER_MOVE_COLUMNS,
    RollbackRefused,
    move_pointer,
    pointer_moves_file,
    prediction_set_anchors,
    target_anchor,
)
from hmrec.weekly.state import next_tick_anchor, read_current
from hmrec.weekly.storage import LocalStateStore, MountedStateStore

ANCHORS = ("2020-09-09", "2020-09-16", "2020-09-23")
FIRST, MIDDLE, LAST = (date.fromisoformat(anchor) for anchor in ANCHORS)


@pytest.fixture
def paths(tmp_path: Path) -> ProjectPaths:
    workspace = build_workspace(tmp_path)
    for anchor in ANCHORS:
        write_state_set(workspace.pipeline_state_dir, anchor)
    write_pointer(workspace.pipeline_state_dir, ANCHORS[-1])
    return workspace


def _store(paths: ProjectPaths) -> LocalStateStore:
    return LocalStateStore(paths.pipeline_state_dir)


def _layout(paths: ProjectPaths) -> StateLayout:
    return StateLayout(paths.pipeline_state_dir)


def _pointer_text(paths: ProjectPaths) -> str:
    return _layout(paths).current_file.read_text(encoding="utf-8")


# ---------------------------------------------------------------- which set a target names


def test_targets_are_the_sets_next_to_the_current_one_or_a_date() -> None:
    anchors = [FIRST, MIDDLE, LAST]
    assert target_anchor(anchors, LAST, "previous") == MIDDLE
    assert target_anchor(anchors, MIDDLE, "previous") == FIRST
    assert target_anchor(anchors, FIRST, "next") == MIDDLE
    assert target_anchor(anchors, LAST, "2020-09-09") == FIRST
    with pytest.raises(RollbackRefused, match="no prediction set is older"):
        target_anchor(anchors, FIRST, "previous")
    with pytest.raises(RollbackRefused, match="no prediction set is newer"):
        target_anchor(anchors, LAST, "next")
    with pytest.raises(RollbackRefused, match="no prediction set for 2020-09-02"):
        target_anchor(anchors, LAST, "2020-09-02")
    with pytest.raises(RollbackRefused, match="already names"):
        target_anchor(anchors, LAST, "2020-09-23")
    with pytest.raises(ValueError, match="must be 'previous', 'next' or a date"):
        target_anchor(anchors, LAST, "yesterday")


def test_only_folders_with_a_manifest_are_prediction_sets(paths: ProjectPaths) -> None:
    layout = _layout(paths)
    (layout.predictions_dir / ".staging" / "2020-09-30").mkdir(parents=True)
    (layout.predictions_dir / "2020-09-30").mkdir()  # a folder without a manifest
    assert prediction_set_anchors(layout) == [FIRST, MIDDLE, LAST]


# ---------------------------------------------------------------- moving the pointer


def test_rolling_back_and_forward_again_gives_the_pointer_o5_wrote(paths: ProjectPaths) -> None:
    original = _pointer_text(paths)
    move = move_pointer(paths, _store(paths), "previous", STAMP)
    assert (move.from_anchor, move.to_anchor) == (LAST, MIDDLE)
    assert _pointer_text(paths) == pointer_text("2020-09-16")
    current = read_current(_store(paths), _layout(paths))
    assert current is not None and current.prediction_set == "predictions/2020-09-16"
    assert current.closed_week == date(2020, 9, 9) and current.run_id == "2020-09-16-serve-a1"
    forward = move_pointer(paths, _store(paths), "next", STAMP)
    assert (forward.from_anchor, forward.to_anchor) == (MIDDLE, LAST)
    assert _pointer_text(paths) == original
    # Every move is one row of the table.
    rows = read_rows(pointer_moves_file(paths), POINTER_MOVE_COLUMNS)
    assert [(row["from_anchor"], row["to_anchor"], row["target"]) for row in rows] == [
        ("2020-09-23", "2020-09-16", "previous"),
        ("2020-09-16", "2020-09-23", "next"),
    ]
    assert rows[0]["to_model_version"] == move.record.model_version
    assert rows[0]["git_commit"] == STAMP.git_commit


def test_a_set_can_be_named_by_its_anchor(paths: ProjectPaths) -> None:
    move = move_pointer(paths, _store(paths), "2020-09-09", STAMP)
    assert move.to_anchor == FIRST and _pointer_text(paths) == pointer_text("2020-09-09")


def test_the_pointer_moves_on_a_mounted_state_too(paths: ProjectPaths) -> None:
    move_pointer(paths, MountedStateStore(paths.pipeline_state_dir), "previous", STAMP)
    assert _pointer_text(paths) == pointer_text("2020-09-16")


def test_a_rollback_changes_nothing_but_the_pointer_and_its_table(paths: ProjectPaths) -> None:
    state = paths.pipeline_state_dir

    def content() -> dict[str, bytes]:
        files = (path for path in state.rglob("*") if path.is_file())
        return {str(path.relative_to(state)): path.read_bytes() for path in files}

    before = content()
    move_pointer(paths, _store(paths), "previous", STAMP)
    after = content()
    changed = {name for name in before | after if before.get(name) != after.get(name)}
    assert changed == {"state/current.json"}


# ---------------------------------------------------------------- refusals


def _refused(paths: ProjectPaths, target: str, message: str) -> None:
    before = _pointer_text(paths)
    with pytest.raises(RollbackRefused, match=message):
        move_pointer(paths, _store(paths), target, STAMP)
    assert _pointer_text(paths) == before
    assert not pointer_moves_file(paths).exists()


def test_a_target_that_does_not_exist_is_refused(paths: ProjectPaths) -> None:
    _refused(paths, "next", "no prediction set is newer")
    _refused(paths, "2020-08-05", "no prediction set for 2020-08-05")


def test_a_target_whose_prediction_file_fails_its_checksum_is_refused(paths: ProjectPaths) -> None:
    _layout(paths).prediction_file(MIDDLE).write_bytes(b"another table")
    _refused(paths, "previous", "checksum failed")


def test_a_target_without_a_loadable_serving_form_is_refused(paths: ProjectPaths) -> None:
    layout = _layout(paths)
    serving = layout.serving_dir(MIDDLE)
    data = (serving / CUSTOMER_IDS_FILE).read_bytes()
    # The last id ends with "0" (CAROL); one changed character is enough.
    (serving / CUSTOMER_IDS_FILE).write_bytes(data[:-1] + b"1")
    _refused(paths, "previous", "cannot be loaded.*sha256")
    (serving / SERVING_RECORD_FILE).unlink()
    _refused(paths, "previous", "cannot be loaded.*not complete")


def test_a_serving_form_of_another_prediction_file_is_refused(paths: ProjectPaths) -> None:
    layout = _layout(paths)
    record_file = layout.serving_dir(MIDDLE) / SERVING_RECORD_FILE
    raw = json.loads(record_file.read_text(encoding="utf-8"))
    record_file.write_text(json.dumps({**raw, "prediction_sha256": "0" * 64}), encoding="utf-8")
    _refused(paths, "previous", "does not belong to its prediction file")


def test_a_set_that_was_never_published_is_refused(paths: ProjectPaths) -> None:
    _layout(paths).published_file(MIDDLE).unlink()
    _refused(paths, "previous", "never published")


def test_nothing_is_moved_before_the_first_publication_or_from_uncommitted_code(
    paths: ProjectPaths,
) -> None:
    dirty = dataclasses.replace(STAMP, git_dirty=True)
    with pytest.raises(RuntimeError, match="committed code"):
        move_pointer(paths, _store(paths), "previous", dirty)
    _layout(paths).current_file.unlink()
    with pytest.raises(RollbackRefused, match="nothing is published yet"):
        move_pointer(paths, _store(paths), "previous", STAMP)


# ---------------------------------------------------------------- the pipeline after a rollback


def test_the_next_tick_follows_the_latest_published_week_not_a_rolled_back_pointer(
    paths: ProjectPaths,
) -> None:
    from hmrec.data.folds import load_fold_manifest
    from hmrec.weekly.calendar import week_named

    first_week = week_named("V1", load_fold_manifest(paths.fold_manifest_file))
    store, layout = _store(paths), _layout(paths)
    due = date(2020, 9, 30)  # one week after the newest published set
    assert next_tick_anchor(store, layout, first_week) == due
    move_pointer(paths, store, "2020-09-09", STAMP)
    # Two sets back: the tick that is due is still the one after the newest set.
    assert next_tick_anchor(store, layout, first_week) == due
