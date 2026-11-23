"""Roll back: point `current` at another published prediction set (docs/mlops.md §9, §11; M14).

The service serves the prediction set state/current.json names and follows the
pointer without a restart. If the newest set is bad, the pointer is moved:

    previous    the newest set that is OLDER than the current one (roll back);
    next        the oldest set that is NEWER than the current one (roll forward
                again, one set at a time);
    a date      the set of that anchor.

A target is refused (RollbackRefused) unless ALL of this holds:

    it exists           predictions/{anchor}/manifest.json is there and its
                        week has the published marker of step O5;
    its checksum        the sha256 of predictions/{anchor}/top.parquet equals
                        the manifest's;
    it can be served    its serving form loads with every check of
                        hmrec.serving_set (sha256 of each file included) and
                        belongs to exactly that prediction file.

Nothing but state/current.json is written, plus one row per move in the table
pointer_moves.csv next to the pipeline's other tables. The pointer's content is
rebuilt from the prediction set's own manifest (anchor, mode, models, run id),
so moving back and forward again gives the file step O5 wrote.

What a rollback does NOT do: it changes no prediction set and no model, and it
does not stop the pipeline. The next tick is still the one after the latest
published week (hmrec.weekly.state.next_tick_anchor) and its step O5 moves the
pointer forward again.

This module imports neither torch nor lightgbm.
"""

import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from hmrec.csv_records import append_row
from hmrec.locks import file_sha256
from hmrec.logging_utils import get_logger
from hmrec.paths import ProjectPaths, StateLayout
from hmrec.run_records import RunStamp
from hmrec.serving_set import ServingRecord, ServingSetError, load_serving_set
from hmrec.weekly.calendar import DAYS_PER_WEEK
from hmrec.weekly.context import require_committed_code
from hmrec.weekly.state import (
    CurrentPointer,
    ModelPointer,
    read_current,
    switch_current,
)
from hmrec.weekly.storage import StateStore

LOGGER = get_logger("weekly.rollback")

PREVIOUS = "previous"
NEXT = "next"
POINTER_MOVES_TABLE = "pointer_moves"
POINTER_MOVE_COLUMNS: tuple[str, ...] = (
    "recorded_at",
    "from_anchor",
    "to_anchor",
    "target",
    "to_model_version",
    "to_prediction_sha256",
    "git_commit",
)


class RollbackRefused(RuntimeError):
    """The pointer was not moved: the target does not exist or failed a check."""


@dataclass(frozen=True)
class PointerMove:
    """One move of the `current` pointer.

    Attributes:
        from_anchor: The anchor the pointer named before.
        to_anchor: The anchor it names now.
        record: The serving record of the new set.
    """

    from_anchor: date
    to_anchor: date
    record: ServingRecord


def prediction_set_anchors(layout: StateLayout) -> list[date]:
    """The anchors of the prediction sets of a state, oldest first.

    A prediction set is a folder predictions/{ISO date}/ that holds a manifest.
    Other folders (the staging folder) are not prediction sets.
    """
    if not layout.predictions_dir.is_dir():
        return []
    anchors: list[date] = []
    for folder in layout.predictions_dir.iterdir():
        try:
            anchor = date.fromisoformat(folder.name)
        except ValueError:
            continue
        if layout.prediction_manifest_file(anchor).is_file():
            anchors.append(anchor)
    return sorted(anchors)


def target_anchor(anchors: list[date], current: date, target: str) -> date:
    """The anchor a rollback target names.

    Args:
        anchors: The anchors of the state's prediction sets, oldest first.
        current: The anchor of the current pointer.
        target: "previous", "next", or an ISO date.

    Raises:
        RollbackRefused: If there is no such prediction set, or the target is
            the current one already.
        ValueError: If `target` is none of the three forms.
    """
    if target == PREVIOUS:
        older = [anchor for anchor in anchors if anchor < current]
        if not older:
            raise RollbackRefused(f"no prediction set is older than the current one ({current})")
        return older[-1]
    if target == NEXT:
        newer = [anchor for anchor in anchors if anchor > current]
        if not newer:
            raise RollbackRefused(f"no prediction set is newer than the current one ({current})")
        return newer[0]
    try:
        anchor = date.fromisoformat(target)
    except ValueError as error:
        raise ValueError(
            f"the target must be {PREVIOUS!r}, {NEXT!r} or a date YYYY-MM-DD, got {target!r}"
        ) from error
    if anchor == current:
        raise RollbackRefused(f"the pointer already names the prediction set of {anchor}")
    if anchor not in anchors:
        raise RollbackRefused(f"there is no prediction set for {anchor}")
    return anchor


def checked_target(layout: StateLayout, anchor: date) -> tuple[dict[str, Any], ServingRecord]:
    """The manifest and the serving record of a target that passed every check.

    Raises:
        RollbackRefused: With the first check of the module docstring that fails.
    """
    manifest_file = layout.prediction_manifest_file(anchor)
    if not manifest_file.is_file():
        raise RollbackRefused(f"there is no prediction set for {anchor}: {manifest_file} not found")
    if not layout.published_file(anchor).is_file():
        raise RollbackRefused(f"the prediction set of {anchor} was never published by step O5")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    predictions_file = layout.prediction_file(anchor)
    if not predictions_file.is_file():
        raise RollbackRefused(f"{predictions_file} not found")
    recorded = manifest["predictions"]["sha256"]
    sha256 = file_sha256(predictions_file)
    if sha256 != recorded:
        raise RollbackRefused(
            f"checksum failed: {predictions_file} has sha256 {sha256}, its manifest says {recorded}"
        )
    try:
        serving = load_serving_set(layout.serving_dir(anchor))
    except ServingSetError as error:
        raise RollbackRefused(f"the serving form of {anchor} cannot be loaded: {error}") from error
    description = serving.record.description
    if description.prediction_sha256 != recorded or description.anchor != anchor.isoformat():
        raise RollbackRefused(
            f"the serving form of {anchor} does not belong to its prediction file "
            f"(it names sha256 {description.prediction_sha256} of {description.anchor})"
        )
    return manifest, serving.record


def pointer_of(manifest: dict[str, Any], anchor: date, layout: StateLayout) -> CurrentPointer:
    """The pointer step O5 wrote for a prediction set, rebuilt from the set's manifest.

    A tick at anchor `a` closes the week before `a`, so the latest closed week
    of that pointer is one week before the anchor.
    """
    folder = layout.prediction_set_dir(anchor)
    return CurrentPointer(
        anchor=anchor,
        mode=manifest["mode"],
        closed_week=anchor - timedelta(days=DAYS_PER_WEEK),
        prediction_set=folder.relative_to(layout.root).as_posix(),
        models=tuple(
            ModelPointer(model["role"], model["path"], model["sha256"])
            for model in manifest["models"]
        ),
        run_id=manifest["run_id"],
    )


def move_pointer(
    paths: ProjectPaths, store: StateStore, target: str, stamp: RunStamp
) -> PointerMove:
    """Point state/current.json at another published prediction set.

    Args:
        paths: The project paths of the state (hmrec.paths.tick_root_paths).
        store: The state (hmrec.weekly.driver.state_store).
        target: "previous", "next", or the anchor of a set as an ISO date.
        stamp: The commit of the code and whether it is dirty.

    Returns:
        The move that was made.

    Raises:
        RollbackRefused: If nothing is published yet, or the target does not
            exist or fails a check. The pointer is unchanged then.
        RuntimeError: If the code is not a committed tree.
    """
    require_committed_code(stamp)
    layout = StateLayout(paths.pipeline_state_dir)
    current = read_current(store, layout)
    if current is None:
        raise RollbackRefused("nothing is published yet: there is no current pointer to move")
    anchor = target_anchor(prediction_set_anchors(layout), current.anchor, target)
    manifest, record = checked_target(layout, anchor)
    switch_current(store, layout, pointer_of(manifest, anchor, layout))
    row = {
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "from_anchor": current.anchor.isoformat(),
        "to_anchor": anchor.isoformat(),
        "target": target,
        "to_model_version": record.model_version,
        "to_prediction_sha256": record.description.prediction_sha256,
        "git_commit": stamp.git_commit,
    }
    append_row(pointer_moves_file(paths), POINTER_MOVE_COLUMNS, row)
    LOGGER.info("current: %s -> %s (%s)", current.anchor, anchor, record.set_id)
    return PointerMove(current.anchor, anchor, record)


def pointer_moves_file(paths: ProjectPaths) -> Path:
    """The table of pointer moves, next to the pipeline's other tables of this state."""
    return paths.replay_table_file(POINTER_MOVES_TABLE)
