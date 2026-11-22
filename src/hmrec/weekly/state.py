"""What the state says about a week: closed, published, the current pointer (mlops §6.4).

    weeks/{anchor}/tick.json        written when a tick first starts: its mode. A
                                    tick is never run again in another mode.
    weeks/{anchor}/published.json   written by O5: the week's predictions are out.
    weeks/{anchor}/closed.json      written when C2-C4 of the week are done (by the
                                    NEXT tick: labels exist only after the week).
    state/current.json              the current prediction set, the latest closed
                                    week and the model files with their sha256.
                                    Replaced ONLY by step O5.

A week before the first tick is closed in the bootstrap state (docs/mlops.md
§6.3): it has no marker.

All files go through a StateStore (hmrec.weekly.storage). This module imports
only the standard library and the pipeline's calendar.
"""

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from hmrec.paths import StateLayout
from hmrec.weekly.calendar import Week
from hmrec.weekly.storage import StateStore

CURRENT_SCHEMA_VERSION = 1
CURRENT_KEYS = (
    "schema_version",
    "anchor",
    "mode",
    "closed_week",
    "prediction_set",
    "models",
    "run_id",
)
MODEL_KEYS = ("role", "path", "sha256")
RANKER_ROLE = "ranker"
TWOTOWER_ROLE = "twotower_chain"


@dataclass(frozen=True)
class ModelPointer:
    """A model file behind a prediction set.

    Attributes:
        role: "ranker" or "twotower_chain".
        path: Where the file is, relative to the state root, with "/".
        sha256: sha256 of the file.
    """

    role: str
    path: str
    sha256: str


@dataclass(frozen=True)
class CurrentPointer:
    """The content of state/current.json.

    Attributes:
        anchor: The week the current prediction set predicts.
        mode: "eval" or "serve".
        closed_week: Anchor of the latest closed week.
        prediction_set: The prediction set's folder, relative to the state root.
        models: The Ranker and the R6 chain model the predictions came from.
        run_id: The run of the tick that published it.
    """

    anchor: date
    mode: str
    closed_week: date
    prediction_set: str
    models: tuple[ModelPointer, ...]
    run_id: str


def pointer_to_json(pointer: CurrentPointer) -> dict[str, Any]:
    """The pointer as the JSON object that is written."""
    return {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "anchor": pointer.anchor.isoformat(),
        "mode": pointer.mode,
        "closed_week": pointer.closed_week.isoformat(),
        "prediction_set": pointer.prediction_set,
        "models": [
            {"role": model.role, "path": model.path, "sha256": model.sha256}
            for model in pointer.models
        ],
        "run_id": pointer.run_id,
    }


def pointer_from_json(raw: Any) -> CurrentPointer:
    """Build a CurrentPointer from a parsed state/current.json.

    Raises:
        ValueError: If a key is missing or unknown, or the schema version is another one.
    """
    if not isinstance(raw, dict) or set(raw) != set(CURRENT_KEYS):
        raise ValueError(f"state/current.json must have exactly the keys {CURRENT_KEYS}")
    if raw["schema_version"] != CURRENT_SCHEMA_VERSION:
        raise ValueError(
            f"state/current.json has schema_version {raw['schema_version']!r}, "
            f"expected {CURRENT_SCHEMA_VERSION}"
        )
    models = []
    for model in raw["models"]:
        if not isinstance(model, dict) or set(model) != set(MODEL_KEYS):
            raise ValueError(f"a model of state/current.json must have the keys {MODEL_KEYS}")
        models.append(ModelPointer(model["role"], model["path"], model["sha256"]))
    return CurrentPointer(
        anchor=date.fromisoformat(raw["anchor"]),
        mode=raw["mode"],
        closed_week=date.fromisoformat(raw["closed_week"]),
        prediction_set=raw["prediction_set"],
        models=tuple(models),
        run_id=raw["run_id"],
    )


def read_current(store: StateStore, layout: StateLayout) -> CurrentPointer | None:
    """The current pointer; None before the first tick has published."""
    if not store.exists(layout.current_file):
        return None
    return pointer_from_json(json.loads(store.read_text(layout.current_file)))


def switch_current(store: StateStore, layout: StateLayout, pointer: CurrentPointer) -> None:
    """Replace state/current.json. Only step O5 calls this (docs/mlops.md §6.4)."""
    text = json.dumps(pointer_to_json(pointer), indent=2) + "\n"
    store.replace_text(layout.current_file, text)


def write_marker(store: StateStore, path: Path, content: dict[str, Any]) -> None:
    """Write a marker file (tick.json, published.json, closed.json) once."""
    store.write_text_once(path, json.dumps(content, indent=2) + "\n")


def is_published(store: StateStore, layout: StateLayout, week: Week) -> bool:
    """True if the week's predictions were published (its tick finished O5)."""
    return store.exists(layout.published_file(week.anchor))


def is_closed(store: StateStore, layout: StateLayout, week: Week, first_week: Week) -> bool:
    """True if the week is closed: in the bootstrap state, or by a tick of the pipeline.

    Args:
        store: The state.
        layout: Its layout.
        week: The week asked about.
        first_week: The week the first tick opens; every week before it is
            closed in the bootstrap state.
    """
    if week.ordinal < first_week.ordinal:
        return True
    return store.exists(layout.closed_file(week.anchor))


def require_tick_may_start(
    store: StateStore, layout: StateLayout, week: Week, previous: Week, first_week: Week
) -> None:
    """Fail unless the tick that opens `week` may start (docs/mlops.md §6.4).

    A tick closes the week before its anchor and opens the week of its anchor.
    It may start only if the pipeline has reached that point without a gap:

        the first tick      the week before it is closed in the bootstrap state;
        every later tick    the tick before it has finished: the previous week
                            has published predictions (so there is something to
                            close), and the week before THAT is closed.

    Raises:
        RuntimeError: If the tick would skip a week, or lies before the first tick.
    """
    if week.ordinal < first_week.ordinal:
        raise RuntimeError(
            f"week {week.name} ({week.anchor}) is before the first tick {first_week.name} "
            f"({first_week.anchor}); the bootstrap state already holds it"
        )
    if week.ordinal == first_week.ordinal:
        return
    if not is_published(store, layout, previous):
        raise RuntimeError(
            f"the tick for {week.name} ({week.anchor}) cannot start: the previous week "
            f"{previous.name} ({previous.anchor}) is not closed and has no published "
            "predictions, so the tick before this one has not finished. A week cannot be "
            "skipped; run the ticks in order."
        )


def require_same_mode(
    store: StateStore, layout: StateLayout, week: Week, mode: str, run_id: str
) -> None:
    """Record the mode of a week's tick the first time; later runs must use the same mode.

    Raises:
        RuntimeError: If the tick of this week was started before in another mode.
    """
    tick_file = layout.week_dir(week.anchor) / "tick.json"
    if not store.exists(tick_file):
        content = {"anchor": week.anchor.isoformat(), "week": week.name, "mode": mode}
        write_marker(store, tick_file, {**content, "first_run_id": run_id})
        return
    recorded = json.loads(store.read_text(tick_file))["mode"]
    if recorded != mode:
        raise RuntimeError(
            f"the tick of week {week.name} ({week.anchor}) was started in mode {recorded!r}; "
            f"it cannot be continued in mode {mode!r}"
        )


def next_run_id(store: StateStore, layout: StateLayout, anchor: date, mode: str) -> str:
    """The id of a new run of a tick: "{anchor}-{mode}-a{attempt}".

    The attempt number counts the runs of this tick that already have a manifest
    or a folder, so a resumed tick gets a new id and never replaces a manifest.
    No clock is asked.
    """
    attempt = 1
    while store.exists(layout.runs_dir / f"{anchor.isoformat()}-{mode}-a{attempt}"):
        attempt += 1
    return f"{anchor.isoformat()}-{mode}-a{attempt}"
