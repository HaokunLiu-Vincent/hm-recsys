"""The prediction set the service holds, and how it follows the `current` pointer (mlops §9).

    PredictionStore     reads {state}/state/current.json, loads the serving form of
                        the prediction set it names, and keeps it in memory.
    PointerWatcher      a thread that asks the store to look at the pointer again
                        every `poll_seconds` seconds.

Switching without a restart: when the pointer's text changes, the new set is
loaded and checked COMPLETELY (hmrec.serving_set.load_serving_set) next to the
old one, and only then put in its place. A request sees the old set or the new
one, never a half-loaded one.

A set that fails to load leaves the old one in place. The failure is written
as an alert line (kind "prediction_set_load_failed"), once for each pointer
text, and the load is tried again at every poll: a file that was only late
appears then. At start-up there is no old set: the service then answers 503
until a set loads.

The state folder is only read. This module imports the standard library, numpy
(through hmrec.serving_set) and nothing else of the project.
"""

import json
import threading
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from hmrec.serving_set import ServingSet, ServingSetError, load_serving_set
from hmrec.structured_logs import (
    PREDICTION_SET_LOAD_FAILED,
    SEVERITY_INFO,
    emit_alert,
    json_line,
    write_line,
)

# Where the pointer and a serving form lie inside the state (hmrec.paths.StateLayout
# holds the same names for the pipeline; the service's image has no hmrec.paths).
POINTER_PARTS = ("state", "current.json")
SERVING_DIR = "serving"
LOG_KIND_FIELD = "hmrec_log"
LOG_KIND_SWITCH = "prediction_set_loaded"
UNCHANGED = "unchanged"
LOADED = "loaded"
FAILED = "failed"


@dataclass(frozen=True)
class LoadedSet:
    """A prediction set in memory.

    Attributes:
        serving: The loaded serving set.
        pointer_text: The exact text of current.json the set was loaded for.
        prediction_set: The set's folder as the pointer names it, e.g.
            "predictions/2020-09-23".
    """

    serving: ServingSet
    pointer_text: str
    prediction_set: str


def serving_folder(state_root: Path, pointer_text: str) -> tuple[Path, str, str]:
    """The serving folder the pointer names, the set's folder name and its anchor.

    Args:
        state_root: The folder that holds the state.
        pointer_text: The text of state/current.json.

    Raises:
        ServingSetError: If the text is not a pointer, or the folder it names
            would lie outside the state (an absolute path or one with "..").
    """
    try:
        pointer = json.loads(pointer_text)
        prediction_set = pointer["prediction_set"]
        anchor = pointer["anchor"]
    except (ValueError, KeyError, TypeError) as error:
        raise ServingSetError(f"state/current.json is not a pointer: {error}") from error
    if not isinstance(prediction_set, str) or not isinstance(anchor, str):
        raise ServingSetError("state/current.json: prediction_set and anchor must be text")
    relative = PurePosixPath(prediction_set)
    if relative.is_absolute() or ".." in relative.parts or prediction_set == "":
        raise ServingSetError(
            f"state/current.json names {prediction_set!r}, which is not a folder of the state"
        )
    return state_root.joinpath(*relative.parts) / SERVING_DIR, prediction_set, anchor


class PredictionStore:
    """The prediction set of the service; replaced when the pointer changes."""

    def __init__(self, state_root: Path) -> None:
        """Remember where the state is; nothing is read yet.

        Args:
            state_root: The folder that holds the pipeline's state (read only).
        """
        self._state_root = state_root
        self._current: LoadedSet | None = None
        # The pointer text whose load failed last: its alert line is written once.
        self._failed_pointer_text: str | None = None
        # One refresh at a time: the start-up load and the watcher never overlap.
        self._lock = threading.Lock()

    @property
    def current(self) -> LoadedSet | None:
        """The set that is served now; None until the first set has loaded."""
        return self._current

    def refresh(self) -> str:
        """Look at the pointer and load the set it names if it changed.

        Returns:
            "unchanged" if the pointer's text is the one of the set in memory,
            "loaded" if a new set was loaded and put in place, "failed" if the
            pointer or the set could not be read (the old set stays).
        """
        with self._lock:
            pointer_file = self._state_root.joinpath(*POINTER_PARTS)
            try:
                pointer_text = pointer_file.read_text(encoding="utf-8")
            except OSError as error:
                self._failed(f"<unreadable: {pointer_file}>", f"{pointer_file}: {error}", "")
                return FAILED
            if self._current is not None and pointer_text == self._current.pointer_text:
                return UNCHANGED
            prediction_set = ""
            try:
                folder, prediction_set, anchor = serving_folder(self._state_root, pointer_text)
                serving = load_serving_set(folder)
                if serving.record.description.anchor != anchor:
                    raise ServingSetError(
                        f"{folder} is the set of {serving.record.description.anchor}, "
                        f"the pointer says {anchor}"
                    )
            except ServingSetError as error:
                self._failed(pointer_text, str(error), prediction_set)
                return FAILED
            previous = self._current
            # One assignment: a request reads the old set or the new one.
            self._current = LoadedSet(serving, pointer_text, prediction_set)
            self._failed_pointer_text = None
            fields = {
                LOG_KIND_FIELD: LOG_KIND_SWITCH,
                "prediction_set": serving.record.set_id,
                "previous_prediction_set": ""
                if previous is None
                else previous.serving.record.set_id,
                "n_customers": serving.record.n_customers,
            }
            write_line(json_line(SEVERITY_INFO, "prediction set loaded", fields))
            return LOADED

    def _failed(self, pointer_text: str, reason: str, prediction_set: str) -> None:
        """Write the alert line of a failed load, once for each pointer text."""
        if pointer_text == self._failed_pointer_text:
            return
        self._failed_pointer_text = pointer_text
        kept = "" if self._current is None else self._current.serving.record.set_id
        fields = {
            "prediction_set": prediction_set,
            "reason": reason,
            "still_serving": kept,
        }
        emit_alert(PREDICTION_SET_LOAD_FAILED, "a prediction set could not be loaded", fields)


class PointerWatcher(threading.Thread):
    """Calls PredictionStore.refresh every `poll_seconds` seconds, in its own thread."""

    def __init__(self, store: PredictionStore, poll_seconds: float) -> None:
        """Prepare the thread (start() runs it).

        Args:
            store: The store to refresh.
            poll_seconds: Seconds between two looks at the pointer; > 0.

        Raises:
            ValueError: If poll_seconds is not positive.
        """
        if poll_seconds <= 0:
            raise ValueError(f"poll_seconds must be > 0, got {poll_seconds}")
        # daemon: the thread never keeps the process alive when the server stops.
        super().__init__(name="pointer-watcher", daemon=True)
        self._store = store
        self._poll_seconds = poll_seconds
        self._stopped = threading.Event()

    def run(self) -> None:
        """Wait, refresh, and again, until stop() is called."""
        # wait() returns True as soon as stop() sets the event, False after the timeout.
        while not self._stopped.wait(self._poll_seconds):
            self._store.refresh()

    def stop(self) -> None:
        """End the thread after its current wait or refresh."""
        self._stopped.set()
