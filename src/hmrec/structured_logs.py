"""Structured log lines: one JSON object per line (docs/mlops.md §9, §10; M14).

Cloud Run sends every line a container writes to its output to Cloud Logging. A
line that is a JSON object becomes a structured entry: its keys can be searched
and an alert policy can match them. Two kinds of lines are written this way:

    alert lines     one line for each event that must reach a person. Every
                    alert line holds the FIXED field `hmrec_alert` with the kind
                    of the event. The alert policies of infra/setup.sh match
                    that field and nothing else:

                        step_failed                  a step failed, or a check
                                                     stopped a tick
                        input_check_failed           the input-data check of C1
                                                     stopped a tick
                        quality_alert                a labelled quality flag was
                                                     raised in 3 consecutive weeks
                        prediction_set_load_failed   the service could not load
                                                     a prediction set

    request lines   one line per request of the service (hmrec.serving.app).

On the laptop the same lines appear in the terminal; nothing is sent anywhere.

A drill can be labelled: if the environment holds HMREC_ALERT_NOTE, its text is
added to every alert line as the field `note` (for example "provoked on
purpose"). The note changes nothing else.

This module is at the bottom layer and imports only the standard library: the
pipeline (both sides) and the service's small image use it.
"""

import json
import os
import sys
from collections.abc import Mapping
from typing import TextIO

# The fixed field of an alert line. infra/setup.sh matches `jsonPayload.hmrec_alert`.
ALERT_FIELD = "hmrec_alert"
STEP_FAILED = "step_failed"
INPUT_CHECK_FAILED = "input_check_failed"
QUALITY_ALERT = "quality_alert"
PREDICTION_SET_LOAD_FAILED = "prediction_set_load_failed"
ALERT_KINDS: tuple[str, ...] = (
    STEP_FAILED,
    INPUT_CHECK_FAILED,
    QUALITY_ALERT,
    PREDICTION_SET_LOAD_FAILED,
)
ALERT_NOTE_VARIABLE = "HMREC_ALERT_NOTE"
NOTE_FIELD = "note"
# Keys Cloud Logging reads from a JSON line: the level of the entry and its text.
SEVERITY_FIELD = "severity"
MESSAGE_FIELD = "message"
SEVERITY_INFO = "INFO"
SEVERITY_ERROR = "ERROR"

Scalar = str | int | float | bool


def json_line(severity: str, message: str, fields: Mapping[str, Scalar]) -> str:
    """One log line: a JSON object with `severity`, `message` and the given fields.

    Args:
        severity: "INFO" or "ERROR" (the level Cloud Logging gives the entry).
        message: Short text for a person.
        fields: More keys of the object; values are text, numbers or booleans.

    Returns:
        The object as ONE line of text, without a line break at its end.

    Raises:
        ValueError: If a field is named `severity` or `message` (it would
            replace the line's own key).
    """
    reserved = {SEVERITY_FIELD, MESSAGE_FIELD} & set(fields)
    if reserved:
        raise ValueError(f"a log field may not be named {sorted(reserved)}")
    content = {SEVERITY_FIELD: severity, MESSAGE_FIELD: message, **fields}
    # allow_nan=False: NaN is not JSON, and Cloud Logging would keep the line as plain text.
    return json.dumps(content, ensure_ascii=True, allow_nan=False)


def write_line(line: str, stream: TextIO | None = None) -> None:
    """Write one log line and flush, so that it is not lost if the process ends.

    Args:
        line: The line, without a line break.
        stream: Where to write; None means the standard output of the process.
    """
    target = sys.stdout if stream is None else stream
    # One write call for the line and its line break: two threads never mix their lines.
    target.write(line + "\n")
    target.flush()


def alert_line(
    kind: str, message: str, fields: Mapping[str, Scalar], environment: Mapping[str, str]
) -> str:
    """The line of one alert: severity ERROR, the fixed field with the kind, the fields.

    Args:
        kind: One of ALERT_KINDS.
        message: What happened, for the person who reads the alert.
        fields: Details, e.g. {"step": "O2_features", "run_id": "..."}.
        environment: The environment of the process; HMREC_ALERT_NOTE, if set
            and not empty, becomes the field `note`.

    Raises:
        ValueError: If the kind is unknown, or a field has a reserved name.
    """
    if kind not in ALERT_KINDS:
        raise ValueError(f"unknown alert kind {kind!r}; the kinds are {ALERT_KINDS}")
    reserved = {ALERT_FIELD, NOTE_FIELD} & set(fields)
    if reserved:
        raise ValueError(f"an alert field may not be named {sorted(reserved)}")
    content: dict[str, Scalar] = {ALERT_FIELD: kind, **fields}
    note = environment.get(ALERT_NOTE_VARIABLE, "")
    if note != "":
        content[NOTE_FIELD] = note
    return json_line(SEVERITY_ERROR, message, content)


def emit_alert(
    kind: str,
    message: str,
    fields: Mapping[str, Scalar],
    stream: TextIO | None = None,
) -> None:
    """Write the line of one alert to the output of this process.

    Args:
        kind: One of ALERT_KINDS.
        message: What happened.
        fields: Details of the event.
        stream: Where to write; None means the standard output.
    """
    write_line(alert_line(kind, message, fields, os.environ), stream)
