"""Tests for hmrec.structured_logs: one JSON object per line, the fixed alert field."""

import io
import json

import pytest

from hmrec.structured_logs import (
    ALERT_FIELD,
    ALERT_KINDS,
    ALERT_NOTE_VARIABLE,
    QUALITY_ALERT,
    STEP_FAILED,
    alert_line,
    emit_alert,
    json_line,
    write_line,
)


def test_a_log_line_is_one_json_object_on_one_line() -> None:
    line = json_line("INFO", "request", {"status": 200, "latency_ms": 0.5, "source": "ranker"})
    assert "\n" not in line
    assert json.loads(line) == {
        "severity": "INFO",
        "message": "request",
        "status": 200,
        "latency_ms": 0.5,
        "source": "ranker",
    }
    # A text with a line break stays ONE line: the break is escaped inside the JSON text.
    assert "\n" not in json_line("ERROR", "first line\nsecond line", {})
    with pytest.raises(ValueError, match="may not be named"):
        json_line("INFO", "request", {"severity": "DEBUG"})
    # NaN is not JSON; Cloud Logging would keep such a line as plain text.
    with pytest.raises(ValueError):
        json_line("INFO", "request", {"latency_ms": float("nan")})


def test_an_alert_line_holds_the_fixed_field_with_its_kind() -> None:
    line = alert_line(STEP_FAILED, "tick stopped", {"step": "O2_features"}, environment={})
    assert json.loads(line) == {
        "severity": "ERROR",
        "message": "tick stopped",
        ALERT_FIELD: STEP_FAILED,
        "step": "O2_features",
    }
    # The four kinds the alert policies of infra/setup.sh match.
    assert ALERT_KINDS == (
        "step_failed",
        "input_check_failed",
        "quality_alert",
        "prediction_set_load_failed",
    )
    assert ALERT_FIELD == "hmrec_alert"
    with pytest.raises(ValueError, match="unknown alert kind"):
        alert_line("something_else", "x", {}, environment={})
    with pytest.raises(ValueError, match="may not be named"):
        alert_line(STEP_FAILED, "x", {ALERT_FIELD: QUALITY_ALERT}, environment={})


def test_a_drill_note_from_the_environment_is_added_to_every_alert_line() -> None:
    environment = {ALERT_NOTE_VARIABLE: "provoked on purpose"}
    line = alert_line(QUALITY_ALERT, "flag", {"rule": "recall/union"}, environment)
    assert json.loads(line)["note"] == "provoked on purpose"
    # An empty note adds no field.
    assert "note" not in json.loads(
        alert_line(QUALITY_ALERT, "flag", {}, {ALERT_NOTE_VARIABLE: ""})
    )


def test_emit_alert_writes_exactly_one_line(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ALERT_NOTE_VARIABLE, raising=False)
    stream = io.StringIO()
    emit_alert(STEP_FAILED, "tick stopped", {"step": "C1_ingest"}, stream)
    write_line(json_line("INFO", "next", {}), stream)
    lines = stream.getvalue().splitlines()
    assert len(lines) == 2 and stream.getvalue().endswith("\n")
    assert json.loads(lines[0])[ALERT_FIELD] == STEP_FAILED
    assert ALERT_FIELD not in json.loads(lines[1])
