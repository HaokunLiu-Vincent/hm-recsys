"""Tests for the recommendation service (hmrec.serving): the API, the pointer, the logs.

A small synthetic state (serving_example.py): two published prediction sets,

    2020-09-16   ALICE (0, 1)   BOB (2, 3)   CAROL (4, 0)
    2020-09-23   ALICE (3, 4)   BOB (2, 3)                  (CAROL is not in it)

with the fallback list (0000000103, 0000000101) in both. The application is
called directly as an ASGI application; no server is started.
"""

import asyncio
import json
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

import pytest
from serving_example import (
    ALICE,
    ARTICLES,
    BOB,
    CAROL,
    FALLBACK_IDS,
    TOP,
    UNKNOWN,
    get,
    get_now,
    pointer_text,
    write_pointer,
    write_state_set,
)

from hmrec.serving.app import (
    IMAGE_COMMIT_VARIABLE,
    POLL_SECONDS_VARIABLE,
    STATE_ROOT_VARIABLE,
    create_app,
    create_app_from_environment,
    settings_from_environment,
)
from hmrec.serving.store import FAILED, LOADED, UNCHANGED, PointerWatcher, PredictionStore
from hmrec.serving_set import TOP_ARTICLES_FILE
from hmrec.structured_logs import ALERT_FIELD, PREDICTION_SET_LOAD_FAILED

OLD, NEW = "2020-09-16", "2020-09-23"
NEW_TOP = {ALICE: (3, 4), BOB: (2, 3)}
ROUTE = "/v1/recommendations/"


@pytest.fixture
def state(tmp_path: Path) -> Path:
    """A state folder with the two prediction sets; `current` names the newer one."""
    root = tmp_path / "pipeline"
    write_state_set(root, OLD, TOP)
    write_state_set(root, NEW, NEW_TOP)
    write_pointer(root, NEW)
    return root


def _app(state: Path) -> Any:
    """The application with its start-up load done (the watcher thread is not started)."""
    app = create_app(state, poll_seconds=60.0, commit="a" * 40)
    assert app.state.store.refresh() == LOADED
    return app


def _log_lines(capsys: pytest.CaptureFixture[str]) -> list[dict[str, Any]]:
    return [json.loads(line) for line in capsys.readouterr().out.splitlines()]


# ---------------------------------------------------------------- the two answers


def test_a_known_customer_gets_the_precomputed_top_k_in_original_ids(state: Path) -> None:
    app = _app(state)
    status, body = get_now(app, ROUTE + ALICE)
    assert status == 200
    record = app.state.store.current.serving.record
    assert body == {
        "articles": [ARTICLES[3], ARTICLES[4]],
        "prediction_set": {"anchor": NEW, "model_version": record.model_version},
        "source": "ranker",
    }
    assert len(record.model_version) == 12
    assert get_now(app, ROUTE + BOB)[1]["articles"] == [ARTICLES[2], ARTICLES[3]]


def test_an_unknown_customer_gets_the_fallback_list(state: Path) -> None:
    app = _app(state)
    # CAROL is in the older set only: for the current set she is unknown.
    for customer in (UNKNOWN, CAROL, "not-a-customer-id"):
        status, body = get_now(app, ROUTE + customer)
        assert status == 200
        assert body["articles"] == list(FALLBACK_IDS) and body["source"] == "fallback"
        assert body["prediction_set"]["anchor"] == NEW


def test_healthz_says_which_set_is_loaded(state: Path) -> None:
    app = _app(state)
    status, body = get_now(app, "/healthz")
    assert status == 200
    assert body["status"] == "ok" and body["prediction_set"]["anchor"] == NEW
    assert body["n_customers"] == 2 and body["commit"] == "a" * 40
    # The same answer under /v1/health: Cloud Run's front end keeps /healthz for itself.
    assert get_now(app, "/v1/health") == (status, body)
    assert get_now(app, "/v1/unknown-route")[0] == 404


# ---------------------------------------------------------------- the pointer


def test_a_changed_pointer_switches_the_set_without_a_restart(state: Path) -> None:
    app = _app(state)
    store: PredictionStore = app.state.store
    assert store.refresh() == UNCHANGED
    assert get_now(app, ROUTE + CAROL)[1]["source"] == "fallback"
    write_pointer(state, OLD)  # a rollback
    assert store.refresh() == LOADED
    status, body = get_now(app, ROUTE + CAROL)
    assert body["source"] == "ranker" and body["articles"] == [ARTICLES[4], ARTICLES[0]]
    assert body["prediction_set"]["anchor"] == OLD
    assert get_now(app, ROUTE + ALICE)[1]["articles"] == [ARTICLES[0], ARTICLES[1]]
    write_pointer(state, NEW)  # and forward again
    assert store.refresh() == LOADED
    assert get_now(app, ROUTE + ALICE)[1]["articles"] == [ARTICLES[3], ARTICLES[4]]


def test_a_broken_set_leaves_the_old_one_in_place_and_is_logged_as_an_error(
    state: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    app = _app(state)
    store: PredictionStore = app.state.store
    capsys.readouterr()
    # The older set's Top-K file is damaged (a wrong checksum); the pointer moves to it.
    damaged = state / "predictions" / OLD / "serving" / TOP_ARTICLES_FILE
    # CAROL's second article is position 0; the last int32 of the file becomes 1.
    damaged.write_bytes(damaged.read_bytes()[:-4] + b"\x01\x00\x00\x00")
    write_pointer(state, OLD)
    assert store.refresh() == FAILED
    status, body = get_now(app, ROUTE + ALICE)
    assert status == 200 and body["prediction_set"]["anchor"] == NEW  # still the old set
    assert body["articles"] == [ARTICLES[3], ARTICLES[4]]
    alerts = [line for line in _log_lines(capsys) if ALERT_FIELD in line]
    assert len(alerts) == 1
    alert = alerts[0]
    assert alert["severity"] == "ERROR" and alert[ALERT_FIELD] == PREDICTION_SET_LOAD_FAILED
    assert alert["prediction_set"] == f"predictions/{OLD}" and "sha256" in alert["reason"]
    assert alert["still_serving"].startswith(NEW)
    # The load is tried again at the next poll, but the same failure is not logged twice.
    assert store.refresh() == FAILED
    assert [line for line in _log_lines(capsys) if ALERT_FIELD in line] == []
    # Pointing back at the good set works; nothing was lost.
    write_pointer(state, NEW)
    assert store.refresh() == UNCHANGED


@pytest.mark.parametrize(
    ("pointer", "message"),
    [
        ("not json", "not a pointer"),
        (json.dumps({"anchor": NEW}), "not a pointer"),
        (json.dumps({"anchor": NEW, "prediction_set": "../outside"}), "not a folder of the state"),
        (json.dumps({"anchor": NEW, "prediction_set": "/etc"}), "not a folder of the state"),
        (json.dumps({"anchor": NEW, "prediction_set": "predictions/2020-01-01"}), "is missing"),
        # The folder exists but holds the set of another week.
        (
            json.dumps({"anchor": "2020-09-30", "prediction_set": f"predictions/{NEW}"}),
            "pointer says",
        ),
    ],
)
def test_a_pointer_that_names_no_loadable_set_is_refused(
    state: Path, capsys: pytest.CaptureFixture[str], pointer: str, message: str
) -> None:
    app = _app(state)
    capsys.readouterr()
    (state / "state" / "current.json").write_text(pointer, encoding="utf-8")
    assert app.state.store.refresh() == FAILED
    alerts = [line for line in _log_lines(capsys) if ALERT_FIELD in line]
    assert len(alerts) == 1 and message in alerts[0]["reason"]
    assert get_now(app, ROUTE + ALICE)[1]["prediction_set"]["anchor"] == NEW


def test_without_a_loadable_set_at_start_up_the_service_answers_503(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "pipeline"
    app = create_app(root, poll_seconds=60.0, commit="a" * 40)
    assert app.state.store.refresh() == FAILED  # there is no pointer at all
    assert get_now(app, ROUTE + ALICE) == (503, {"detail": "no prediction set is loaded"})
    assert get_now(app, "/healthz")[0] == 503 and get_now(app, "/v1/health")[0] == 503
    alerts = [line for line in _log_lines(capsys) if ALERT_FIELD in line]
    assert len(alerts) == 1 and alerts[0]["still_serving"] == ""
    # A set that appears later is loaded by the next poll.
    write_state_set(root, NEW, NEW_TOP)
    write_pointer(root, NEW)
    assert app.state.store.refresh() == LOADED
    assert get_now(app, ROUTE + ALICE)[0] == 200


# ---------------------------------------------------------------- start-up and the timer


def test_start_up_loads_the_set_and_the_timer_follows_the_pointer(state: Path) -> None:
    app = create_app(state, poll_seconds=0.02, commit="a" * 40)

    async def scenario() -> tuple[str, str]:
        # The message type of ASGI is a mutable mapping (starlette.types.Message).
        to_app: asyncio.Queue[MutableMapping[str, Any]] = asyncio.Queue()
        from_app: asyncio.Queue[MutableMapping[str, Any]] = asyncio.Queue()
        lifespan = asyncio.create_task(
            app({"type": "lifespan", "asgi": {"version": "3.0"}}, to_app.get, from_app.put)
        )
        await to_app.put({"type": "lifespan.startup"})
        assert (await from_app.get())["type"] == "lifespan.startup.complete"
        # Loaded during start-up: the first request is answered from the set.
        _, first = await get(app, ROUTE + ALICE)
        write_pointer(state, OLD)
        anchor = first["prediction_set"]["anchor"]
        for _ in range(250):  # at most 5 seconds; the timer ticks every 20 ms
            await asyncio.sleep(0.02)
            _, body = await get(app, ROUTE + ALICE)
            anchor = body["prediction_set"]["anchor"]
            if anchor == OLD:
                break
        await to_app.put({"type": "lifespan.shutdown"})
        assert (await from_app.get())["type"] == "lifespan.shutdown.complete"
        await lifespan
        return first["prediction_set"]["anchor"], anchor

    assert asyncio.run(scenario()) == (NEW, OLD)


def test_the_watcher_needs_a_positive_interval(state: Path) -> None:
    with pytest.raises(ValueError, match="poll_seconds must be > 0"):
        PointerWatcher(PredictionStore(state), 0.0)


# ---------------------------------------------------------------- logs


def test_one_log_line_per_request_and_no_customer_id_in_any_line(
    state: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    app = _app(state)
    capsys.readouterr()
    get_now(app, ROUTE + ALICE)
    get_now(app, ROUTE + UNKNOWN)
    get_now(app, "/healthz")
    get_now(app, "/no/such/" + BOB)
    output = capsys.readouterr().out
    lines = [json.loads(line) for line in output.splitlines()]
    assert [line["hmrec_log"] for line in lines] == ["request"] * 4
    set_id = app.state.store.current.serving.record.set_id
    template = "/v1/recommendations/{customer_id}"
    assert [(line["route"], line["status"], line["source"]) for line in lines] == [
        (template, 200, "ranker"),
        (template, 200, "fallback"),
        ("/healthz", 200, ""),
        ("unmatched", 404, ""),
    ]
    assert [line["prediction_set"] for line in lines] == [set_id, set_id, set_id, ""]
    assert all(isinstance(line["latency_ms"], float) and line["latency_ms"] >= 0 for line in lines)
    # No customer id, known or unknown, in a route or anywhere else of the output.
    for customer in (ALICE, UNKNOWN, BOB):
        assert customer not in output


def test_the_load_log_and_the_alert_hold_no_customer_id(
    state: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _app(state)
    lines = _log_lines(capsys)
    assert [line["hmrec_log"] for line in lines] == ["prediction_set_loaded"]
    assert lines[0]["n_customers"] == 2 and lines[0]["previous_prediction_set"] == ""


# ---------------------------------------------------------------- settings


def test_settings_come_from_the_environment(state: Path) -> None:
    environment = {STATE_ROOT_VARIABLE: str(state), POLL_SECONDS_VARIABLE: "30"}
    assert settings_from_environment(environment) == (state, 30.0, "not an image")
    commit = {**environment, IMAGE_COMMIT_VARIABLE: "b" * 40}
    assert settings_from_environment(commit)[2] == "b" * 40
    for broken, message in [
        ({POLL_SECONDS_VARIABLE: "30"}, STATE_ROOT_VARIABLE),
        ({**environment, STATE_ROOT_VARIABLE: "relative/state"}, "absolute path"),
        ({STATE_ROOT_VARIABLE: str(state)}, POLL_SECONDS_VARIABLE),
        ({**environment, POLL_SECONDS_VARIABLE: "soon"}, "number of seconds"),
        ({**environment, POLL_SECONDS_VARIABLE: "0"}, "must be > 0"),
    ]:
        with pytest.raises(ValueError, match=message):
            settings_from_environment(broken)


def test_the_factory_reads_the_process_environment(
    state: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(STATE_ROOT_VARIABLE, str(state))
    monkeypatch.setenv(POLL_SECONDS_VARIABLE, "60")
    monkeypatch.delenv(IMAGE_COMMIT_VARIABLE, raising=False)
    app = create_app_from_environment()
    assert app.state.store.refresh() == LOADED
    assert get_now(app, "/healthz")[1]["commit"] == "not an image"
    assert pointer_text(NEW) == (state / "state" / "current.json").read_text(encoding="utf-8")
