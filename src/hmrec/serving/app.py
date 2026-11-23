"""The HTTP API of the recommendation service (docs/mlops.md §9; M14).

    GET /v1/recommendations/{customer_id}
        200  {"articles": [12 article ids, best first],
              "prediction_set": {"anchor": "2020-09-23", "model_version": "4f76121744a7"},
              "source": "ranker" | "fallback"}
        503  no prediction set is loaded yet
    GET /healthz
        200  a prediction set is loaded (its id, its size, the image's commit)
        503  none is loaded

`customer_id` and the article ids are the ORIGINAL identifiers of the dataset.
"ranker": the customer is in the prediction set and gets the precomputed Top-12.
"fallback": anyone else gets the popularity list of R1 for the "age missing"
bucket of the same week (hmrec.serving_set). The model version is the first 12
characters of the sha256 of the Ranker file behind the set.

Logs: one JSON line per request with its route (the template, never the path),
status, latency, source and prediction-set id. **No customer id is written to a
log line**, and uvicorn's own access log, which prints the path, is switched off
where the server is started (Dockerfile.serving, Makefile).

Settings come from the environment, never from a config file (the image holds
none) and never from a number in this file:

    HMREC_SERVING_STATE_ROOT     the folder of the pipeline's state (read only)
    HMREC_SERVING_POLL_SECONDS   seconds between two looks at the `current` pointer
    HMREC_IMAGE_GIT_COMMIT       the commit the image was built from (set by the image)

Start it with uvicorn's factory option:
    uvicorn hmrec.serving.app:create_app_from_environment --factory --no-access-log
"""

import os
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from hmrec.serving.store import LOG_KIND_FIELD, PointerWatcher, PredictionStore
from hmrec.structured_logs import SEVERITY_INFO, json_line, write_line

STATE_ROOT_VARIABLE = "HMREC_SERVING_STATE_ROOT"
POLL_SECONDS_VARIABLE = "HMREC_SERVING_POLL_SECONDS"
IMAGE_COMMIT_VARIABLE = "HMREC_IMAGE_GIT_COMMIT"
NOT_AN_IMAGE = "not an image"
RECOMMENDATIONS_ROUTE = "/v1/recommendations/{customer_id}"
HEALTH_ROUTE = "/healthz"
UNMATCHED_ROUTE = "unmatched"
LOG_KIND_REQUEST = "request"
HTTP_OK = 200
HTTP_UNAVAILABLE = 503
MILLISECONDS_PER_SECOND = 1000.0
NOT_LOADED = {"detail": "no prediction set is loaded"}


def route_template(request: Request) -> str:
    """The route a request matched, as its template; "unmatched" for an unknown path.

    The template ("/v1/recommendations/{customer_id}") is logged instead of the
    path, so that no customer id reaches a log line.
    """
    route = request.scope.get("route")
    path = getattr(route, "path", None)
    return path if isinstance(path, str) else UNMATCHED_ROUTE


def create_app(state_root: Path, poll_seconds: float, commit: str) -> FastAPI:
    """Build the service.

    Args:
        state_root: The folder of the pipeline's state (read only).
        poll_seconds: Seconds between two looks at the `current` pointer.
        commit: The commit of the running code, shown by /healthz.

    Returns:
        The application. Its start-up loads the current prediction set and
        starts the pointer watcher; its shutdown stops the watcher.
    """
    store = PredictionStore(state_root)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        # Loaded BEFORE the server accepts a request: the first request after a cold
        # start is answered from the set, not with 503. A failed load is logged by
        # the store; the watcher tries again.
        store.refresh()
        watcher = PointerWatcher(store, poll_seconds)
        watcher.start()
        yield
        watcher.stop()

    # No interactive documentation pages: the service has two routes and no public use.
    application = FastAPI(
        title="hmrec recommendations",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    application.state.store = store

    @application.middleware("http")
    async def log_request(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        started = time.perf_counter()
        response = await call_next(request)
        latency_ms = (time.perf_counter() - started) * MILLISECONDS_PER_SECOND
        fields = {
            LOG_KIND_FIELD: LOG_KIND_REQUEST,
            "route": route_template(request),
            "status": response.status_code,
            "latency_ms": round(latency_ms, 3),
            # Set by the route below; empty for a request that got no recommendation.
            "source": getattr(request.state, "source", ""),
            "prediction_set": getattr(request.state, "prediction_set", ""),
        }
        write_line(json_line(SEVERITY_INFO, LOG_KIND_REQUEST, fields))
        return response

    @application.get(RECOMMENDATIONS_ROUTE)
    async def recommendations(customer_id: str, request: Request) -> JSONResponse:
        loaded = store.current
        if loaded is None:
            return JSONResponse(NOT_LOADED, status_code=HTTP_UNAVAILABLE)
        articles, source = loaded.serving.recommend(customer_id)
        record = loaded.serving.record
        request.state.source = source
        request.state.prediction_set = record.set_id
        content = {
            "articles": list(articles),
            "prediction_set": {
                "anchor": record.description.anchor,
                "model_version": record.model_version,
            },
            "source": source,
        }
        return JSONResponse(content, status_code=HTTP_OK)

    @application.get(HEALTH_ROUTE)
    async def healthz(request: Request) -> JSONResponse:
        loaded = store.current
        if loaded is None:
            return JSONResponse({**NOT_LOADED, "commit": commit}, status_code=HTTP_UNAVAILABLE)
        record = loaded.serving.record
        request.state.prediction_set = record.set_id
        content = {
            "status": "ok",
            "prediction_set": {
                "anchor": record.description.anchor,
                "model_version": record.model_version,
            },
            "n_customers": record.n_customers,
            "commit": commit,
        }
        return JSONResponse(content, status_code=HTTP_OK)

    return application


def settings_from_environment(environment: Mapping[str, str]) -> tuple[Path, float, str]:
    """The state root, the poll interval and the commit from the environment.

    Raises:
        ValueError: If the state root is missing or not an absolute path, or
            the poll interval is missing or not a positive number.
    """
    state_value = environment.get(STATE_ROOT_VARIABLE, "")
    if state_value == "" or not Path(state_value).is_absolute():
        raise ValueError(f"{STATE_ROOT_VARIABLE} must be an absolute path, got {state_value!r}")
    poll_value = environment.get(POLL_SECONDS_VARIABLE, "")
    try:
        poll_seconds = float(poll_value)
    except ValueError as error:
        raise ValueError(
            f"{POLL_SECONDS_VARIABLE} must be a number of seconds, got {poll_value!r}"
        ) from error
    if poll_seconds <= 0:
        raise ValueError(f"{POLL_SECONDS_VARIABLE} must be > 0, got {poll_value!r}")
    commit = environment.get(IMAGE_COMMIT_VARIABLE, NOT_AN_IMAGE)
    return Path(state_value), poll_seconds, commit


def create_app_from_environment() -> FastAPI:
    """The application with the settings of the process's environment (uvicorn --factory)."""
    state_root, poll_seconds, commit = settings_from_environment(os.environ)
    return create_app(state_root, poll_seconds, commit)
