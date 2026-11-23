"""Measure the recommendation service with synthetic requests (docs/mlops.md §9; M14).

    probe          send GET /v1/recommendations/{id} for a list of customer ids over
                   ONE kept-alive connection and time every request on this machine.
    summarize      n, p50, p95, minimum, maximum and mean of a list of latencies.
    log_latencies  the latencies the SERVICE measured for the same requests, read
                   from its own request lines (hmrec.serving.app), as Cloud Logging
                   returns them.

Two views of one request, on purpose: the service's line holds the time between
receiving the request and sending the answer; the client's time adds the network
between this machine and the region, which is most of it for a lookup service.

Percentiles are nearest-rank: the p-th percentile of n sorted values is the value
at position ceil(p x n), counted from 1. No value is interpolated.

The token (an identity token for a private Cloud Run service) is passed in by
the caller and is never written anywhere; no customer id is written to a table
either: a request is recorded by its position and its kind.

This module is at the bottom layer and imports only the standard library.
"""

import http.client
import json
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, urlsplit

RECOMMENDATIONS_PATH = "/v1/recommendations/"
MILLISECONDS_PER_SECOND = 1000.0
REQUEST_LOG_KIND = "request"
RECOMMENDATIONS_ROUTE = "/v1/recommendations/{customer_id}"
HTTP_OK = 200


@dataclass(frozen=True)
class ProbeResult:
    """One request as this machine saw it.

    Attributes:
        position: 1 for the first request of the series, 2 for the second, ...
        status: The HTTP status code.
        latency_ms: Time from sending the request to having read the whole answer.
        source: "ranker" or "fallback" ("" if the answer was not a recommendation).
        prediction_set: "{anchor}/{model_version}" of the answer ("" likewise).
        articles: The article ids of the answer.
    """

    position: int
    status: int
    latency_ms: float
    source: str
    prediction_set: str
    articles: tuple[str, ...]


@dataclass(frozen=True)
class LatencySummary:
    """Summary of a list of latencies in milliseconds (float64).

    Attributes:
        n: Number of values.
        p50_ms: Median (nearest rank).
        p95_ms: 95th percentile (nearest rank).
        min_ms: Smallest value.
        max_ms: Largest value.
        mean_ms: Mean.
    """

    n: int
    p50_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    mean_ms: float


def percentile(values: Sequence[float], share: float) -> float:
    """The nearest-rank percentile: the value at position ceil(share x n) of the sorted values.

    Args:
        values: At least one number.
        share: The percentile as a share in (0, 1], e.g. 0.95.

    Raises:
        ValueError: If there is no value or the share is outside (0, 1].
    """
    if len(values) == 0:
        raise ValueError("a percentile needs at least one value")
    if not 0.0 < share <= 1.0:
        raise ValueError(f"the share must be in (0, 1], got {share}")
    ordered = sorted(values)
    return ordered[math.ceil(share * len(ordered)) - 1]


def summarize(latencies_ms: Sequence[float]) -> LatencySummary:
    """n, p50, p95, minimum, maximum and mean of some latencies.

    Raises:
        ValueError: If the list is empty.
    """
    return LatencySummary(
        n=len(latencies_ms),
        p50_ms=percentile(latencies_ms, 0.5),
        p95_ms=percentile(latencies_ms, 0.95),
        min_ms=min(latencies_ms),
        max_ms=max(latencies_ms),
        mean_ms=math.fsum(latencies_ms) / len(latencies_ms),
    )


def open_connection(base_url: str, timeout_seconds: float) -> http.client.HTTPConnection:
    """A connection to the service (HTTPS unless the URL says http).

    Raises:
        ValueError: If the URL has no host.
    """
    parts = urlsplit(base_url)
    if parts.hostname is None:
        raise ValueError(f"{base_url!r} is not a URL with a host")
    if parts.scheme == "http":
        return http.client.HTTPConnection(parts.hostname, parts.port, timeout=timeout_seconds)
    return http.client.HTTPSConnection(parts.hostname, parts.port, timeout=timeout_seconds)


def get_json(
    connection: http.client.HTTPConnection, path: str, token: str
) -> tuple[int, Any, float]:
    """One GET request on an open connection.

    Returns:
        (status code, the parsed JSON body or None if it is not JSON, milliseconds
        from sending the request to having read the whole answer).
    """
    headers = {"Authorization": f"Bearer {token}"} if token != "" else {}
    started = time.perf_counter()
    connection.request("GET", path, headers=headers)
    response = connection.getresponse()
    body = response.read()
    latency_ms = (time.perf_counter() - started) * MILLISECONDS_PER_SECOND
    try:
        content = json.loads(body)
    except ValueError:
        content = None
    return response.status, content, latency_ms


def probe(
    base_url: str, customer_ids: Sequence[str], token: str, timeout_seconds: float
) -> list[ProbeResult]:
    """Ask for the recommendations of every customer id, in order, over one connection.

    The first request includes opening the connection (and, if the service was
    scaled to zero, starting an instance); every later one reuses it.

    Args:
        base_url: The service's URL, e.g. "https://...run.app".
        customer_ids: The ids to ask for.
        token: An identity token ("" for a service that needs none).
        timeout_seconds: Time limit of one request.
    """
    connection = open_connection(base_url, timeout_seconds)
    results: list[ProbeResult] = []
    try:
        for position, customer_id in enumerate(customer_ids, start=1):
            path = RECOMMENDATIONS_PATH + quote(customer_id, safe="")
            status, content, latency_ms = get_json(connection, path, token)
            answer = content if isinstance(content, dict) and status == HTTP_OK else {}
            set_id = answer.get("prediction_set", {})
            results.append(
                ProbeResult(
                    position=position,
                    status=status,
                    latency_ms=latency_ms,
                    source=answer.get("source", ""),
                    prediction_set=(
                        f"{set_id['anchor']}/{set_id['model_version']}" if set_id else ""
                    ),
                    articles=tuple(answer.get("articles", ())),
                )
            )
    finally:
        connection.close()
    return results


def log_latencies(entries: Sequence[Any]) -> list[float]:
    """The latencies of recommendation requests in log entries of the service.

    Args:
        entries: Log entries as Cloud Logging returns them in JSON: objects with a
            `jsonPayload`. Entries that are not request lines of the
            recommendations route are skipped.

    Returns:
        `latency_ms` of every recommendation request, in the order of the entries.
    """
    latencies: list[float] = []
    for entry in entries:
        payload = entry.get("jsonPayload") if isinstance(entry, dict) else None
        if not isinstance(payload, dict):
            continue
        is_request = payload.get("hmrec_log") == REQUEST_LOG_KIND
        if is_request and payload.get("route") == RECOMMENDATIONS_ROUTE:
            latencies.append(float(payload["latency_ms"]))
    return latencies
