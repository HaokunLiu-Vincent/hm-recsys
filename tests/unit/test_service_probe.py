"""Tests for hmrec.service_probe and hmrec.service_check (the measurement of the service).

Percentiles are nearest-rank: of the 20 values 1 .. 20 the median is the 10th
value (10) and the 95th percentile the 19th (19); of 3 values the 95th
percentile is the 3rd.
"""

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import polars as pl
import pytest

from hmrec.service_check import expected_fallback, expected_known, unknown_customer_ids
from hmrec.service_probe import log_latencies, percentile, probe, summarize


def test_percentiles_are_nearest_rank() -> None:
    values = [float(value) for value in range(20, 0, -1)]  # unsorted on purpose
    assert percentile(values, 0.5) == 10.0 and percentile(values, 0.95) == 19.0
    assert percentile(values, 1.0) == 20.0 and percentile([7.0], 0.95) == 7.0
    assert percentile([1.0, 2.0, 3.0], 0.95) == 3.0
    summary = summarize(values)
    assert (summary.n, summary.p50_ms, summary.p95_ms) == (20, 10.0, 19.0)
    assert (summary.min_ms, summary.max_ms, summary.mean_ms) == (1.0, 20.0, 10.5)
    with pytest.raises(ValueError, match="at least one value"):
        summarize([])
    with pytest.raises(ValueError, match="share must be in"):
        percentile(values, 0.0)


class _Handler(BaseHTTPRequestHandler):
    """A stand-in for the service: one known customer, a fallback for everyone else."""

    protocol_version = "HTTP/1.1"
    seen: list[tuple[str, str | None]] = []

    def do_GET(self) -> None:
        type(self).seen.append((self.path, self.headers.get("Authorization")))
        customer = self.path.rsplit("/", 1)[-1]
        known = customer == "abc"
        body = json.dumps(
            {
                "articles": ["0000000001", "0000000002"] if known else ["0000000009", "0000000008"],
                "prediction_set": {"anchor": "2020-09-23", "model_version": "4f76121744a7"},
                "source": "ranker" if known else "fallback",
            }
        ).encode()
        status = 500 if customer == "broken" else 200
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        """Keep the test output quiet."""


@pytest.fixture
def service_url() -> Iterator[str]:
    _Handler.seen = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def test_probe_asks_in_order_over_one_connection_with_the_token(service_url: str) -> None:
    results = probe(service_url, ["abc", "who/else", "broken"], "secret-token", 10.0)
    assert [result.position for result in results] == [1, 2, 3]
    assert [(result.status, result.source) for result in results] == [
        (200, "ranker"),
        (200, "fallback"),
        (500, ""),
    ]
    assert results[0].articles == ("0000000001", "0000000002")
    assert results[0].prediction_set == "2020-09-23/4f76121744a7"
    assert results[2].articles == () and results[2].prediction_set == ""
    assert all(result.latency_ms > 0 for result in results)
    # An id is one path segment whatever it holds, and every request carries the token.
    assert [path for path, _ in _Handler.seen] == [
        "/v1/recommendations/abc",
        "/v1/recommendations/who%2Felse",
        "/v1/recommendations/broken",
    ]
    assert {header for _, header in _Handler.seen} == {"Bearer secret-token"}
    with pytest.raises(ValueError, match="not a URL with a host"):
        probe("no-url", ["abc"], "", 1.0)


def test_the_services_own_latencies_are_read_from_its_request_lines() -> None:
    route = "/v1/recommendations/{customer_id}"
    entries = [
        {"jsonPayload": {"hmrec_log": "request", "route": route, "latency_ms": 0.25}},
        {"jsonPayload": {"hmrec_log": "request", "route": "/healthz", "latency_ms": 9.0}},
        {"jsonPayload": {"hmrec_log": "prediction_set_loaded"}},
        {"textPayload": "INFO: Started server process"},
        {"jsonPayload": {"hmrec_log": "request", "route": route, "latency_ms": 0.5}},
    ]
    assert log_latencies(entries) == [0.25, 0.5]


def test_expected_answers_come_from_the_prediction_set_and_the_score_table() -> None:
    predictions = pl.DataFrame(
        {
            "customer_idx": [0, 0, 1, 1, 2, 2],
            "rank": [1, 2, 1, 2, 1, 2],
            "article_idx": [3, 1, 0, 2, 1, 3],
        },
        schema={"customer_idx": pl.Int32, "rank": pl.Int32, "article_idx": pl.Int32},
    )
    customers = pl.DataFrame({"customer_idx": [0, 1, 2], "customer_id": ["aa", "bb", "cc"]})
    customers = customers.cast({"customer_idx": pl.Int32})
    articles = pl.DataFrame(
        {"article_idx": [0, 1, 2, 3], "article_id": ["0100", "0101", "0102", "0103"]}
    ).cast({"article_idx": pl.Int32})
    known = expected_known(predictions, customers, articles, n_customers=5, seed=1)
    assert known == {"aa": ["0103", "0101"], "bb": ["0100", "0102"], "cc": ["0101", "0103"]}
    assert len(expected_known(predictions, customers, articles, n_customers=2, seed=1)) == 2
    with pytest.raises(ValueError, match="no customer id or no article id"):
        expected_known(predictions, customers.head(2), articles, n_customers=5, seed=1)
    # Bucket AGE_UNKNOWN: scores 5 (article 2), 5 (article 1), 3 (article 0); the tie goes
    # to the smaller article_idx, so the best two are articles 1 and 2.
    scores = pl.DataFrame(
        {
            "age_bucket": ["AGE_UNKNOWN", "AGE_UNKNOWN", "AGE_UNKNOWN", "AGE_18_24"],
            "article_idx": [2, 1, 0, 3],
            "score": [5.0, 5.0, 3.0, 99.0],
        }
    ).cast({"article_idx": pl.Int32, "score": pl.Float32})
    assert expected_fallback(scores, articles, "AGE_UNKNOWN", top_k=2) == ["0101", "0102"]
    with pytest.raises(ValueError, match="has no 2 known articles"):
        expected_fallback(scores, articles, "AGE_18_24", top_k=2)


def test_unknown_ids_are_not_in_the_id_map() -> None:
    taken = pl.DataFrame({"customer_id": [f"{0:064x}", f"{2:064x}"]})
    assert unknown_customer_ids(taken, 3) == [f"{1:064x}", f"{3:064x}", f"{4:064x}"]
