"""CLI for M14: measure the recommendation service with synthetic requests (mlops §9).

    expect       what the service should answer, derived from the prediction set
                 and R1's score table of the same tick (hmrec.service_check):
                     uv run python scripts/m14_probe_service.py expect \
                         --predictions <top.parquet> --bucket-scores <r1_bucket_scores.parquet> \
                         --known 1500 --unknown 500 --out <expectations.json>
    requests     send the requests of an expectations file, time them on this
                 machine, compare every answer with the expectation:
                     HMREC_SERVICE_TOKEN=... uv run python scripts/m14_probe_service.py requests \
                         --url https://... --expected <expectations.json> --series warm-1
    server-log   the latencies the service itself logged for a series, from log
                 entries saved as JSON (`gcloud logging read ... --format json`):
                     uv run python scripts/m14_probe_service.py server-log \
                         --log-json <entries.json> --series warm-1

Tables (appended, never edited): results/m14_serving/probe_requests.csv (one row
per request: its position and kind, never a customer id) and probe_summary.csv
(one row per series and side). The identity token is read from the environment
variable HMREC_SERVICE_TOKEN, so it is in no command line.
"""

# Thread setup. This MUST run before polars (or any hmrec module that imports
# polars) is imported: polars reads POLARS_MAX_THREADS only once, at import time.
from hmrec.threads import configure_polars_threads

configure_polars_threads()

import argparse  # noqa: E402  (after the thread setup above, see comment)
import json  # noqa: E402
import os  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import polars as pl  # noqa: E402

from hmrec.config import load_preprocess_config  # noqa: E402
from hmrec.csv_records import append_row  # noqa: E402
from hmrec.data.metadata import scan_processed_parquet  # noqa: E402
from hmrec.git_state import git_state  # noqa: E402
from hmrec.logging_utils import get_logger, setup_logging  # noqa: E402
from hmrec.paths import ProjectPaths, get_paths  # noqa: E402
from hmrec.service_check import (  # noqa: E402
    expected_fallback,
    expected_known,
    unknown_customer_ids,
)
from hmrec.service_probe import (  # noqa: E402
    LatencySummary,
    log_latencies,
    probe,
    summarize,
)

LOGGER = get_logger("scripts.m14_probe_service")
TOKEN_VARIABLE = "HMREC_SERVICE_TOKEN"
KNOWN, UNKNOWN = "known", "unknown"
SIDE_CLIENT, SIDE_SERVER = "client", "server"
REQUEST_COLUMNS: tuple[str, ...] = (
    "series",
    "position",
    "kind",
    "status",
    "latency_ms",
    "source",
    "prediction_set",
    "as_expected",
)
SUMMARY_COLUMNS: tuple[str, ...] = (
    "series",
    "side",
    "n_requests",
    "p50_ms",
    "p95_ms",
    "min_ms",
    "max_ms",
    "mean_ms",
    "first_request_ms",
    "n_known",
    "n_unknown",
    "n_not_as_expected",
    "note",
    "recorded_at",
    "git_commit",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Measure the recommendation service.")
    commands = parser.add_subparsers(dest="command", required=True)
    expect = commands.add_parser("expect", help="derive the expected answers")
    expect.add_argument("--predictions", required=True, type=Path, help="top.parquet of the set")
    expect.add_argument("--bucket-scores", required=True, type=Path, help="R1's score table")
    expect.add_argument("--known", required=True, type=int, help="known customers to draw")
    expect.add_argument("--unknown", required=True, type=int, help="unknown customer ids")
    expect.add_argument("--seed", required=True, type=int)
    expect.add_argument("--out", required=True, type=Path)
    requests = commands.add_parser("requests", help="send the requests and time them")
    requests.add_argument("--url", required=True)
    requests.add_argument("--expected", required=True, type=Path)
    requests.add_argument("--series", required=True, help="a name for this series of requests")
    requests.add_argument("--limit", type=int, default=None, help="send only the first N")
    requests.add_argument("--timeout-seconds", type=float, default=120.0)
    requests.add_argument("--note", default="")
    server = commands.add_parser("server-log", help="summarize the service's own latencies")
    server.add_argument("--log-json", required=True, type=Path)
    server.add_argument("--series", required=True)
    server.add_argument("--note", default="")
    return parser.parse_args()


def write_expectations(args: argparse.Namespace, paths: ProjectPaths) -> None:
    """Derive the expected answers and write them to a JSON file (outside the repository)."""
    preprocess = load_preprocess_config(paths.preprocess_config_file)
    metadata = paths.processed_metadata_file
    customer_map = scan_processed_parquet(paths.customer_id_map_file, metadata, preprocess)
    article_map = scan_processed_parquet(paths.article_id_map_file, metadata, preprocess)
    customers, articles = customer_map.collect(), article_map.collect()
    predictions = pl.read_parquet(args.predictions)
    top_k = int(predictions["rank"].to_numpy().max())
    known = expected_known(predictions, customers, articles, args.known, args.seed)
    fallback = expected_fallback(
        pl.read_parquet(args.bucket_scores), articles, preprocess.unknown_age_bucket, top_k
    )
    # Known and unknown ids alternate in the order of the requests, as far as both last.
    unknown = unknown_customer_ids(customers, args.unknown)
    content = {"known": known, "unknown": unknown, "fallback": fallback}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(content) + "\n", encoding="utf-8")
    LOGGER.info(
        "%d known and %d unknown customers; expected fallback list: %s -> %s",
        len(known),
        len(unknown),
        " ".join(fallback),
        args.out,
    )


def summary_row(
    series: str,
    side: str,
    summary: LatencySummary,
    first_request_ms: float | None,
    counts: tuple[int, int, int] | None,
    note: str,
    commit: str,
) -> dict[str, str]:
    """One row of probe_summary.csv."""
    n_known, n_unknown, n_wrong = counts if counts is not None else ("", "", "")
    return {
        "series": series,
        "side": side,
        "n_requests": str(summary.n),
        "p50_ms": repr(summary.p50_ms),
        "p95_ms": repr(summary.p95_ms),
        "min_ms": repr(summary.min_ms),
        "max_ms": repr(summary.max_ms),
        "mean_ms": repr(summary.mean_ms),
        "first_request_ms": "" if first_request_ms is None else repr(first_request_ms),
        "n_known": str(n_known),
        "n_unknown": str(n_unknown),
        "n_not_as_expected": str(n_wrong),
        "note": note,
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": commit,
    }


def send_requests(args: argparse.Namespace, paths: ProjectPaths, commit: str) -> None:
    """Send the series, compare the answers, append the rows of both tables."""
    expected = json.loads(args.expected.read_text(encoding="utf-8"))
    plan = [(KNOWN, customer) for customer in expected["known"]]
    plan += [(UNKNOWN, customer) for customer in expected["unknown"]]
    # Known and unknown customers are mixed in a fixed way: sorted by the id itself.
    plan.sort(key=lambda item: item[1][::-1])
    if args.limit is not None:
        plan = plan[: args.limit]
    token = os.environ.get(TOKEN_VARIABLE, "")
    results = probe(args.url, [customer for _, customer in plan], token, args.timeout_seconds)
    n_wrong = 0
    for (kind, customer), result in zip(plan, results, strict=True):
        wanted = expected["known"][customer] if kind == KNOWN else expected["fallback"]
        source = "ranker" if kind == KNOWN else "fallback"
        as_expected = (
            result.status == 200 and result.source == source and list(result.articles) == wanted
        )
        n_wrong += 0 if as_expected else 1
        row = {
            "series": args.series,
            "position": str(result.position),
            "kind": kind,
            "status": str(result.status),
            "latency_ms": repr(result.latency_ms),
            "source": result.source,
            "prediction_set": result.prediction_set,
            "as_expected": str(as_expected),
        }
        append_row(paths.serving_table_file("probe_requests"), REQUEST_COLUMNS, row)
    # The first request opens the connection (and may start an instance): it is reported
    # on its own and is not part of the percentiles.
    later = [result.latency_ms for result in results[1:]] or [results[0].latency_ms]
    counts = (
        sum(kind == KNOWN for kind, _ in plan),
        sum(kind == UNKNOWN for kind, _ in plan),
        n_wrong,
    )
    summary = summarize(later)
    row = summary_row(
        args.series, SIDE_CLIENT, summary, results[0].latency_ms, counts, args.note, commit
    )
    append_row(paths.serving_table_file("probe_summary"), SUMMARY_COLUMNS, row)
    LOGGER.info(
        "%s: first request %.1f ms; then n=%d p50 %.1f ms p95 %.1f ms; %d not as expected",
        args.series,
        results[0].latency_ms,
        summary.n,
        summary.p50_ms,
        summary.p95_ms,
        n_wrong,
    )


def summarize_server_log(args: argparse.Namespace, paths: ProjectPaths, commit: str) -> None:
    """Append the summary of the latencies the service logged for a series."""
    entries = json.loads(args.log_json.read_text(encoding="utf-8"))
    summary = summarize(log_latencies(entries))
    row = summary_row(args.series, SIDE_SERVER, summary, None, None, args.note, commit)
    append_row(paths.serving_table_file("probe_summary"), SUMMARY_COLUMNS, row)
    LOGGER.info(
        "%s (server side): n=%d p50 %.3f ms p95 %.3f ms",
        args.series,
        summary.n,
        summary.p50_ms,
        summary.p95_ms,
    )


def main() -> None:
    """Run the chosen command."""
    args = parse_args()
    setup_logging()
    paths = get_paths()
    commit, _ = git_state(paths.project_root, paths.results_root)
    if args.command == "expect":
        write_expectations(args, paths)
    elif args.command == "requests":
        send_requests(args, paths, commit)
    else:
        summarize_server_log(args, paths, commit)


if __name__ == "__main__":
    main()
