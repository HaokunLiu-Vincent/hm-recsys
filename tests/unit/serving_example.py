"""A tiny serving set and a state folder around it, for the tests of the service and rollback.

The set is small enough to check by hand (top_k = 2):

    article ids (position = article_idx)   0000000100 ... 0000000104
    customers, sorted by id                ALICE < BOB < CAROL (64 hexadecimal characters)
        ALICE   positions (0, 1)  ->  0000000100, 0000000101
        BOB     positions (2, 3)  ->  0000000102, 0000000103
        CAROL   positions (4, 0)  ->  0000000104, 0000000100
    fallback list (unknown customers)      0000000103, 0000000101

write_state_set writes a whole published prediction set into a state folder: the
prediction file, its manifest, the serving form and the published marker. The
Ranker "model" of a set is the text of its anchor, so two sets have different
model versions.
"""

import asyncio
import hashlib
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from hmrec.serving_set import (
    Fallback,
    ServingDescription,
    ServingRecord,
    write_serving_set,
)

ALICE = "0a" * 32
BOB = "5b" * 32
CAROL = "c0" * 32
UNKNOWN = "ff" * 32
ARTICLES = tuple(f"{100 + position:010d}" for position in range(5))
TOP = {ALICE: (0, 1), BOB: (2, 3), CAROL: (4, 0)}
FALLBACK_IDS = (ARTICLES[3], ARTICLES[1])
COMMIT = "c" * 40
R1_SOURCE = "weeks/{anchor}/recall_r1_popularity/candidates.parquet"


def example_arrays(
    top: dict[str, tuple[int, int]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The three arrays of the example set: customer ids (sorted), Top-2 positions, article ids."""
    rows = TOP if top is None else top
    customers = sorted(rows)
    customer_ids = np.array(customers, dtype=np.bytes_)
    top_articles = np.array([rows[customer] for customer in customers], dtype=np.int32)
    return customer_ids, top_articles, np.array(ARTICLES, dtype=np.bytes_)


def example_description(
    anchor: str = "2020-09-23", prediction_sha256: str = "d" * 64, ranker_sha256: str = "e" * 64
) -> ServingDescription:
    """The description of the example set."""
    fallback = Fallback("AGE_UNKNOWN", FALLBACK_IDS, R1_SOURCE.format(anchor=anchor), "f" * 64)
    return ServingDescription(
        anchor=anchor,
        week="W" + anchor.replace("-", ""),
        mode="serve",
        prediction_sha256=prediction_sha256,
        ranker_sha256=ranker_sha256,
        fallback=fallback,
        built_by=f"{anchor}-serve-a1",
        git_commit=COMMIT,
    )


def write_example_set(folder: Path, anchor: str = "2020-09-23", **changes: Any) -> ServingRecord:
    """Write the example serving set into `folder` (a serving folder)."""
    customer_ids, top_articles, article_ids = example_arrays(changes.pop("top", None))
    description = example_description(anchor, **changes)
    return write_serving_set(folder, description, customer_ids, top_articles, article_ids)


def pointer_text(anchor: str) -> str:
    """The text of state/current.json that names the set of `anchor` (as step O5 writes it)."""
    week = date.fromisoformat(anchor)
    pointer = {
        "schema_version": 1,
        "anchor": anchor,
        "mode": "serve",
        "closed_week": (week - timedelta(days=7)).isoformat(),
        "prediction_set": f"predictions/{anchor}",
        "models": [
            {
                "role": "ranker",
                "path": f"weeks/{anchor}/ranker_model/model.txt",
                "sha256": hashlib.sha256(anchor.encode()).hexdigest(),
            }
        ],
        "run_id": f"{anchor}-serve-a1",
    }
    return json.dumps(pointer, indent=2) + "\n"


def write_pointer(state_root: Path, anchor: str) -> None:
    """Point state/current.json of a state folder at the set of `anchor`."""
    pointer_file = state_root / "state" / "current.json"
    pointer_file.parent.mkdir(parents=True, exist_ok=True)
    pointer_file.write_text(pointer_text(anchor), encoding="utf-8")


def write_state_set(
    state_root: Path, anchor: str, top: dict[str, tuple[int, int]] | None = None
) -> ServingRecord:
    """Write one PUBLISHED prediction set into a state folder (see the module docstring).

    Returns:
        The serving record of the set.
    """
    folder = state_root / "predictions" / anchor
    folder.mkdir(parents=True)
    predictions_file = folder / "top.parquet"
    predictions_file.write_bytes(f"prediction table of {anchor}".encode())
    prediction_sha256 = hashlib.sha256(predictions_file.read_bytes()).hexdigest()
    ranker_sha256 = hashlib.sha256(anchor.encode()).hexdigest()
    pointer = json.loads(pointer_text(anchor))
    manifest = {
        "schema_version": 1,
        "anchor": anchor,
        "week": "W" + anchor.replace("-", ""),
        "mode": "serve",
        "predictions": {"file": "top.parquet", "sha256": prediction_sha256},
        "models": pointer["models"],
        "run_id": pointer["run_id"],
        "git_commit": COMMIT,
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    week_dir = state_root / "weeks" / anchor
    week_dir.mkdir(parents=True)
    (week_dir / "published.json").write_text("{}\n", encoding="utf-8")
    return write_example_set(
        folder / "serving",
        anchor,
        top=top,
        prediction_sha256=prediction_sha256,
        ranker_sha256=ranker_sha256,
    )


# ---------------------------------------------------------------- calling an ASGI app


async def get(app: Any, path: str) -> tuple[int, Any]:
    """Send one GET request to an ASGI application, without a server and without a client library.

    Returns:
        (status code, the JSON body).
    """
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [],
        "client": ("127.0.0.1", 50000),
        "server": ("testserver", 80),
    }
    request_sent = False
    messages: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": b"", "more_body": False}
        # The client stays connected until the response is complete.
        await asyncio.sleep(3600)
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        messages.append(message)

    await app(scope, receive, send)
    status = next(m["status"] for m in messages if m["type"] == "http.response.start")
    body = b"".join(m.get("body", b"") for m in messages if m["type"] == "http.response.body")
    return status, json.loads(body)


def get_now(app: Any, path: str) -> tuple[int, Any]:
    """`get` for a test that is not itself asynchronous."""
    return asyncio.run(get(app, path))
