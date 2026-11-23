"""The serving form of a prediction set: what the recommendation service loads (mlops §9; M14).

A prediction set (predictions/{anchor}/top.parquet) holds the dense indexes of
data/processed/. The service answers with the ORIGINAL identifiers of the
dataset and must start fast with little memory, so step O5 also writes the same
predictions in a form that needs no id map and no parquet reader:

    predictions/{anchor}/serving/
        customer_ids.npy    bytes, shape [n_customers]: the customer ids of the
                            set, SORTED ascending, so that a customer is found
                            by binary search (numpy.searchsorted).
        top_articles.npy    int32, shape [n_customers, top_k]: row i holds the
                            Top-K of customer_ids[i], best first, as positions
                            in article_ids.
        article_ids.npy     bytes, shape [n_articles]: the article id at every
                            position (the position is the article_idx).
        serving.json        the record: which prediction set and Ranker this is,
                            the fallback list for unknown customers, and size,
                            dtype and sha256 of the three files. Written LAST:
                            a serving set is complete only if its record exists.

Unknown customers get the fallback list of the record: the first top_k articles
of R1's popularity list for the "age missing" bucket of the same week, taken
from the tick's own R1 output (hmrec.weekly.serving_build).

Loading checks everything: the record's keys, the sha256, dtype and shape of
every file, the order of the customer ids and the range of the positions. A set
that fails one check is not served (ServingSetError).

Memory of the real `serve` set: 1,371,980 customers x 64 bytes of id (88 MB) and
x 12 x 4 bytes of positions (66 MB), plus 105,542 article ids (1 MB).

This module is at the bottom layer and imports only the standard library and
numpy: the pipeline writes with it, the service's small image reads with it.
"""

import hashlib
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

SERVING_RECORD_FILE = "serving.json"
CUSTOMER_IDS_FILE = "customer_ids.npy"
TOP_ARTICLES_FILE = "top_articles.npy"
ARTICLE_IDS_FILE = "article_ids.npy"
ARRAY_FILES: tuple[str, ...] = (CUSTOMER_IDS_FILE, TOP_ARTICLES_FILE, ARTICLE_IDS_FILE)
SERVING_SCHEMA_VERSION = 1
# The model version a response names: the first characters of the Ranker file's sha256.
MODEL_VERSION_LENGTH = 12
SOURCE_RANKER = "ranker"
SOURCE_FALLBACK = "fallback"
RECORD_KEYS: tuple[str, ...] = (
    "schema_version",
    "anchor",
    "week",
    "mode",
    "top_k",
    "n_customers",
    "n_articles",
    "prediction_sha256",
    "ranker_sha256",
    "model_version",
    "fallback",
    "files",
    "built_by",
    "git_commit",
)


class ServingSetError(ValueError):
    """A serving set is incomplete or does not pass a check; it must not be served."""


@dataclass(frozen=True)
class Fallback:
    """The list an unknown customer gets.

    Attributes:
        age_bucket: The age bucket whose R1 list this is (the "age missing" bucket).
        article_ids: top_k article ids, best first.
        source: The R1 candidates file the list was read from, relative to the
            state root, with "/".
        source_sha256: sha256 of that file.
    """

    age_bucket: str
    article_ids: tuple[str, ...]
    source: str
    source_sha256: str


@dataclass(frozen=True)
class FileEntry:
    """One array file of a serving set, as its record describes it.

    Attributes:
        sha256: sha256 of the file's bytes.
        dtype: The array's dtype as numpy prints it, e.g. "|S64" or "<i4".
        shape: The array's shape.
    """

    sha256: str
    dtype: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class ServingDescription:
    """What the builder knows about a serving set before its files are written.

    Attributes:
        anchor: The anchor of the predicted week (ISO date).
        week: The week's name, e.g. "T4".
        mode: "eval" or "serve".
        prediction_sha256: sha256 of the prediction set's top.parquet: the
            serving form belongs to exactly that file.
        ranker_sha256: sha256 of the Ranker model file behind the predictions.
        fallback: The list for unknown customers.
        built_by: The run that built it, or "backfill" for a set that was
            published before M14.
        git_commit: The commit of the code that built it.
    """

    anchor: str
    week: str
    mode: str
    prediction_sha256: str
    ranker_sha256: str
    fallback: Fallback
    built_by: str
    git_commit: str


@dataclass(frozen=True)
class ServingRecord:
    """The content of serving.json (the keys are RECORD_KEYS).

    Attributes:
        description: What the builder knew (see ServingDescription).
        top_k: Articles per customer.
        n_customers: Customers in the set.
        n_articles: Articles of the catalog (the length of article_ids.npy).
        model_version: The first 12 characters of `ranker_sha256`.
        files: File name -> its entry, for the three array files.
    """

    description: ServingDescription
    top_k: int
    n_customers: int
    n_articles: int
    model_version: str
    files: dict[str, FileEntry]

    @property
    def set_id(self) -> str:
        """The id of the prediction set in one word: "{anchor}/{model_version}"."""
        return f"{self.description.anchor}/{self.model_version}"


@dataclass(frozen=True)
class ServingSet:
    """A loaded serving set.

    Attributes:
        record: Its record.
        customer_ids: bytes, shape [n_customers], sorted ascending.
        top_articles: int32, shape [n_customers, top_k]: positions in article_ids.
        article_ids: bytes, shape [n_articles].
    """

    record: ServingRecord
    customer_ids: np.ndarray
    top_articles: np.ndarray
    article_ids: np.ndarray

    def position_of(self, customer_id: str) -> int | None:
        """The row of a customer, or None if the customer is not in the set.

        Any text is accepted: an id that is not ASCII, is empty, holds a NUL
        byte or is longer than the ids of the set cannot be a known customer.
        """
        try:
            key = customer_id.encode("ascii")
        except UnicodeEncodeError:
            return None
        # numpy drops NUL bytes at the end of a bytes value when it compares; a key with
        # one could then equal a shorter stored id.
        if key == b"" or b"\x00" in key or len(key) > self.customer_ids.dtype.itemsize:
            return None
        position = int(np.searchsorted(self.customer_ids, np.bytes_(key)))
        if position < self.customer_ids.shape[0] and self.customer_ids[position] == key:
            return position
        return None

    def recommend(self, customer_id: str) -> tuple[tuple[str, ...], str]:
        """The top_k article ids for a customer and where they come from.

        Returns:
            (article ids, "ranker") for a customer of the set: the precomputed
            Top-K. (the fallback list, "fallback") for anyone else.
        """
        position = self.position_of(customer_id)
        if position is None:
            return self.record.description.fallback.article_ids, SOURCE_FALLBACK
        articles = self.article_ids[self.top_articles[position]].tolist()
        return tuple(article.decode("ascii") for article in articles), SOURCE_RANKER


# ---------------------------------------------------------------- the record as JSON


def record_to_json(record: ServingRecord) -> dict[str, Any]:
    """The record as the JSON object that is written."""
    description = record.description
    return {
        "schema_version": SERVING_SCHEMA_VERSION,
        "anchor": description.anchor,
        "week": description.week,
        "mode": description.mode,
        "top_k": record.top_k,
        "n_customers": record.n_customers,
        "n_articles": record.n_articles,
        "prediction_sha256": description.prediction_sha256,
        "ranker_sha256": description.ranker_sha256,
        "model_version": record.model_version,
        "fallback": {
            **asdict(description.fallback),
            "article_ids": list(description.fallback.article_ids),
        },
        "files": {
            name: {"sha256": entry.sha256, "dtype": entry.dtype, "shape": list(entry.shape)}
            for name, entry in record.files.items()
        },
        "built_by": description.built_by,
        "git_commit": description.git_commit,
    }


def record_from_json(raw: Any) -> ServingRecord:
    """Build a ServingRecord from a parsed serving.json.

    Raises:
        ServingSetError: If a key is missing or unknown, the schema version is
            another one, or the files are not exactly the three array files.
    """
    if not isinstance(raw, dict) or set(raw) != set(RECORD_KEYS):
        raise ServingSetError(f"{SERVING_RECORD_FILE} must have exactly the keys {RECORD_KEYS}")
    if raw["schema_version"] != SERVING_SCHEMA_VERSION:
        raise ServingSetError(
            f"{SERVING_RECORD_FILE} has schema_version {raw['schema_version']!r}, "
            f"expected {SERVING_SCHEMA_VERSION}"
        )
    try:
        fallback = Fallback(
            age_bucket=raw["fallback"]["age_bucket"],
            article_ids=tuple(raw["fallback"]["article_ids"]),
            source=raw["fallback"]["source"],
            source_sha256=raw["fallback"]["source_sha256"],
        )
        files = {
            name: FileEntry(entry["sha256"], entry["dtype"], tuple(entry["shape"]))
            for name, entry in raw["files"].items()
        }
    except (KeyError, TypeError) as error:
        raise ServingSetError(f"{SERVING_RECORD_FILE} is not a serving record: {error}") from error
    if set(files) != set(ARRAY_FILES):
        raise ServingSetError(
            f"{SERVING_RECORD_FILE} must describe exactly the files {ARRAY_FILES}"
        )
    description = ServingDescription(
        anchor=raw["anchor"],
        week=raw["week"],
        mode=raw["mode"],
        prediction_sha256=raw["prediction_sha256"],
        ranker_sha256=raw["ranker_sha256"],
        fallback=fallback,
        built_by=raw["built_by"],
        git_commit=raw["git_commit"],
    )
    return ServingRecord(
        description=description,
        top_k=raw["top_k"],
        n_customers=raw["n_customers"],
        n_articles=raw["n_articles"],
        model_version=raw["model_version"],
        files=files,
    )


# ---------------------------------------------------------------- the checks


def check_arrays(
    customer_ids: np.ndarray,
    top_articles: np.ndarray,
    article_ids: np.ndarray,
    fallback: Fallback,
) -> None:
    """Fail unless the three arrays and the fallback list form a valid serving set.

    Args:
        customer_ids: bytes, shape [n_customers].
        top_articles: int32, shape [n_customers, top_k].
        article_ids: bytes, shape [n_articles].
        fallback: The list for unknown customers.

    Raises:
        ServingSetError: With the first rule that is broken.
    """
    if customer_ids.ndim != 1 or customer_ids.dtype.kind != "S" or customer_ids.shape[0] == 0:
        raise ServingSetError("customer ids must be a non-empty 1-D array of bytes")
    if article_ids.ndim != 1 or article_ids.dtype.kind != "S" or article_ids.shape[0] == 0:
        raise ServingSetError("article ids must be a non-empty 1-D array of bytes")
    n_customers = customer_ids.shape[0]
    if top_articles.dtype != np.int32 or top_articles.ndim != 2:
        raise ServingSetError("the Top-K must be a 2-D int32 array")
    if top_articles.shape[0] != n_customers or top_articles.shape[1] == 0:
        raise ServingSetError(
            f"the Top-K has shape {top_articles.shape}, expected {n_customers} rows"
        )
    # Strictly ascending: sorted (binary search needs it) and no customer twice.
    if n_customers > 1 and not bool(np.all(customer_ids[:-1] < customer_ids[1:])):
        raise ServingSetError("the customer ids are not sorted strictly ascending")
    lowest, highest = int(top_articles.min()), int(top_articles.max())
    if lowest < 0 or highest >= article_ids.shape[0]:
        raise ServingSetError(
            f"the Top-K holds article positions {lowest}..{highest}, outside "
            f"0..{article_ids.shape[0] - 1}"
        )
    top_k = top_articles.shape[1]
    listed = fallback.article_ids
    if len(listed) != top_k or len(set(listed)) != top_k:
        raise ServingSetError(
            f"the fallback list must hold {top_k} distinct article ids, got {len(listed)}"
        )
    known = set(article_ids.tolist())
    unknown = [article for article in listed if article.encode("ascii") not in known]
    if unknown:
        raise ServingSetError(f"the fallback list holds unknown article ids: {unknown}")


def bytes_sha256(data: bytes) -> str:
    """sha256 of some bytes, as 64 lower-case hexadecimal characters."""
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------- write


def write_serving_set(
    folder: Path,
    description: ServingDescription,
    customer_ids: np.ndarray,
    top_articles: np.ndarray,
    article_ids: np.ndarray,
) -> ServingRecord:
    """Check a serving set and write its three array files, then its record.

    Args:
        folder: The serving folder (predictions/{anchor}/serving/, or the same
            place under a staging folder). It must not hold a record yet.
        description: What the builder knows about the set.
        customer_ids: bytes, shape [n_customers], sorted ascending.
        top_articles: int32, shape [n_customers, top_k].
        article_ids: bytes, shape [n_articles].

    Returns:
        The record that was written.

    Raises:
        ServingSetError: If the arrays do not pass check_arrays.
        FileExistsError: If the folder already holds a record: a serving set is
            never replaced.
    """
    check_arrays(customer_ids, top_articles, article_ids, description.fallback)
    record_file = folder / SERVING_RECORD_FILE
    if record_file.exists():
        raise FileExistsError(f"{record_file} already exists; a serving set is never replaced")
    folder.mkdir(parents=True, exist_ok=True)
    arrays = {
        CUSTOMER_IDS_FILE: customer_ids,
        TOP_ARTICLES_FILE: top_articles,
        ARTICLE_IDS_FILE: article_ids,
    }
    files: dict[str, FileEntry] = {}
    for name, array in arrays.items():
        buffer = io.BytesIO()
        # allow_pickle=False: the file holds plain numbers and bytes, never Python objects.
        np.save(buffer, array, allow_pickle=False)
        data = buffer.getvalue()
        (folder / name).write_bytes(data)
        files[name] = FileEntry(bytes_sha256(data), array.dtype.str, tuple(array.shape))
    record = ServingRecord(
        description=description,
        top_k=int(top_articles.shape[1]),
        n_customers=int(customer_ids.shape[0]),
        n_articles=int(article_ids.shape[0]),
        model_version=description.ranker_sha256[:MODEL_VERSION_LENGTH],
        files=files,
    )
    # The record is written LAST: its presence says that the three files are complete.
    record_file.write_text(json.dumps(record_to_json(record), indent=2) + "\n", encoding="utf-8")
    return record


# ---------------------------------------------------------------- read


def read_serving_record(folder: Path) -> ServingRecord:
    """Read serving.json of a serving folder.

    Raises:
        ServingSetError: If the record is missing, is not JSON, or is not a
            serving record.
    """
    record_file = folder / SERVING_RECORD_FILE
    if not record_file.is_file():
        raise ServingSetError(f"{record_file} is missing: the serving set is not complete")
    try:
        raw = json.loads(record_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ServingSetError(f"{record_file} cannot be read: {error}") from error
    return record_from_json(raw)


def read_checked_array(folder: Path, name: str, entry: FileEntry) -> np.ndarray:
    """Read one array file and compare it with its entry in the record.

    Raises:
        ServingSetError: If the file is missing, or its sha256, dtype or shape
            is not the recorded one.
    """
    path = folder / name
    try:
        data = path.read_bytes()
    except OSError as error:
        raise ServingSetError(f"{path} cannot be read: {error}") from error
    sha256 = bytes_sha256(data)
    if sha256 != entry.sha256:
        raise ServingSetError(f"{path}: sha256 {sha256} is not the recorded {entry.sha256}")
    try:
        array = np.load(io.BytesIO(data), allow_pickle=False)
    except ValueError as error:
        raise ServingSetError(f"{path} is not an array file: {error}") from error
    if array.dtype.str != entry.dtype or tuple(array.shape) != entry.shape:
        raise ServingSetError(
            f"{path}: dtype {array.dtype.str} and shape {tuple(array.shape)} are not the "
            f"recorded {entry.dtype} and {entry.shape}"
        )
    return array


def load_serving_set(folder: Path) -> ServingSet:
    """Load a serving set and check all of it (see the module docstring).

    Args:
        folder: The serving folder of a prediction set.

    Raises:
        ServingSetError: If anything is missing or a check fails. Nothing of
            such a set is returned.
    """
    record = read_serving_record(folder)
    customer_ids = read_checked_array(folder, CUSTOMER_IDS_FILE, record.files[CUSTOMER_IDS_FILE])
    top_articles = read_checked_array(folder, TOP_ARTICLES_FILE, record.files[TOP_ARTICLES_FILE])
    article_ids = read_checked_array(folder, ARTICLE_IDS_FILE, record.files[ARTICLE_IDS_FILE])
    check_arrays(customer_ids, top_articles, article_ids, record.description.fallback)
    counts = (customer_ids.shape[0], top_articles.shape[1], article_ids.shape[0])
    if counts != (record.n_customers, record.top_k, record.n_articles):
        raise ServingSetError(
            f"{folder}: the arrays hold {counts} customers, articles per customer and "
            f"articles; the record says {(record.n_customers, record.top_k, record.n_articles)}"
        )
    if record.model_version != record.description.ranker_sha256[:MODEL_VERSION_LENGTH]:
        raise ServingSetError(f"{folder}: the model version is not the start of the Ranker sha256")
    return ServingSet(record, customer_ids, top_articles, article_ids)
