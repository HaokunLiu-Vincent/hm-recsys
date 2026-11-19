"""A disk cache of the encoded DeepFM inputs of one target week (speed only; proposal §16.8).

Every training of a target week encodes the same rows in the same way: the bin
edges come from the training rows of the window, and neither the seed nor the
model changes them. So the first training of a (target week, input mode) writes
the encoded arrays, and the later ones read them back.

    artifacts/deepfm/cache/{target}_{numeric_input}_{key}/
        transform.json                  the fitted transform (hmrec.deepfm.inputs)
        train_codes.npy                 uint8 [n_train_rows, n_code_fields]
        train_values.npy / train_missing.npy     only with value fields
        train_labels.npy                uint8 [n_train_rows]
        train_query_ids.npy             int32 [n_train_rows]
        {part}_codes.npy, {part}_values.npy, {part}_missing.npy,
        {part}_query_ids.npy, {part}_article_idx.npy      one set per part file
        cache.json                      written LAST: the source and every file's sha256

`key` is a hash of the SOURCE of the arrays: the target week, the training
folds with the sha256 of their sample files (as recorded by M7 and by LightGBM's
Dataset), the sha256 of the record of the target week's wide table, the feature
columns and the settings of the transform. Another source gives another folder.

The cache changes no value: it stores exactly the arrays hmrec.deepfm.inputs
computes (tests/unit/test_deepfm_cache.py). Before a cache is used,
    - its recorded source must equal the source the training expects, and
    - EVERY file must have the sha256 recorded in cache.json;
anything else raises (a damaged cache is never silently rebuilt).

This module imports neither torch nor LightGBM.
"""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hmrec.deepfm.inputs import EncodedRows, InputTransform, load_transform, save_transform
from hmrec.locks import file_sha256

KEY_LENGTH = 12
RECORD_FILE = "cache.json"
TRANSFORM_FILE = "transform.json"
TRAIN = "train"
CODES = "codes"
VALUES = "values"
MISSING = "missing"
LABELS = "labels"
QUERY_IDS = "query_ids"
ARTICLE_IDX = "article_idx"
UNFINISHED_SUFFIX = ".unfinished"


def source_key(source: Mapping[str, Any]) -> str:
    """12 hex characters identifying a cache source (the sha256 of its sorted JSON)."""
    text = json.dumps(source, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:KEY_LENGTH]


@dataclass(frozen=True)
class CachedTraining:
    """The training rows of a window as read from (or written to) the cache.

    Attributes:
        transform: The fitted input transform.
        rows: The encoded training rows.
        labels: uint8 [n_rows].
        query_ids: int32 [n_rows].
    """

    transform: InputTransform
    rows: EncodedRows
    labels: np.ndarray
    query_ids: np.ndarray


@dataclass(frozen=True)
class CachedPart:
    """The candidates of one part file of the target week, encoded.

    Attributes:
        name: The part file's name, e.g. "part-00000.parquet".
        rows: The encoded candidate rows.
        query_ids: int32 [n_rows].
        article_idx: int32 [n_rows].
    """

    name: str
    rows: EncodedRows
    query_ids: np.ndarray
    article_idx: np.ndarray


def _save_rows(directory: Path, prefix: str, rows: EncodedRows) -> None:
    """Write the arrays of encoded rows as {prefix}_codes.npy (and values / missing)."""
    np.save(directory / f"{prefix}_{CODES}.npy", rows.codes)
    if rows.values is not None and rows.missing is not None:
        np.save(directory / f"{prefix}_{VALUES}.npy", rows.values)
        np.save(directory / f"{prefix}_{MISSING}.npy", rows.missing)


def _load_rows(directory: Path, prefix: str) -> EncodedRows:
    """Read the arrays written by _save_rows."""
    codes = np.load(directory / f"{prefix}_{CODES}.npy")
    values_file = directory / f"{prefix}_{VALUES}.npy"
    if not values_file.exists():
        return EncodedRows(codes, None, None)
    return EncodedRows(codes, np.load(values_file), np.load(directory / f"{prefix}_{MISSING}.npy"))


class CacheWriter:
    """Writes one cache folder; nothing counts until finish() has written cache.json.

    The arrays go to a folder with the suffix ".unfinished"; finish() writes the
    record and renames the folder, so a reader never sees a half-written cache.
    """

    def __init__(self, directory: Path, source: Mapping[str, Any]) -> None:
        """Start a cache in `directory` (which must not exist) for the given source.

        Raises:
            FileExistsError: If the folder, or an unfinished one, already exists.
        """
        self.directory = directory
        self.source = dict(source)
        self.work_dir = directory.with_name(directory.name + UNFINISHED_SUFFIX)
        self.parts: list[str] = []
        if directory.exists() or self.work_dir.exists():
            raise FileExistsError(
                f"{directory} (or its unfinished copy) already exists; a cache is never replaced. "
                "Look at it and move it away by hand."
            )
        self.work_dir.mkdir(parents=True)

    def write_training(self, training: CachedTraining) -> None:
        """Write the transform and the encoded training rows."""
        save_transform(training.transform, self.work_dir / TRANSFORM_FILE)
        _save_rows(self.work_dir, TRAIN, training.rows)
        np.save(self.work_dir / f"{TRAIN}_{LABELS}.npy", training.labels)
        np.save(self.work_dir / f"{TRAIN}_{QUERY_IDS}.npy", training.query_ids)

    def write_part(self, part: CachedPart) -> None:
        """Write the encoded candidates of one part file."""
        prefix = Path(part.name).stem
        _save_rows(self.work_dir, prefix, part.rows)
        np.save(self.work_dir / f"{prefix}_{QUERY_IDS}.npy", part.query_ids)
        np.save(self.work_dir / f"{prefix}_{ARTICLE_IDX}.npy", part.article_idx)
        self.parts.append(part.name)

    def finish(self) -> None:
        """Record every file's sha256 and make the cache visible."""
        files = sorted(path.name for path in self.work_dir.iterdir())
        record = {
            "source": self.source,
            "parts": self.parts,
            "sha256": {name: file_sha256(self.work_dir / name) for name in files},
        }
        (self.work_dir / RECORD_FILE).write_text(json.dumps(record, indent=1), encoding="utf-8")
        self.work_dir.rename(self.directory)


class CacheReader:
    """Reads one finished cache folder after it has been verified."""

    def __init__(self, directory: Path, parts: tuple[str, ...]) -> None:
        """Use open_cache instead: it verifies the folder first."""
        self.directory = directory
        self.parts = parts

    def read_training(self) -> CachedTraining:
        """The transform and the encoded training rows."""
        return CachedTraining(
            transform=load_transform(self.directory / TRANSFORM_FILE),
            rows=_load_rows(self.directory, TRAIN),
            labels=np.load(self.directory / f"{TRAIN}_{LABELS}.npy"),
            query_ids=np.load(self.directory / f"{TRAIN}_{QUERY_IDS}.npy"),
        )

    def read_part(self, name: str) -> CachedPart:
        """The encoded candidates of one part file."""
        if name not in self.parts:
            raise ValueError(f"{self.directory} has no part {name!r}; it holds {self.parts}")
        prefix = Path(name).stem
        return CachedPart(
            name=name,
            rows=_load_rows(self.directory, prefix),
            query_ids=np.load(self.directory / f"{prefix}_{QUERY_IDS}.npy"),
            article_idx=np.load(self.directory / f"{prefix}_{ARTICLE_IDX}.npy"),
        )


def open_cache(directory: Path, source: Mapping[str, Any]) -> CacheReader | None:
    """Verify a cache folder and return a reader, or None if there is no cache yet.

    Args:
        directory: The cache folder of this source.
        source: The source the caller expects (see the module docstring).

    Raises:
        ValueError: If the folder exists but has no record, was written for
            another source, holds other files than its record lists, or a file's
            sha256 differs from the recorded one.
    """
    if not directory.exists():
        return None
    record_file = directory / RECORD_FILE
    if not record_file.is_file():
        raise ValueError(f"{directory} has no {RECORD_FILE}: it is not a finished cache")
    record = json.loads(record_file.read_text(encoding="utf-8"))
    if record["source"] != dict(source):
        raise ValueError(f"{directory} was written for another source than the one expected")
    recorded: dict[str, str] = record["sha256"]
    present = sorted(path.name for path in directory.iterdir() if path.name != RECORD_FILE)
    if present != sorted(recorded):
        raise ValueError(f"{directory} holds other files than its record lists")
    for name, expected in recorded.items():
        actual = file_sha256(directory / name)
        if actual != expected:
            raise ValueError(f"{directory / name}: sha256 is {actual}, its record says {expected}")
    return CacheReader(directory, tuple(record["parts"]))
