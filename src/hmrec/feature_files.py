"""The record of a fold's feature files: artifacts/features/{fold}/features.json.

features.json is written LAST when a fold's wide table is built, so its
presence means the fold is complete. It lists every part file with its sha256
and row count, and the column list of the table.

artifacts/ is not in git and a wide table is frozen once written (AGENTS.md
§3.1). verify_feature_files is how a reader proves it got the frozen table:
every listed part exists with the recorded sha256, rows and columns, and
there is no part file that the record does not know. The Ranker code (M7) must
call it before it reads a fold's features.

This module belongs to the bottom layer (standard library, polars, hmrec.locks):
features/ writes the record, ranker/ verifies it, and neither imports the other.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from hmrec.locks import file_sha256

METADATA_FILE = "features.json"
PART_GLOB = "part-*.parquet"
FORMAT_VERSION = 1


@dataclass(frozen=True)
class PartFile:
    """One part file of a fold's wide table.

    Attributes:
        name: File name, e.g. "part-00003.parquet".
        sha256: sha256 of the file's bytes.
        n_rows: Rows in the file.
    """

    name: str
    sha256: str
    n_rows: int


@dataclass(frozen=True)
class FeatureFiles:
    """The content of features.json.

    Attributes:
        fold: Fold name.
        columns: Every column of the wide table in file order: query_id,
            article_idx, then the features.
        categorical_features: The features stored as int16 category codes.
        parts: The part files in order (a query is never split over two parts).
        n_rows: Rows of the whole table (= sum of the parts' rows).
        n_queries: Queries of the fold.
        details: Other recorded facts (check counts, timings, git commit).
    """

    fold: str
    columns: tuple[str, ...]
    categorical_features: tuple[str, ...]
    parts: tuple[PartFile, ...]
    n_rows: int
    n_queries: int
    details: dict[str, Any]


def describe_part(path: Path) -> PartFile:
    """Read back a written part file: its sha256 and its row count."""
    n_rows = int(pl.scan_parquet(path).select(pl.len()).collect().item())
    return PartFile(path.name, file_sha256(path), n_rows)


def write_feature_files(directory: Path, files: FeatureFiles) -> None:
    """Write features.json. It is written once and never replaced.

    Raises:
        FileExistsError: If the file exists.
        ValueError: If the parts' rows do not add up to n_rows.
    """
    path = directory / METADATA_FILE
    if path.exists():
        raise FileExistsError(f"{path} already exists; features are never replaced")
    if sum(part.n_rows for part in files.parts) != files.n_rows:
        raise ValueError(f"{files.fold}: the parts' rows do not add up to {files.n_rows}")
    content = {
        "format_version": FORMAT_VERSION,
        "fold": files.fold,
        "n_rows": files.n_rows,
        "n_queries": files.n_queries,
        "columns": list(files.columns),
        "categorical_features": list(files.categorical_features),
        "parts": [
            {"name": part.name, "sha256": part.sha256, "n_rows": part.n_rows}
            for part in files.parts
        ],
        "details": files.details,
    }
    path.write_text(json.dumps(content, indent=2), encoding="utf-8")


def read_feature_files(directory: Path) -> FeatureFiles:
    """Read features.json (no check of the part files; see verify_feature_files).

    Raises:
        FileNotFoundError: If the fold has no features.json (it is not finished).
    """
    path = directory / METADATA_FILE
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found: the fold's features are not finished")
    content: Mapping[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if content["format_version"] != FORMAT_VERSION:
        raise ValueError(f"{path}: format_version {content['format_version']} is not supported")
    return FeatureFiles(
        fold=content["fold"],
        columns=tuple(content["columns"]),
        categorical_features=tuple(content["categorical_features"]),
        parts=tuple(
            PartFile(part["name"], part["sha256"], int(part["n_rows"])) for part in content["parts"]
        ),
        n_rows=int(content["n_rows"]),
        n_queries=int(content["n_queries"]),
        details=dict(content["details"]),
    )


def verify_feature_files(directory: Path, expected_columns: Sequence[str]) -> FeatureFiles:
    """Prove that a fold's part files are exactly the ones recorded in features.json.

    Call this BEFORE reading a fold's features (M7 does, for every fold it
    trains or predicts on).

    Checks: the recorded column list equals `expected_columns`; the part files
    on disk are exactly the recorded ones; every part has the recorded sha256,
    row count and columns; the rows add up to n_rows.

    Args:
        directory: artifacts/features/{fold}/.
        expected_columns: The columns the caller expects: query_id,
            article_idx, then the feature names of configs/features.yaml.

    Returns:
        The verified record.

    Raises:
        FileNotFoundError: If features.json or a part file is missing.
        ValueError: On any difference.
    """
    files = read_feature_files(directory)
    if list(files.columns) != list(expected_columns):
        raise ValueError(
            f"{directory}: features.json lists other columns than expected "
            f"(recorded {len(files.columns)}, expected {len(list(expected_columns))})"
        )
    on_disk = sorted(path.name for path in directory.glob(PART_GLOB))
    recorded = [part.name for part in files.parts]
    if on_disk != sorted(recorded):
        raise ValueError(
            f"{directory}: part files on disk {on_disk} differ from the record {recorded}"
        )
    for part in files.parts:
        path = directory / part.name
        actual = describe_part(path)
        if actual != part:
            raise ValueError(
                f"{path}: sha256 / rows are {actual.sha256} / {actual.n_rows}, but features.json "
                f"records {part.sha256} / {part.n_rows}. This is not the frozen file."
            )
        if list(pl.read_parquet_schema(path)) != list(files.columns):
            raise ValueError(f"{path}: its columns differ from the recorded column list")
    if sum(part.n_rows for part in files.parts) != files.n_rows:
        raise ValueError(f"{directory}: the parts' rows do not add up to {files.n_rows}")
    return files
