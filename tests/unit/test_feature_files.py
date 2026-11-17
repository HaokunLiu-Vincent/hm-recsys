"""Tests for hmrec.feature_files: the record of a fold's part files and its verification."""

import json
from pathlib import Path

import polars as pl
import pytest

from hmrec.feature_files import (
    FeatureFiles,
    PartFile,
    describe_part,
    read_feature_files,
    verify_feature_files,
    write_feature_files,
)
from hmrec.locks import file_sha256

COLUMNS = ("query_id", "article_idx", "pop_rank")


def _part(directory: Path, name: str, query_ids: list[int]) -> PartFile:
    table = pl.DataFrame(
        {
            "query_id": query_ids,
            "article_idx": [5] * len(query_ids),
            "pop_rank": [1.0] * len(query_ids),
        },
        schema={"query_id": pl.Int32, "article_idx": pl.Int32, "pop_rank": pl.Float32},
    )
    directory.mkdir(parents=True, exist_ok=True)
    table.write_parquet(directory / name)
    return describe_part(directory / name)


def _fold(tmp_path: Path) -> tuple[Path, FeatureFiles]:
    """A finished fold: two part files (2 + 1 rows) and their features.json."""
    directory = tmp_path / "features" / "F1"
    parts = (
        _part(directory, "part-00000.parquet", [1, 2]),
        _part(directory, "part-00001.parquet", [3]),
    )
    files = FeatureFiles(
        "F1", COLUMNS, (), parts, n_rows=3, n_queries=3, details={"git_commit": "abc"}
    )
    write_feature_files(directory, files)
    return directory, files


def test_part_description_is_the_files_sha256_and_row_count(tmp_path: Path) -> None:
    part = _part(tmp_path, "part-00000.parquet", [1, 2])
    assert part == PartFile("part-00000.parquet", file_sha256(tmp_path / "part-00000.parquet"), 2)


def test_record_round_trip_and_verification(tmp_path: Path) -> None:
    directory, files = _fold(tmp_path)
    assert read_feature_files(directory) == files
    assert verify_feature_files(directory, COLUMNS) == files
    content = json.loads((directory / "features.json").read_text())
    assert [part["n_rows"] for part in content["parts"]] == [2, 1]
    assert content["columns"] == list(COLUMNS)
    assert all(len(part["sha256"]) == 64 for part in content["parts"])


def test_record_is_written_once_and_rows_must_add_up(tmp_path: Path) -> None:
    directory, files = _fold(tmp_path)
    with pytest.raises(FileExistsError, match="never replaced"):
        write_feature_files(directory, files)
    other = tmp_path / "features" / "F2"
    other.mkdir(parents=True)
    with pytest.raises(ValueError, match="do not add up"):
        write_feature_files(other, FeatureFiles("F2", COLUMNS, (), files.parts, 4, 3, {}))
    assert not (other / "features.json").exists()


def test_a_fold_without_a_record_is_not_finished(tmp_path: Path) -> None:
    directory = tmp_path / "features" / "F1"
    _part(directory, "part-00000.parquet", [1])
    with pytest.raises(FileNotFoundError, match="not finished"):
        verify_feature_files(directory, COLUMNS)


def test_a_replaced_part_file_is_refused(tmp_path: Path) -> None:
    directory, _ = _fold(tmp_path)
    # The same rows written again with another value: other bytes, same row count.
    pl.DataFrame(
        {"query_id": [1, 2], "article_idx": [5, 5], "pop_rank": [9.0, 9.0]},
        schema={"query_id": pl.Int32, "article_idx": pl.Int32, "pop_rank": pl.Float32},
    ).write_parquet(directory / "part-00000.parquet")
    with pytest.raises(ValueError, match="This is not the frozen file"):
        verify_feature_files(directory, COLUMNS)


def test_missing_and_extra_part_files_are_refused(tmp_path: Path) -> None:
    directory, _ = _fold(tmp_path)
    _part(directory, "part-00002.parquet", [4])  # a file the record does not know
    with pytest.raises(ValueError, match="differ from the record"):
        verify_feature_files(directory, COLUMNS)
    (directory / "part-00002.parquet").unlink()
    (directory / "part-00001.parquet").unlink()  # a recorded file is gone
    with pytest.raises(ValueError, match="differ from the record"):
        verify_feature_files(directory, COLUMNS)


def test_other_columns_than_expected_are_refused(tmp_path: Path) -> None:
    directory, _ = _fold(tmp_path)
    with pytest.raises(ValueError, match="lists other columns than expected"):
        verify_feature_files(directory, (*COLUMNS, "rep_rank"))
    with pytest.raises(ValueError, match="lists other columns than expected"):
        verify_feature_files(directory, ("article_idx", "query_id", "pop_rank"))  # other order
