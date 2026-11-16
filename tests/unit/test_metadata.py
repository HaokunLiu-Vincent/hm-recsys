import dataclasses
import json
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from hmrec.config import AgeBucket, PreprocessConfig, load_preprocess_config
from hmrec.data.metadata import (
    build_processed_metadata,
    check_processed_metadata,
    read_processed_metadata,
    scan_processed_parquet,
    write_processed_metadata,
)

REPO_PREPROCESS_YAML = Path(__file__).resolve().parents[2] / "configs" / "preprocess.yaml"


@pytest.fixture
def config() -> PreprocessConfig:
    return load_preprocess_config(REPO_PREPROCESS_YAML)


@pytest.fixture
def metadata_path(tmp_path: Path, config: PreprocessConfig) -> Path:
    path = tmp_path / "processed" / "metadata.json"
    write_processed_metadata(path, build_processed_metadata(config, {"transactions": 7}))
    return path


def test_written_metadata_has_the_documented_fields(metadata_path: Path) -> None:
    stored = json.loads(metadata_path.read_text())
    assert stored["day_zero"] == "2018-09-20"
    assert stored["age_buckets"][0] == {"name": "AGE_18_24", "min_age": 18, "max_age": 24}
    assert stored["invalid_age_bucket"] == "AGE_INVALID"
    assert stored["unknown_age_bucket"] == "AGE_UNKNOWN"
    assert len(stored["config_hash"]) == 64
    assert stored["row_counts"] == {"transactions": 7}
    assert read_processed_metadata(metadata_path) == stored


def test_same_config_passes(metadata_path: Path, config: PreprocessConfig) -> None:
    check_processed_metadata(metadata_path, config)


def test_changed_day_zero_is_named_in_the_error(
    metadata_path: Path, config: PreprocessConfig
) -> None:
    changed = dataclasses.replace(config, day_zero=date(2018, 9, 21))
    with pytest.raises(ValueError, match="day_zero is '2018-09-20' in the processed data"):
        check_processed_metadata(metadata_path, changed)


def test_changed_age_bucket_edge_raises(metadata_path: Path, config: PreprocessConfig) -> None:
    buckets = list(config.age_buckets)
    buckets[0] = AgeBucket(name="AGE_18_24", min_age=18, max_age=25)
    buckets[1] = AgeBucket(name="AGE_25_34", min_age=26, max_age=34)
    changed = dataclasses.replace(config, age_buckets=tuple(buckets))
    with pytest.raises(ValueError, match="age_buckets"):
        check_processed_metadata(metadata_path, changed)


def test_other_config_change_is_caught_by_the_hash(
    metadata_path: Path, config: PreprocessConfig
) -> None:
    changed = dataclasses.replace(config, transactions_row_group_size=5)
    with pytest.raises(ValueError, match="config_hash"):
        check_processed_metadata(metadata_path, changed)


def test_missing_metadata_raises(tmp_path: Path, config: PreprocessConfig) -> None:
    with pytest.raises(FileNotFoundError, match="run scripts/m0_preprocess.py"):
        check_processed_metadata(tmp_path / "metadata.json", config)


def test_scan_processed_parquet_checks_first(
    metadata_path: Path, config: PreprocessConfig, tmp_path: Path
) -> None:
    table_path = tmp_path / "table.parquet"
    pl.DataFrame({"x": [1, 2]}).write_parquet(table_path)
    assert scan_processed_parquet(table_path, metadata_path, config).collect()["x"].to_list() == [
        1,
        2,
    ]
    changed = dataclasses.replace(config, day_zero=date(2018, 9, 21))
    with pytest.raises(ValueError, match="day_zero"):
        scan_processed_parquet(table_path, metadata_path, changed)
