"""data/processed/metadata.json: which settings the processed data was built with.

M0 writes this file at the end of preprocessing. It records day_zero, the age
bucket edges and a hash of configs/preprocess.yaml. Any code that reads
data/processed/ must first call `check_processed_metadata`, which fails if the
current config differs. Without this check, changing day_zero in the config
would silently shift every t_dat_day by some days and break the time cut-off.
"""

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from hmrec.config import PreprocessConfig, preprocess_config_hash

# The fields compared by check_processed_metadata. Row counts and timestamps
# are informative only and are not compared.
CHECKED_FIELDS = ("day_zero", "age_buckets", "invalid_age_bucket", "unknown_age_bucket")


def build_processed_metadata(
    config: PreprocessConfig, row_counts: dict[str, int]
) -> dict[str, Any]:
    """Build the content of metadata.json.

    Args:
        config: The preprocess config used for this run.
        row_counts: Table name -> number of rows written, e.g. {"transactions": 31788324}.

    Returns:
        A JSON-serializable dict with keys day_zero (str YYYY-MM-DD), age_buckets
        (list of {name, min_age, max_age}), invalid_age_bucket, unknown_age_bucket,
        config_hash, row_counts and created_at_utc.
    """
    return {
        "day_zero": config.day_zero.isoformat(),
        "age_buckets": [asdict(bucket) for bucket in config.age_buckets],
        "invalid_age_bucket": config.invalid_age_bucket,
        "unknown_age_bucket": config.unknown_age_bucket,
        "config_hash": preprocess_config_hash(config),
        "row_counts": dict(row_counts),
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def write_processed_metadata(path: Path, metadata: dict[str, Any]) -> None:
    """Write metadata.json (pretty-printed, keys sorted)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_processed_metadata(path: Path) -> dict[str, Any]:
    """Read metadata.json.

    Raises:
        FileNotFoundError: If the file does not exist (M0 has not been run).
        ValueError: If the top level is not a JSON object.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found; run scripts/m0_preprocess.py before reading processed data"
        )
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError(f"{path}: expected a JSON object, got {type(parsed).__name__}")
    return parsed


def check_processed_metadata(path: Path, config: PreprocessConfig) -> None:
    """Fail if data/processed/ was built with a different preprocess config.

    The individual fields are compared first so the error says WHAT changed;
    the hash comparison then catches every other difference.

    Args:
        path: data/processed/metadata.json.
        config: The current preprocess config.

    Raises:
        FileNotFoundError: If metadata.json does not exist.
        ValueError: If any field or the config hash differs.
    """
    stored = read_processed_metadata(path)
    expected = build_processed_metadata(config, row_counts={})
    for field in CHECKED_FIELDS:
        if stored.get(field) != expected[field]:
            raise ValueError(
                f"{path}: {field} is {stored.get(field)!r} in the processed data but "
                f"{expected[field]!r} in the current config; re-run scripts/m0_preprocess.py"
            )
    if stored.get("config_hash") != expected["config_hash"]:
        raise ValueError(
            f"{path}: config_hash {stored.get('config_hash')!r} does not match the current "
            f"preprocess config ({expected['config_hash']!r}); re-run scripts/m0_preprocess.py"
        )


def scan_processed_parquet(
    table_path: Path, metadata_path: Path, config: PreprocessConfig
) -> pl.LazyFrame:
    """Lazily scan a table in data/processed/ after checking metadata.json.

    Downstream code should read processed tables through this function, so a
    config change can never be silently mixed with old processed data.

    Args:
        table_path: e.g. paths.transactions_file.
        metadata_path: paths.processed_metadata_file.
        config: The current preprocess config.

    Returns:
        pl.scan_parquet(table_path).

    Raises:
        FileNotFoundError / ValueError: See check_processed_metadata.
    """
    check_processed_metadata(metadata_path, config)
    return pl.scan_parquet(table_path)
