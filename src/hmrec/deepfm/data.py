"""Read what a DeepFM training needs: the M7 sample files and the target week's wide table.

Training rows are THE SAME rows LightGBM trained on (proposal §16.1): the sample
files M7 wrote for the locked ratio. A file is used only if

    - its sha256 equals the one in its own record (hmrec.ranker.sampling), and
    - that sha256 is the one recorded in the LightGBM Dataset of the same target
      week (artifacts/lgb_datasets/full/{target}_r{ratio}/dataset.json).

Nothing is sampled again and no label file is opened here: a sample file
already carries its label column (written by M7 from F and V labels).

The target week is scored from its frozen wide table, verified with
hmrec.feature_files.verify_feature_files before any part is read.

This module imports neither torch nor LightGBM. dataset.json is read as plain
JSON, because the module that owns it (hmrec.ranker.dataset) imports LightGBM.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from hmrec.deepfm.inputs import RawRows
from hmrec.locks import FeaturesLock
from hmrec.ranker.sampling import LABEL, verify_sample_file

QUERY_ID = "query_id"
ARTICLE_IDX = "article_idx"
DATASET_RECORD_FILE = "dataset.json"
CODES_KEY = "codes"


@dataclass(frozen=True)
class FeatureColumns:
    """The locked features, split into the two kinds of columns.

    Attributes:
        numeric: The numeric features in lock order (float32 columns).
        categorical: The categorical features in lock order (int16 code columns).
    """

    numeric: tuple[str, ...]
    categorical: tuple[str, ...]


def feature_columns(lock: FeaturesLock) -> FeatureColumns:
    """Split the locked feature list into numeric and categorical features."""
    categorical = set(lock.categorical_features)
    names = [feature.name for feature in lock.features]
    return FeatureColumns(
        numeric=tuple(name for name in names if name not in categorical),
        categorical=tuple(name for name in names if name in categorical),
    )


def categorical_sizes(category_codes_file: Path, names: Sequence[str]) -> tuple[int, ...]:
    """Number of codes of every categorical field: its values plus 1 for "missing" (code 0).

    Read from the shared code table artifacts/features/category_codes.json, so
    a field has the same size for every fold, also for a value that happens not
    to occur in one training window.

    Raises:
        KeyError: If a field is not in the table.
    """
    table = json.loads(category_codes_file.read_text(encoding="utf-8"))[CODES_KEY]
    return tuple(len(table[name]) + 1 for name in names)


# ---------------------------------------------------------------- sample files


def lightgbm_sample_sha256(dataset_dir: Path) -> dict[str, str]:
    """Fold -> sha256 of the sample file the LightGBM Dataset in `dataset_dir` was built from.

    Raises:
        FileNotFoundError: If the Dataset's record does not exist.
    """
    record = dataset_dir / DATASET_RECORD_FILE
    if not record.is_file():
        raise FileNotFoundError(f"{record} not found: LightGBM has no Dataset for this week")
    return dict(json.loads(record.read_text(encoding="utf-8"))["sample_sha256"])


def check_sample_files(
    sample_files: dict[str, Path], lightgbm_sha256: dict[str, str]
) -> dict[str, str]:
    """Fail unless every sample file is the one LightGBM trained on.

    Args:
        sample_files: Training fold -> its sample file, in time order.
        lightgbm_sha256: From lightgbm_sample_sha256 for the same target week.

    Returns:
        Fold -> the checked sha256.

    Raises:
        ValueError: If the folds differ from LightGBM's, a file's sha256 differs
            from its own record, or from the one LightGBM recorded.
    """
    if list(sample_files) != list(lightgbm_sha256):
        raise ValueError(
            f"training folds {list(sample_files)} are not the folds of the LightGBM Dataset "
            f"{list(lightgbm_sha256)}"
        )
    checked = {}
    for fold, path in sample_files.items():
        record = verify_sample_file(path)  # the file against its own record
        if record.sha256 != lightgbm_sha256[fold]:
            raise ValueError(
                f"{path}: sha256 {record.sha256} is not the one LightGBM trained on "
                f"({lightgbm_sha256[fold]})"
            )
        checked[fold] = record.sha256
    return checked


@dataclass(frozen=True)
class TrainingRows:
    """The training rows of one window, as stored in the sample files.

    Attributes:
        raw: The feature values (numeric float32, categorical int16 codes).
        labels: uint8 [n_rows]; 1 = bought in the label week.
        query_ids: int32 [n_rows]; rows of a query are adjacent and query_id never decreases.
    """

    raw: RawRows
    labels: np.ndarray
    query_ids: np.ndarray


def count_rows(files: Sequence[Path]) -> list[int]:
    """Rows of every Parquet file, from its metadata."""
    return [int(pl.scan_parquet(file).select(pl.len()).collect().item()) for file in files]


def load_training_rows(sample_files: Sequence[Path], columns: FeatureColumns) -> TrainingRows:
    """Read the sample files of the training folds, one file at a time, into one set of arrays.

    The arrays are allocated once at their final size and filled file by file,
    so the peak is the final arrays plus one file, not twice the final arrays.

    Args:
        sample_files: The training folds' sample files in time order.
        columns: Which columns are numeric and which categorical.

    Raises:
        ValueError: If query_id decreases somewhere (the files must be in time order).
    """
    rows_per_file = count_rows(sample_files)
    n_rows = sum(rows_per_file)
    numeric = np.empty((n_rows, len(columns.numeric)), dtype=np.float32)
    categorical = np.empty((n_rows, len(columns.categorical)), dtype=np.int16)
    labels = np.empty(n_rows, dtype=np.uint8)
    query_ids = np.empty(n_rows, dtype=np.int32)
    start = 0
    for file, n_file_rows in zip(sample_files, rows_per_file, strict=True):
        stop = start + n_file_rows
        table = pl.read_parquet(file)
        numeric[start:stop] = table.select(columns.numeric).to_numpy()
        categorical[start:stop] = table.select(columns.categorical).to_numpy()
        labels[start:stop] = table[LABEL].to_numpy()
        query_ids[start:stop] = table[QUERY_ID].to_numpy()
        start = stop
    if n_rows > 1 and bool(np.any(np.diff(query_ids) < 0)):
        raise ValueError("query_id decreases: the sample files must be given in time order")
    return TrainingRows(RawRows(numeric, categorical), labels, query_ids)


# ---------------------------------------------------------------- the target week


@dataclass(frozen=True)
class CandidateRows:
    """All candidates of some queries of the target week (one part file).

    Attributes:
        raw: The feature values.
        query_ids: int32 [n_rows].
        article_idx: int32 [n_rows].
    """

    raw: RawRows
    query_ids: np.ndarray
    article_idx: np.ndarray


def read_candidate_part(part_file: Path, columns: FeatureColumns) -> CandidateRows:
    """Read one part file of a (verified) wide table."""
    table = pl.read_parquet(part_file)
    raw = RawRows(
        numeric=table.select(columns.numeric).to_numpy().astype(np.float32, copy=False),
        categorical=table.select(columns.categorical).to_numpy().astype(np.int16, copy=False),
    )
    return CandidateRows(raw, table[QUERY_ID].to_numpy(), table[ARTICLE_IDX].to_numpy())
