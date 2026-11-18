"""Streaming LightGBM Dataset construction from sample files (proposal §1.3, §9).

Input: the sample files of the 4 training weeks (hmrec.ranker.sampling), in
time order. Output: a constructed lgb.Dataset saved as a LightGBM binary file
under artifacts/lgb_datasets/, with a .json record next to it.

Why lgb.Sequence: 4 weeks of raw float32 features are 3-6 GB. LightGBM only needs
the BINNED data (1 byte per value). A Sequence hands the rows over one Parquet
row group at a time (28 MB), so the raw features are never in memory at once.

Facts about lgb.Sequence in LightGBM 4.7.0 that this module relies on (checked on
synthetic data, docs/decisions.md 2026-11-18):
    - seq[i] (one row) is used to sample rows for the bin boundaries and MUST
      return float64; the sampled indices come in increasing order;
    - seq[a:b] (a batch of `batch_size` rows) may return float32;
    - building twice gives byte-identical binary files; but the bins differ from
      the ones of a Dataset built from one numpy array, so every Ranker of this
      project (M7, M8) is built through this module.

Group order: rows are sorted by (query_id, article_idx) inside a file and
query_id grows with the fold's position, so the concatenated rows are sorted by
query_id and LightGBM's group sizes are the run lengths of query_id.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl
import pyarrow.parquet as pq

from hmrec.locks import file_sha256
from hmrec.ranker.sampling import LABEL, QUERY_ID

BINARY_FILE = "train.bin"
RECORD_FILE = "dataset.json"


class ParquetSequence(lgb.Sequence):
    """Rows of one sample file as float features, read one block at a time.

    A block is `block_rows` consecutive rows = exactly one Parquet row group of the
    file (hmrec.ranker.sampling.RowGroupWriter writes row groups of this size), so
    reading a block reads one row group and nothing else. Only the most recently
    used block is kept in memory.

    Attributes:
        batch_size: Rows LightGBM asks for at a time (= block_rows).
    """

    def __init__(
        self, path: Path, feature_names: Sequence[str], n_rows: int, block_rows: int
    ) -> None:
        """Create the sequence; nothing is read until rows are requested.

        Args:
            path: The sample file.
            feature_names: The feature columns, in model order.
            n_rows: Rows of the file (from its record).
            block_rows: Rows per block (the file's row group size).
        """
        if block_rows <= 0:
            raise ValueError(f"block_rows must be > 0, got {block_rows}")
        self.path = path
        self.feature_names = list(feature_names)
        self.n_rows = n_rows
        self.batch_size = block_rows
        self._cached_index = -1
        self._cached_block = np.zeros((0, len(self.feature_names)), dtype=np.float32)
        self._file: pq.ParquetFile | None = None

    def __len__(self) -> int:
        """Rows of the file."""
        return self.n_rows

    def _open(self) -> pq.ParquetFile:
        """Open the file once and check that its row groups are the blocks."""
        if self._file is None:
            parquet_file = pq.ParquetFile(self.path)
            metadata = parquet_file.metadata
            sizes = [metadata.row_group(i).num_rows for i in range(metadata.num_row_groups)]
            # Every row group but the last must hold exactly batch_size rows; otherwise
            # "block i = row group i" would be wrong and rows would be misplaced.
            if sum(sizes) != self.n_rows or any(size != self.batch_size for size in sizes[:-1]):
                raise ValueError(
                    f"{self.path}: row groups {sizes[:3]}... do not match {self.n_rows} rows in "
                    f"blocks of {self.batch_size}"
                )
            self._file = parquet_file
        return self._file

    def _block(self, block_index: int) -> np.ndarray:
        """Rows [block_index * batch_size, ...) as float32 [rows, n_features]."""
        if block_index != self._cached_index:
            table = self._open().read_row_group(block_index, columns=self.feature_names)
            # Select again: read_row_group returns the columns in FILE order.
            block = pl.DataFrame(pl.from_arrow(table)).select(self.feature_names)
            # float32 for all columns: the int16 category codes (0..132) are exact in float32.
            values = block.cast(pl.Float32).to_numpy()
            self._cached_block = np.ascontiguousarray(values, dtype=np.float32)
            self._cached_index = block_index
        return self._cached_block

    def _rows(self, start: int, stop: int) -> np.ndarray:
        """Rows [start, stop) as float32 [stop - start, n_features]."""
        pieces = []
        position = start
        while position < stop:
            block_index = position // self.batch_size
            block_start = block_index * self.batch_size
            block = self._block(block_index)
            piece_stop = min(stop, block_start + block.shape[0])
            pieces.append(block[position - block_start : piece_stop - block_start])
            position = piece_stop
        if len(pieces) == 1:
            return pieces[0]
        return np.concatenate(pieces)

    def __getitem__(self, idx: int | slice | list[int]) -> np.ndarray:
        """One row (float64 [n_features]) or a range of rows (float32 [rows, n_features])."""
        if isinstance(idx, slice):
            start, stop, step = idx.indices(self.n_rows)
            if step != 1:
                raise ValueError(f"only contiguous slices are supported, got step {step}")
            return self._rows(start, stop)
        if isinstance(idx, int | np.integer):
            row = int(idx)
            if not 0 <= row < self.n_rows:
                raise IndexError(f"row {row} is outside 0..{self.n_rows - 1}")
            block = self._block(row // self.batch_size)
            # float64 and a fresh array: LightGBM requires doubles for the bin sample.
            return block[row % self.batch_size].astype(np.float64)
        raise TypeError(f"index must be an int or a slice, got {type(idx).__name__}")


def run_lengths(values: np.ndarray) -> np.ndarray:
    """Lengths of the runs of equal consecutive values, int32 [n_runs].

    E.g. [7, 7, 9, 9, 9, 12] -> [2, 3, 1].
    """
    if len(values) == 0:
        return np.zeros(0, dtype=np.int32)
    boundaries = np.flatnonzero(np.diff(values)) + 1
    edges = np.concatenate(([0], boundaries, [len(values)]))
    return np.diff(edges).astype(np.int32)


def read_labels_and_groups(sample_files: Sequence[Path]) -> tuple[np.ndarray, np.ndarray]:
    """Labels and LightGBM group sizes of the concatenated sample files.

    Args:
        sample_files: The sample files in time order.

    Returns:
        (labels, group_sizes): labels is int8 [n_rows] in file order; group_sizes
        is int32 [n_queries], the number of rows of each query in row order.

    Raises:
        ValueError: If query_id ever decreases: groups would not match the rows.
    """
    keys = pl.concat(
        [pl.read_parquet(sample_file, columns=[QUERY_ID, LABEL]) for sample_file in sample_files]
    )
    query_ids = keys[QUERY_ID].to_numpy()
    if np.any(np.diff(query_ids) < 0):
        raise ValueError(
            "query_id decreases in the concatenated sample files: the files must be sorted "
            "by query_id and given in time order"
        )
    return keys[LABEL].to_numpy().astype(np.int8), run_lengths(query_ids)


@dataclass(frozen=True)
class DatasetSource:
    """One sample file that goes into a Dataset.

    Attributes:
        fold: The training fold.
        sample_file: Its sample file.
        n_rows: Rows of the file (from its record).
        sha256: sha256 of the file (from its record).
    """

    fold: str
    sample_file: Path
    n_rows: int
    sha256: str


def build_dataset(
    sources: Sequence[DatasetSource],
    feature_names: Sequence[str],
    categorical_names: Sequence[str],
    params: Mapping[str, Any],
    block_rows: int,
) -> tuple[lgb.Dataset, int]:
    """Build (bin) a LightGBM Dataset from the sample files, streaming the rows.

    Args:
        sources: The training folds' sample files in time order.
        feature_names: Feature columns in model order.
        categorical_names: The categorical features (integer codes) among them.
        params: LightGBM parameters that define the bins (hmrec.ranker.train.dataset_params).
        block_rows: Row group size of the sample files.

    Returns:
        (the constructed Dataset with labels and group sizes, its number of queries).
    """
    sequences = [
        ParquetSequence(source.sample_file, feature_names, source.n_rows, block_rows)
        for source in sources
    ]
    labels, group_sizes = read_labels_and_groups([source.sample_file for source in sources])
    if len(labels) != sum(source.n_rows for source in sources):
        raise ValueError("the sample files' rows differ from their records")
    dataset = lgb.Dataset(
        sequences,
        label=labels,
        group=group_sizes,
        feature_name=list(feature_names),
        categorical_feature=list(categorical_names),
        params=dict(params),
    )
    return dataset.construct(), len(group_sizes)


# ---------------------------------------------------------------- binary file + record


@dataclass(frozen=True)
class DatasetRecord:
    """dataset.json: what a binary Dataset was built from.

    Attributes:
        experiment: "full" or "minus_r6".
        target_fold: The fold the model trained on this Dataset predicts.
        ratio: Negatives per positive.
        train_folds: The training folds in time order.
        sample_sha256: fold -> sha256 of its sample file.
        n_rows: Training rows.
        n_queries: Training queries (LightGBM groups).
        feature_names: Features in model order.
        categorical_names: The categorical features.
        params: The LightGBM parameters the bins were built with.
        binary_sha256: sha256 of train.bin.
        git_commit: HEAD when it was built.
    """

    experiment: str
    target_fold: str
    ratio: int
    train_folds: tuple[str, ...]
    sample_sha256: dict[str, str]
    n_rows: int
    n_queries: int
    feature_names: tuple[str, ...]
    categorical_names: tuple[str, ...]
    params: dict[str, Any]
    binary_sha256: str
    git_commit: str


def read_dataset_record(directory: Path) -> DatasetRecord:
    """Read dataset.json.

    Raises:
        FileNotFoundError: If it is missing (the Dataset is not built or not finished).
    """
    path = directory / RECORD_FILE
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found: the Dataset is not built")
    content: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    for key in ("train_folds", "feature_names", "categorical_names"):
        content[key] = tuple(content[key])
    return DatasetRecord(**content)


def dataset_is_built(directory: Path) -> bool:
    """True if the Dataset is finished; a binary file without a record is an error.

    Raises:
        ValueError: If train.bin exists without dataset.json (an interrupted run;
            nothing is deleted automatically).
    """
    if (directory / RECORD_FILE).exists():
        return True
    if (directory / BINARY_FILE).exists():
        raise ValueError(
            f"{directory / BINARY_FILE} exists but {RECORD_FILE} does not: an earlier run "
            "stopped half way. Move the file away by hand, then run again."
        )
    return False


def save_dataset(dataset: lgb.Dataset, directory: Path, record: DatasetRecord) -> DatasetRecord:
    """Save a constructed Dataset as train.bin and write its record (written last).

    Args:
        dataset: The constructed Dataset.
        directory: artifacts/lgb_datasets/{experiment}/{target_fold}_r{ratio}/.
        record: The record; its binary_sha256 is filled in here.

    Returns:
        The record as written.
    """
    if dataset_is_built(directory):
        raise FileExistsError(f"{directory} already holds a Dataset; it is never replaced")
    directory.mkdir(parents=True, exist_ok=True)
    binary_file = directory / BINARY_FILE
    dataset.save_binary(binary_file)
    written = DatasetRecord(**{**asdict(record), "binary_sha256": file_sha256(binary_file)})
    (directory / RECORD_FILE).write_text(json.dumps(asdict(written), indent=2), encoding="utf-8")
    return written


def load_dataset(
    directory: Path, expected: DatasetRecord, train_params: Mapping[str, Any]
) -> lgb.Dataset:
    """Load a binary Dataset after proving it is the expected one.

    Checks: the record equals `expected` in everything but binary_sha256 and
    git_commit (same folds, same sample files by sha256, same features, same bin
    parameters), and train.bin has the recorded sha256.

    Args:
        directory: Where train.bin and dataset.json are.
        expected: What the caller expects, built from the CURRENT sample records
            and config (binary_sha256 and git_commit are ignored).
        train_params: The LightGBM parameters of the training run.

    Returns:
        The Dataset, ready for lgb.train.

    Raises:
        ValueError: On any difference.
    """
    record = read_dataset_record(directory)
    ignored = {"binary_sha256", "git_commit"}
    recorded = {key: value for key, value in asdict(record).items() if key not in ignored}
    wanted = {key: value for key, value in asdict(expected).items() if key not in ignored}
    different = sorted(key for key in wanted if recorded[key] != wanted[key])
    if different:
        raise ValueError(
            f"{directory}: the Dataset was built from other inputs than expected; "
            f"different: {different}"
        )
    binary_file = directory / BINARY_FILE
    actual = file_sha256(binary_file)
    if actual != record.binary_sha256:
        raise ValueError(
            f"{binary_file}: sha256 is {actual}, but its record says {record.binary_sha256}"
        )
    return lgb.Dataset(binary_file, params=dict(train_params)).construct()
