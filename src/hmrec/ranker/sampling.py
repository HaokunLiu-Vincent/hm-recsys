"""In-query random negative sampling for Ranker training (proposal §9.1).

One of the two places allowed to read labels (the other is hmrec.eval).

Input: a fold's frozen wide table (artifacts/features/{fold}/) and its labels.
Output: one Parquet file per (fold, negatives-per-positive ratio r) with columns
    query_id, article_idx, label (int8: 1 = bought in the label week), then the features
sorted by (query_id, article_idx), in row groups of a fixed size.

The sample of one fold and ratio:
    - only queries with at least one positive candidate are kept (LambdaMART has a
      zero gradient on a query without positives);
    - every positive row is kept;
    - negatives are drawn uniformly at random, without replacement, from the SAME
      query's candidates: min(number of negatives, r x number of positives).

How the draw is made reproducible: every query has its own random generator seeded
with (seed, query_id). It shuffles the query's negatives (taken in article_idx
order) and the first r x positives of the shuffled list are kept. So
    - a query's sample depends only on the seed, its query_id and its candidates,
      not on chunking, on the processing order or on other queries;
    - the sample for a smaller r is a subset of the sample for a larger r.

A sample file is written once (its .json record is written last) and never replaced.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from hmrec.config import RankerSamplingConfig
from hmrec.data.folds import Fold
from hmrec.feature_files import METADATA_FILE, verify_feature_files
from hmrec.locks import file_sha256
from hmrec.logging_utils import get_logger
from hmrec.ranker.ablation import ARTICLE_IDX, QUERY_ID, Experiment

LOGGER = get_logger("ranker.sampling")

FOLD = "fold"
LABEL = "label"
NEGATIVE_ORDER = "negative_order"
N_POSITIVES = "n_positives"
KEYS = [QUERY_ID, ARTICLE_IDX]
PARQUET_COMPRESSION = "zstd"


# ---------------------------------------------------------------- labels


def load_training_labels(
    fold: Fold, labels_file: Path, allowed_phases: Sequence[str]
) -> pl.DataFrame:
    """Read a fold's label set G_q for training, after checking the fold's phase.

    Args:
        fold: The training fold.
        labels_file: artifacts/labels/{fold}.parquet.
        allowed_phases: Phases whose labels the caller may read (M7: F and V).

    Returns:
        One row per (query_id, article_idx) bought in the label week.
        Columns: query_id (int32), article_idx (int32).

    Raises:
        ValueError: If the fold's phase is not allowed (the file is not opened), or
            the file holds another fold.
    """
    if fold.phase not in allowed_phases:
        raise ValueError(
            f"labels of fold {fold.name} (phase {fold.phase}) must not be read here: "
            f"allowed phases are {tuple(allowed_phases)}"
        )
    labels = pl.read_parquet(labels_file, columns=[QUERY_ID, FOLD, ARTICLE_IDX])
    folds_in_file = labels[FOLD].unique().to_list()
    if folds_in_file != [fold.name]:
        raise ValueError(f"{labels_file} holds folds {folds_in_file}, expected [{fold.name!r}]")
    return labels.select(QUERY_ID, ARTICLE_IDX)


# ---------------------------------------------------------------- sampling (pure)


def label_positive_queries(rows: pl.DataFrame, labels: pl.DataFrame) -> pl.DataFrame:
    """Add the label column and keep only queries with at least one positive row.

    Args:
        rows: One row per (query_id, article_idx): a chunk of whole queries.
        labels: One row per (query_id, article_idx) in G_q.

    Returns:
        The rows of the kept queries plus `label` (int8) and `n_positives` (the
        query's number of positive rows), sorted by (query_id, article_idx).
    """
    positives = labels.select(QUERY_ID, ARTICLE_IDX).with_columns(
        pl.lit(1, dtype=pl.Int8).alias(LABEL)
    )
    marked = rows.join(positives, on=KEYS, how="left").with_columns(pl.col(LABEL).fill_null(0))
    counted = marked.with_columns(
        pl.col(LABEL).cast(pl.Int32).sum().over(QUERY_ID).alias(N_POSITIVES)
    )
    # Sorted on purpose: the negatives of a query are shuffled starting from this order.
    return counted.filter(pl.col(N_POSITIVES) > 0).sort(KEYS)


def negative_orders(query_ids: np.ndarray, labels: np.ndarray, seed: int) -> np.ndarray:
    """Give every negative row its position in its query's random order.

    Args:
        query_ids: int32 [n_rows], non-decreasing (a query's rows are adjacent).
        labels: int8 [n_rows], 1 = positive, 0 = negative.
        seed: The sampling seed.

    Returns:
        int32 [n_rows]: -1 for a positive row; for a negative row, i if it is the
        i-th (0-based) negative drawn in its query. Keeping the rows with order
        < m draws m negatives uniformly without replacement.
    """
    n_rows = len(query_ids)
    order = np.full(n_rows, -1, dtype=np.int32)
    if n_rows == 0:
        return order
    if np.any(np.diff(query_ids) < 0):
        raise ValueError("rows must be sorted by query_id before negatives are drawn")
    boundaries = np.flatnonzero(np.diff(query_ids)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [n_rows]))
    for start, end in zip(starts.tolist(), ends.tolist(), strict=True):
        negatives = start + np.flatnonzero(labels[start:end] == 0)
        # One generator per query, seeded with (seed, query_id): see the module docstring.
        generator = np.random.default_rng([seed, int(query_ids[start])])
        shuffled = generator.permutation(negatives)
        order[shuffled] = np.arange(len(negatives), dtype=np.int32)
    return order


def sample_chunk(
    rows: pl.DataFrame, labels: pl.DataFrame, seed: int, ratios: Sequence[int]
) -> dict[int, pl.DataFrame]:
    """Sample the training rows of a chunk of whole queries, for several ratios at once.

    Args:
        rows: One row per (query_id, article_idx) with the feature columns.
        labels: One row per (query_id, article_idx) in G_q (any queries).
        seed: The sampling seed.
        ratios: Negatives per positive, e.g. (40, 60, 80).

    Returns:
        ratio -> the sampled rows: every positive row and, per query,
        min(negatives, ratio x positives) negatives of that query. Columns:
        query_id, article_idx, label (int8), then the other columns of `rows`;
        sorted by (query_id, article_idx).
    """
    labelled = label_positive_queries(rows, labels)
    order = negative_orders(labelled[QUERY_ID].to_numpy(), labelled[LABEL].to_numpy(), seed)
    ordered = labelled.with_columns(pl.Series(NEGATIVE_ORDER, order, dtype=pl.Int32))
    feature_columns = [name for name in rows.columns if name not in KEYS]
    samples: dict[int, pl.DataFrame] = {}
    for ratio in ratios:
        if ratio <= 0:
            raise ValueError(f"a negatives-per-positive ratio must be > 0, got {ratio}")
        # Positives have order -1, so they always pass.
        kept = ordered.filter(pl.col(NEGATIVE_ORDER) < ratio * pl.col(N_POSITIVES))
        samples[ratio] = kept.select(*KEYS, LABEL, *feature_columns)
    return samples


# ---------------------------------------------------------------- writing


class RowGroupWriter:
    """Append tables to ONE Parquet file in row groups of exactly `row_group_rows` rows.

    Only the last row group may be smaller. Fixed row groups let the reader
    (hmrec.ranker.dataset.ParquetSequence) load one row group at a time and know
    where every row is. Rows are buffered until a full row group is available.
    """

    def __init__(self, path: Path, row_group_rows: int) -> None:
        """Create the writer; the file is created by the first append."""
        if row_group_rows <= 0:
            raise ValueError(f"row_group_rows must be > 0, got {row_group_rows}")
        self.path = path
        self.row_group_rows = row_group_rows
        self.n_rows = 0
        self._buffer: pl.DataFrame | None = None
        self._writer: pq.ParquetWriter | None = None

    def append(self, table: pl.DataFrame) -> None:
        """Add rows (in order); full row groups are written at once."""
        self._buffer = table if self._buffer is None else pl.concat([self._buffer, table])
        while self._buffer.height >= self.row_group_rows:
            self._write(self._buffer.head(self.row_group_rows))
            self._buffer = self._buffer.slice(self.row_group_rows)

    def close(self) -> None:
        """Write the remaining rows as the last row group and close the file."""
        if self._buffer is None:
            raise ValueError(f"{self.path}: nothing was appended")
        if self._buffer.height > 0 or self._writer is None:
            self._write(self._buffer)
        if self._writer is not None:
            self._writer.close()

    def _write(self, table: pl.DataFrame) -> None:
        """Write one row group."""
        arrow_table = table.to_arrow()
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = pq.ParquetWriter(
                self.path, arrow_table.schema, compression=PARQUET_COMPRESSION
            )
        self._writer.write_table(arrow_table, row_group_size=self.row_group_rows)
        self.n_rows += table.height


# ---------------------------------------------------------------- record


@dataclass(frozen=True)
class SampleRecord:
    """The .json record next to a sample file.

    Attributes:
        fold: The training fold.
        experiment: "full" or "minus_r6".
        ratio: Negatives per positive.
        seed: The sampling seed.
        n_rows: Rows of the file.
        n_queries: Queries in the file (each has at least one positive).
        n_positives: Positive rows.
        rows_per_query_mean: Mean rows per query.
        rows_per_query_p50: Median rows per query.
        rows_per_query_p90: 90th percentile of rows per query.
        columns: The file's columns in order.
        sha256: sha256 of the file.
        features_record_sha256: sha256 of the features.json the rows came from.
        git_commit: HEAD when the file was written.
    """

    fold: str
    experiment: str
    ratio: int
    seed: int
    n_rows: int
    n_queries: int
    n_positives: int
    rows_per_query_mean: float
    rows_per_query_p50: float
    rows_per_query_p90: float
    columns: tuple[str, ...]
    sha256: str
    features_record_sha256: str
    git_commit: str

    @property
    def positive_share(self) -> float:
        """Positive rows / all rows."""
        return self.n_positives / self.n_rows


def write_sample_record(path: Path, record: SampleRecord) -> None:
    """Write the record. It is written once and never replaced."""
    if path.exists():
        raise FileExistsError(f"{path} already exists; samples are never replaced")
    path.write_text(json.dumps(asdict(record), indent=2), encoding="utf-8")


def read_sample_record(path: Path) -> SampleRecord:
    """Read a sample's record.

    Raises:
        FileNotFoundError: If the record is missing (the sample is not finished).
    """
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found: the sample is not finished")
    content: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    content["columns"] = tuple(content["columns"])
    return SampleRecord(**content)


def record_file(sample_file: Path) -> Path:
    """The record of a sample file: the same name with .json."""
    return sample_file.with_suffix(".json")


def verify_sample_file(sample_file: Path) -> SampleRecord:
    """Prove that a sample file is the one its record describes (sha256), before it is used."""
    record = read_sample_record(record_file(sample_file))
    actual = file_sha256(sample_file)
    if actual != record.sha256:
        raise ValueError(
            f"{sample_file}: sha256 is {actual}, but its record says {record.sha256}. "
            "This is not the file that was written."
        )
    return record


# ---------------------------------------------------------------- one fold


class _SampleStats:
    """Counts of one ratio's sample while a fold is processed."""

    def __init__(self) -> None:
        self.n_positives = 0
        self.rows_per_query: list[np.ndarray] = []

    def add(self, sample: pl.DataFrame) -> None:
        """Count one chunk's sample."""
        self.n_positives += int(sample[LABEL].sum())
        counts = sample.group_by(QUERY_ID).agg(pl.len().alias("n"))["n"]
        self.rows_per_query.append(counts.to_numpy())

    def all_rows_per_query(self) -> np.ndarray:
        """Rows per query over the whole fold, int64 [n_queries]."""
        if not self.rows_per_query:
            return np.zeros(0, dtype=np.int64)
        return np.concatenate(self.rows_per_query).astype(np.int64)


def pending_ratios(sample_files: Mapping[int, Path]) -> list[int]:
    """The ratios whose sample is not written yet.

    Raises:
        ValueError: If a sample file exists without its record (a half-written
            file from an interrupted run; nothing is deleted automatically).
    """
    pending = []
    for ratio, sample_file in sample_files.items():
        if record_file(sample_file).exists():
            continue
        if sample_file.exists():
            raise ValueError(
                f"{sample_file} exists but its record does not: an earlier run stopped half "
                "way. Move the file away by hand, then run again."
            )
        pending.append(ratio)
    return pending


@dataclass(frozen=True)
class SampleJob:
    """What to sample for one fold.

    Attributes:
        fold: The training fold.
        features_dir: artifacts/features/{fold}/.
        labels_file: artifacts/labels/{fold}.parquet.
        experiment: Which rows and features (full or minus_r6).
        wide_columns: The columns the frozen wide table must have.
        sample_files: ratio -> output Parquet file.
        allowed_phases: Phases whose labels may be read (M7: F and V).
        git_commit: HEAD of this run.
    """

    fold: Fold
    features_dir: Path
    labels_file: Path
    experiment: Experiment
    wide_columns: tuple[str, ...]
    sample_files: Mapping[int, Path]
    allowed_phases: tuple[str, ...]
    git_commit: str


def _make_record(
    job: SampleJob, ratio: int, seed: int, writer: RowGroupWriter, stats: _SampleStats
) -> SampleRecord:
    """The record of one finished sample file."""
    rows_per_query = stats.all_rows_per_query()
    if rows_per_query.size == 0:
        raise ValueError(f"{job.fold.name}: no query has a positive candidate; nothing to train on")
    return SampleRecord(
        fold=job.fold.name,
        experiment=job.experiment.name,
        ratio=ratio,
        seed=seed,
        n_rows=writer.n_rows,
        n_queries=int(rows_per_query.size),
        n_positives=stats.n_positives,
        rows_per_query_mean=float(rows_per_query.mean()),
        rows_per_query_p50=float(np.quantile(rows_per_query, 0.5)),
        rows_per_query_p90=float(np.quantile(rows_per_query, 0.9)),
        columns=(QUERY_ID, ARTICLE_IDX, LABEL, *job.experiment.feature_names),
        sha256=file_sha256(job.sample_files[ratio]),
        features_record_sha256=file_sha256(job.features_dir / METADATA_FILE),
        git_commit=job.git_commit,
    )


def sample_fold(job: SampleJob, settings: RankerSamplingConfig) -> dict[int, SampleRecord]:
    """Write the missing sample files of one fold; finished ones are left untouched.

    The fold's phase is checked before any file is opened, then the frozen wide
    table is verified (hmrec.feature_files) and read one part file at a time (a
    part holds whole queries), so only one part is in memory.

    Args:
        job: The fold, its inputs and outputs.
        settings: Seed and row group size (configs/ranker.yaml).

    Returns:
        ratio -> record, for every ratio of the job (written now or earlier).
    """
    labels = load_training_labels(job.fold, job.labels_file, job.allowed_phases)
    ratios = pending_ratios(job.sample_files)
    if ratios:
        feature_files = verify_feature_files(job.features_dir, job.wide_columns)
        writers = {
            ratio: RowGroupWriter(job.sample_files[ratio], settings.row_group_rows)
            for ratio in ratios
        }
        stats = {ratio: _SampleStats() for ratio in ratios}
        for part in feature_files.parts:
            rows = job.experiment.apply(pl.read_parquet(job.features_dir / part.name))
            samples = sample_chunk(rows, labels, settings.seed, ratios)
            for ratio, sample in samples.items():
                writers[ratio].append(sample)
                stats[ratio].add(sample)
            LOGGER.info("%s %s: sampled %s", job.fold.name, job.experiment.name, part.name)
        for ratio in ratios:
            writers[ratio].close()
            record = _make_record(job, ratio, settings.seed, writers[ratio], stats[ratio])
            write_sample_record(record_file(job.sample_files[ratio]), record)
    return {
        ratio: read_sample_record(record_file(sample_file))
        for ratio, sample_file in job.sample_files.items()
    }
