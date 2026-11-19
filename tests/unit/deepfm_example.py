"""A tiny workspace for the DeepFM tests; not a test module itself.

Everything lives under a temporary folder and nothing of the project's data is
read (only the real fold manifest, the real features lock and the real configs):

    folds F1 F2 F3 F4 (training) and V1 (target), each with 30 queries of 12
    candidates (360 rows): the 71 locked features with random values, some
    missing, and labels that a model can learn (the 2 candidates with the
    largest value of the first numeric feature are the positives).
    For every fold: the wide table with its features.json, the label file and
    the M7 sample file with ratio 40 (written with hmrec.ranker.sampling: with
    10 negatives per query every row is kept).
    A stand-in for LightGBM's Dataset record of V1 (dataset.json with the four
    sample files' sha256) and the shared category code table.
"""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl

from hmrec.config import (
    DeepFMConfig,
    RankerSamplingConfig,
    load_base_config,
    load_deepfm_config,
)
from hmrec.data.folds import load_fold_manifest
from hmrec.deepfm.trial import TrialInputs
from hmrec.feature_files import FeatureFiles, describe_part, write_feature_files
from hmrec.locks import file_sha256, load_features_lock
from hmrec.paths import ProjectPaths, build_paths
from hmrec.ranker.ablation import build_experiment, wide_table_columns
from hmrec.ranker.sampling import SampleJob, sample_fold

REPO = Path(__file__).resolve().parents[2]
BASE = load_base_config(REPO / "configs" / "base.yaml")
MANIFEST = load_fold_manifest(REPO / "configs" / "fold_manifest.yaml")
LOCK = load_features_lock(REPO / "configs" / "locks" / "features.lock.yaml")
REAL_CONFIG = load_deepfm_config(REPO / "configs" / "deepfm.yaml")
TRAIN_FOLDS = ("F1", "F2", "F3", "F4")
TARGET = "V1"
N_QUERIES = 30
N_CANDIDATES = 12
N_POSITIVES = 2
RATIO = 40
CATEGORY_SIZE = 4  # values per categorical field in the tiny code table (codes 0 ... 4)
QUERY_ID_OFFSET = 10_000_000
CATEGORICAL = tuple(LOCK.categorical_features)
NUMERIC = tuple(f.name for f in LOCK.features if f.name not in set(CATEGORICAL))
# Columns that have missing values in the tiny data (the 2nd and 3rd numeric features).
MISSING_COLUMNS = NUMERIC[1:3]
SIGNAL = NUMERIC[0]


def tiny_config(**training: object) -> DeepFMConfig:
    """The real configuration made small: 4-dim embeddings, one hidden layer, 2 epochs on CPU."""
    model = replace(REAL_CONFIG.model, embedding_dim=4, hidden_units=(8,))
    settings = {"epochs": 2, "batch_rows": 64, "device": "cpu", **training}
    return replace(
        REAL_CONFIG,
        model=model,
        n_bins=4,
        training=replace(REAL_CONFIG.training, **settings),  # type: ignore[arg-type]  # test helper
        scoring_rows_per_chunk=100,
    )


def fold_ordinal(fold: str) -> int:
    return MANIFEST.names.index(fold)


def make_wide(fold: str, seed: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(wide table, labels) of one fold; see the module docstring."""
    rng = np.random.default_rng(seed)
    n_rows = N_QUERIES * N_CANDIDATES
    first_query = fold_ordinal(fold) * QUERY_ID_OFFSET
    query_ids = np.repeat(first_query + np.arange(N_QUERIES), N_CANDIDATES).astype(np.int32)
    article_idx = np.tile(np.arange(N_CANDIDATES), N_QUERIES).astype(np.int32)
    columns: dict[str, np.ndarray] = {"query_id": query_ids, "article_idx": article_idx}
    for name in NUMERIC:
        values = rng.normal(size=n_rows).astype(np.float32)
        if name in MISSING_COLUMNS:
            values[rng.random(n_rows) < 0.3] = np.nan
        columns[name] = values
    for name in CATEGORICAL:
        columns[name] = rng.integers(0, CATEGORY_SIZE + 1, size=n_rows).astype(np.int16)
    wide = pl.DataFrame(columns).select(wide_table_columns(LOCK))
    # Positives: the N_POSITIVES candidates of a query with the largest signal value.
    rank = pl.col(SIGNAL).rank("ordinal", descending=True).over("query_id")
    labels = wide.filter(rank <= N_POSITIVES).select(
        "query_id", pl.lit(fold).alias("fold"), "article_idx"
    )
    return wide, labels.sort("query_id", "article_idx")


def write_fold(paths: ProjectPaths, fold: str, wide: pl.DataFrame, labels: pl.DataFrame) -> None:
    """Write a fold's wide table (two parts), its features.json and its label file."""
    directory = paths.features_dir(fold)
    directory.mkdir(parents=True)
    split_at = wide["query_id"][0] + N_QUERIES // 2
    pieces = [
        wide.filter(pl.col("query_id") < split_at),
        wide.filter(pl.col("query_id") >= split_at),
    ]
    parts = []
    for number, piece in enumerate(pieces):
        piece.write_parquet(paths.features_part_file(fold, number))
        parts.append(describe_part(paths.features_part_file(fold, number)))
    record = FeatureFiles(
        fold, tuple(wide.columns), CATEGORICAL, tuple(parts), wide.height, N_QUERIES, {}
    )
    write_feature_files(directory, record)
    paths.labels_file(fold).parent.mkdir(parents=True, exist_ok=True)
    labels.write_parquet(paths.labels_file(fold))


def write_sample(paths: ProjectPaths, fold: str) -> None:
    """Draw the fold's M7 sample file with the M7 code (ratio 40)."""
    job = SampleJob(
        fold=MANIFEST.get(fold),
        features_dir=paths.features_dir(fold),
        labels_file=paths.labels_file(fold),
        experiment=build_experiment("full", LOCK),
        wide_columns=wide_table_columns(LOCK),
        sample_files={RATIO: paths.ranker_sample_file("full", fold, RATIO)},
        allowed_phases=("F", "V"),
        git_commit="m7",
    )
    sample_fold(job, RankerSamplingConfig(seed=42, ratios=(RATIO,), row_group_rows=100))


def write_lightgbm_record(paths: ProjectPaths) -> None:
    """A stand-in for the record of LightGBM's Dataset of the target week."""
    sha256 = {
        fold: file_sha256(paths.ranker_sample_file("full", fold, RATIO)) for fold in TRAIN_FOLDS
    }
    directory = paths.ranker_dataset_dir("full", TARGET, RATIO)
    directory.mkdir(parents=True)
    (directory / "dataset.json").write_text(json.dumps({"sample_sha256": sha256}))


def write_category_codes(paths: ProjectPaths) -> None:
    codes = {name: list(range(CATEGORY_SIZE)) for name in CATEGORICAL}
    paths.category_codes_file.parent.mkdir(parents=True, exist_ok=True)
    paths.category_codes_file.write_text(json.dumps({"format_version": 1, "codes": codes}))


def build_workspace(root: Path, target_seed: int = 100, **training: object) -> TrialInputs:
    """Write the tiny workspace under `root` and return the inputs of its trials.

    Args:
        root: An empty folder (tmp_path).
        target_seed: Seed of the TARGET week's values; the training folds never change.
        **training: Values of the `training` section to replace (e.g. loss, epochs).
    """
    paths = build_paths(BASE, root)
    for seed, fold in enumerate(TRAIN_FOLDS):
        write_fold(paths, fold, *make_wide(fold, seed))
        write_sample(paths, fold)
    write_fold(paths, TARGET, *make_wide(TARGET, target_seed))
    write_lightgbm_record(paths)
    write_category_codes(paths)
    return TrialInputs(paths, BASE, tiny_config(**training), MANIFEST, LOCK)
