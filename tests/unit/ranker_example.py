"""Tiny wide tables, labels and configs shared by the Ranker tests (not a test module itself).

Nothing here reads project data. The feature list is the real locked one
(configs/locks/features.lock.yaml), so the tables have the real 73 columns.

Two kinds of data:
    wide_table(rows)        hand-written rows; every feature not given is missing
                            (NaN for numbers, code 0 for categories).
    write_learnable_fold    a fold of random queries with a clear signal, for tests
                            that train a real (tiny) LightGBM model:
                              - every query has 12 candidates;
                              - a candidate is returned by a random non-empty set of
                                the six channels, with a random rank in each;
                              - `ui_decay_ptype` is a random number, and the candidate
                                with the LARGEST value of each query is its only label.
                            So a model that learns "higher ui_decay_ptype is better"
                            ranks the label first, while rrf_score knows nothing about it.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from hmrec.config import (
    BaseConfig,
    ChunkingConfig,
    MemoryConfig,
    PathsConfig,
    RankerConfig,
    load_yaml,
    parse_ranker_config,
)
from hmrec.data.folds import load_fold_manifest
from hmrec.feature_files import FeatureFiles, describe_part, write_feature_files
from hmrec.fusion.rrf import best_rank_expr, n_channels_expr, rrf_score_expr
from hmrec.fusion.union import RANK_COLUMNS
from hmrec.hpo.v_trial import TrialInputs
from hmrec.locks import load_features_lock
from hmrec.paths import ProjectPaths, build_paths

REPO = Path(__file__).resolve().parents[2]
LOCK = load_features_lock(REPO / "configs" / "locks" / "features.lock.yaml")
MANIFEST = load_fold_manifest(REPO / "configs" / "fold_manifest.yaml")
FEATURE_NAMES = tuple(feature.name for feature in LOCK.features)
WIDE_COLUMNS = ("query_id", "article_idx", *FEATURE_NAMES)
METRICS = ("map@12", "recall@12", "ndcg@12")
SIGNAL_FEATURE = "ui_decay_ptype"
CANDIDATES_PER_QUERY = 12
NAN = float("nan")

WIDE_SCHEMA = pl.Schema(
    {
        "query_id": pl.Int32,
        "article_idx": pl.Int32,
        **{
            feature.name: pl.Int16 if feature.dtype == "int16" else pl.Float32
            for feature in LOCK.features
        },
    }
)


def wide_table(rows: Sequence[Mapping[str, Any]]) -> pl.DataFrame:
    """A wide table from hand-written rows; features that are not given are missing."""
    columns: dict[str, list[Any]] = {}
    for name, dtype in WIDE_SCHEMA.items():
        missing = 0 if dtype == pl.Int16 else NAN
        columns[name] = [row.get(name, missing) for row in rows]
    return pl.DataFrame(columns, schema=WIDE_SCHEMA).sort("query_id", "article_idx")


def with_aggregates(wide: pl.DataFrame) -> pl.DataFrame:
    """Fill n_channels, best_rank and rrf_score from the six rank columns (as the wide table)."""
    return wide.with_columns(
        n_channels_expr(RANK_COLUMNS),
        best_rank_expr(RANK_COLUMNS),
        rrf_score_expr(RANK_COLUMNS, 60),
    )


def write_features(directory: Path, fold: str, parts: Sequence[pl.DataFrame]) -> None:
    """Write part files and features.json, like hmrec.features.build does."""
    directory.mkdir(parents=True, exist_ok=True)
    described = []
    for index, part in enumerate(parts):
        path = directory / f"part-{index:05d}.parquet"
        part.write_parquet(path)
        described.append(describe_part(path))
    write_feature_files(
        directory,
        FeatureFiles(
            fold=fold,
            columns=WIDE_COLUMNS,
            categorical_features=LOCK.categorical_features,
            parts=tuple(described),
            n_rows=sum(part.height for part in parts),
            n_queries=sum(part["query_id"].n_unique() for part in parts),
            details={},
        ),
    )


def write_labels(path: Path, fold: str, pairs: Sequence[tuple[int, int]]) -> None:
    """Write artifacts/labels/{fold}.parquet from (query_id, article_idx) pairs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "query_id": [pair[0] for pair in pairs],
            "fold": [fold] * len(pairs),
            "article_idx": [pair[1] for pair in pairs],
        },
        schema={"query_id": pl.Int32, "fold": pl.String, "article_idx": pl.Int32},
    ).write_parquet(path)


def learnable_fold(
    first_query_id: int, n_queries: int, seed: int
) -> tuple[pl.DataFrame, list[tuple[int, int]]]:
    """Random queries with a signal (module docstring): (wide table, label pairs)."""
    generator = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    labels: list[tuple[int, int]] = []
    for query_id in range(first_query_id, first_query_id + n_queries):
        signal = generator.random(CANDIDATES_PER_QUERY)
        labels.append((query_id, int(signal.argmax())))
        for article_idx in range(CANDIDATES_PER_QUERY):
            row: dict[str, Any] = {"query_id": query_id, "article_idx": article_idx}
            returned = generator.random(len(RANK_COLUMNS)) < 0.4
            if not returned.any():
                returned[generator.integers(len(RANK_COLUMNS))] = True
            for column, is_returned in zip(RANK_COLUMNS, returned, strict=True):
                if is_returned:
                    row[column] = float(generator.integers(1, 101))
            row[SIGNAL_FEATURE] = float(signal[article_idx])
            row["tt_dot"] = float(generator.random())
            row["product_type_no"] = int(generator.integers(0, 5))
            rows.append(row)
    return with_aggregates(wide_table(rows)), labels


def write_learnable_fold(
    paths: ProjectPaths, fold: str, first_query_id: int, n_queries: int, seed: int
) -> None:
    """Write a learnable fold's features (two part files) and labels under `paths`."""
    wide, labels = learnable_fold(first_query_id, n_queries, seed)
    split_at = first_query_id + n_queries // 2
    parts = [
        wide.filter(pl.col("query_id") < split_at),
        wide.filter(pl.col("query_id") >= split_at),
    ]
    write_features(paths.features_dir(fold), fold, parts)
    write_labels(paths.labels_file(fold), fold, labels)


def tiny_ranker_raw(**overrides: Any) -> dict[str, Any]:
    """The raw content of configs/ranker.yaml shrunk for tiny data: 20 trees, checkpoints every 10.

    `overrides` replace top-level keys of the raw config (e.g. histogram={...}).
    """
    raw = load_yaml(REPO / "configs" / "ranker.yaml")
    raw.update(
        {
            "num_trees": 20,
            "checkpoint_every": 10,
            "train_metric_every": 10,
            "scoring_query_chunk_size": 7,
            "sampling": {"seed": 42, "ratios": [2, 4], "row_group_rows": 64},
            "tuned_defaults": {
                "num_leaves": 4,
                "min_data_in_leaf": 5,
                "negatives_per_positive": 2,
            },
            "histogram": {
                "mode": "col_wise",
                "comparison_trees": 6,
                "warmup_trees": 2,
                "min_row_wise_speedup": 0.15,
            },
            "speed_test": {"target_fold": "V1", "num_trees": 20},
        }
    )
    raw["dataset_params"] = {**raw["dataset_params"], "min_data_in_bin": 1}
    raw["train_params"] = {**raw["train_params"], "min_data_per_group": 5}
    raw.update(overrides)
    return raw


def tiny_ranker_config(**overrides: Any) -> RankerConfig:
    """tiny_ranker_raw as a parsed RankerConfig."""
    return parse_ranker_config(tiny_ranker_raw(**overrides))


def tiny_base_config() -> BaseConfig:
    """A BaseConfig for tests: 2 threads; roots relative to the test's folder."""
    return BaseConfig(
        paths=PathsConfig(data_root="data", artifacts_root="artifacts", results_root="results"),
        seed=0,
        num_threads=2,
        memory=MemoryConfig(limit_gb=8.0, sample_interval_seconds=0.1),
        chunking=ChunkingConfig(
            query_chunk_size=10,
            item_chunk_size=10,
            dense_score_query_chunk_size=10,
            twotower_score_query_chunk_size=10,
            pair_chunk_size=10,
        ),
    )


def tiny_inputs(root: Path, **ranker_overrides: Any) -> TrialInputs:
    """TrialInputs whose artifacts and results live under `root` (a tmp folder)."""
    base = tiny_base_config()
    return TrialInputs(
        paths=build_paths(base, root),
        base=base,
        ranker=tiny_ranker_config(**ranker_overrides),
        manifest=MANIFEST,
        lock=LOCK,
        metrics=METRICS,
    )


# query ids of the learnable folds: far apart, growing with the fold's position.
LEARNABLE_FOLDS = {"F1": 1000, "F2": 2000, "F3": 3000, "F4": 4000, "V1": 5000}


def write_learnable_folds(paths: ProjectPaths, n_queries: int = 60) -> None:
    """Write F1 ... F4 and V1 as learnable folds (different seeds)."""
    for seed, (fold, first_query_id) in enumerate(LEARNABLE_FOLDS.items()):
        write_learnable_fold(paths, fold, first_query_id, n_queries, seed)


def build_workspace(root: Path, **ranker_overrides: Any) -> TrialInputs:
    """Learnable folds F1 ... F4 and V1 with their samples (ratios 2 and 4) under `root`.

    Uses the real sampling code, so tests that need training data get exactly
    what the pipeline would write.
    """
    # Imported here: ranker_example is also used by tests of lower-level modules.
    from hmrec.hpo.v_phase import sample_job
    from hmrec.ranker.sampling import sample_fold

    inputs = tiny_inputs(root, **ranker_overrides)
    write_learnable_folds(inputs.paths)
    for fold in ("F1", "F2", "F3", "F4"):
        job = sample_job(inputs, fold, "full", inputs.ranker.sampling.ratios, "test-commit")
        sample_fold(job, inputs.ranker.sampling)
    return inputs
