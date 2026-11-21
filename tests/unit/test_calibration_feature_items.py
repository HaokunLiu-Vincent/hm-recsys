"""Tests for hmrec.calibration.feature_items and the composed input paths of item E.

The feature build itself is the locked feature code (tested in test_feature_build.py)
and runs on the full data only in the calibration. Here: where item E reads its
six channels from, and the two counts it makes independently of the feature code.
"""

from dataclasses import replace
from pathlib import Path

import polars as pl
import pytest

from hmrec.calibration.context import CalibrationContext
from hmrec.calibration.feature_items import (
    deduplicated_candidate_count,
    input_paths,
    run_end_to_end_features,
    written_feature_keys,
)
from hmrec.config import load_base_config, load_calibration_config
from hmrec.locks import ALL_CHANNELS
from hmrec.paths import (
    build_paths,
    calibration_output_paths,
    end_to_end_input_paths,
    get_paths,
)
from hmrec.run_records import RunStamp

REPO_PATHS = get_paths()
BASE = load_base_config(REPO_PATHS.base_config_file)
CONFIG = load_calibration_config(REPO_PATHS.m10_calibration_config_file)
COMMIT = "0123456789abcdef0123456789abcdef01234567"


def _context(root: Path, run: str = "run3") -> CalibrationContext:
    paths = build_paths(BASE, root)
    return CalibrationContext(
        paths=paths,
        main_paths=calibration_output_paths(paths, "linux-arm64", run, "main"),
        recorded_model_paths=calibration_output_paths(paths, "linux-arm64", run, "recorded_model"),
        base=BASE,
        config=CONFIG,
        run=run,
        platform_name="linux-arm64",
        stamp=RunStamp(COMMIT, False),
        started=0.0,
    )


def _pairs(rows: list[tuple[int, int]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows, schema={"query_id": pl.Int32, "article_idx": pl.Int32}, orient="row"
    ).with_columns(pl.lit(1, dtype=pl.Int32).alias("rank"))


def test_r6_comes_from_the_run_of_item_e_and_r1_to_r5_from_the_configured_run(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path, run="run3")
    sources = input_paths(context)
    calibration = context.paths.calibration_artifacts_dir / "linux-arm64"
    run1, run3 = calibration / "run1" / "main", calibration / "run3" / "main"
    assert CONFIG.statistical_run == "run1"
    for channel in ALL_CHANNELS:
        tree = run3 if channel == "r6_twotower" else run1
        assert (
            sources.candidates_file("V4", channel)
            == tree / "candidates" / "V4" / f"{channel}.parquet"
        )
    # The side files of the feature code follow their channel.
    assert (
        sources.r1_bucket_scores_file("V4")
        == run1 / "candidates" / "V4" / "r1_bucket_scores.parquet"
    )
    assert sources.wals_vectors_dir("V4") == run1 / "models" / "wals" / "V4"
    assert sources.twotower_vectors_dir("V4") == run3 / "models" / "twotower" / "vectors" / "V4"
    # Processed tables and embeddings are the stage-1 inputs; nothing is read from a
    # stage-1 candidates folder.
    assert sources.transactions_file == context.paths.transactions_file
    assert sources.image_embeddings_by_idx_file == context.paths.image_embeddings_by_idx_file
    stage1_candidates = context.paths.candidates_dir("V4")
    assert all(
        stage1_candidates not in sources.candidates_file("V4", c).parents for c in ALL_CHANNELS
    )


def test_one_run_for_everything_reads_one_tree(tmp_path: Path) -> None:
    # If A5 had applied (no repeatable setting), item E would read R6 of part 2 as well.
    paths = build_paths(BASE, tmp_path)
    sources = end_to_end_input_paths(paths, "linux-arm64", "run1", "run1")
    assert sources.statistical_artifacts_root == sources.artifacts_root
    assert (
        sources.candidates_file("V4", "r6_twotower").parent
        == sources.candidates_file("V4", "r1_popularity").parent
    )


def test_deduplicated_candidate_count() -> None:
    # Query 1: articles 10, 11 (R1) and 11, 12 (R2) -> 3 distinct. Query 2: article 10 from
    # both channels -> 1 distinct. The same article for two queries is two pairs.
    candidates = {
        "r1_popularity": _pairs([(1, 10), (1, 11), (2, 10)]),
        "r2_repurchase": _pairs([(1, 11), (1, 12), (2, 10)]),
    }
    assert deduplicated_candidate_count(candidates) == 4


def test_written_feature_keys_counts_rows_and_distinct_keys_of_the_files(tmp_path: Path) -> None:
    context = _context(tmp_path)
    directory = context.main_paths.features_dir("V4")
    directory.mkdir(parents=True)
    _pairs([(1, 10), (1, 11)]).write_parquet(directory / "part-00000.parquet")
    _pairs([(2, 10), (2, 12)]).write_parquet(directory / "part-00001.parquet")
    assert written_feature_keys(context) == (4, 4)
    # A key that repeats in another part is seen: 5 rows, 4 distinct keys.
    _pairs([(1, 10)]).write_parquet(directory / "part-00002.parquet")
    assert written_feature_keys(context) == (5, 4)


def test_item_e_stops_at_an_input_with_another_sha256_before_anything_is_built(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    processed = context.paths.processed_dir
    processed.mkdir(parents=True)
    for name in CONFIG.processed_sha256:
        (processed / name).write_bytes(b"not the recorded file")
    with pytest.raises(ValueError, match="sha256"):
        run_end_to_end_features(replace(context, run="run3"))
    assert not context.main_paths.features_dir("V4").exists()
