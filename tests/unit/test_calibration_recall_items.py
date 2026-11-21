"""Tests for hmrec.calibration.recall_items (torch side).

The R2 item is run end to end on three purchase lines: a "stage-1" result is
first produced with the stage-1 function into a temporary stage-1 tree, then the
calibration item generates the same channel into the calibration tree and
compares. On the same machine the two must be identical, which checks that the
item adds no computation of its own.

The R6 items and the R5 fit run only on the full data (the calibration itself);
here their building blocks are tested: the measurement rows, the comparison of
two checkpoints, and the refusal to reuse an output.
"""

import csv
import hashlib
from dataclasses import replace
from pathlib import Path

import polars as pl
import pytest
import torch

import hmrec.calibration.context as context_module
import hmrec.calibration.recall_items as recall_items
from conftest import make_queries, make_recall_inputs
from hmrec.calibration.compare import compare_candidates
from hmrec.calibration.context import CalibrationContext
from hmrec.calibration.recall_items import (
    RecallSetup,
    candidate_measurements,
    fine_tune_measurements,
    parameter_difference,
    require_new_output,
    resource_measurements,
    run_r6_predict,
    run_statistical_channel,
)
from hmrec.calibration.records import Measurement
from hmrec.config import (
    load_base_config,
    load_calibration_config,
    load_catalog_config,
    load_preprocess_config,
)
from hmrec.data.folds import load_fold_manifest
from hmrec.memory import ProcessPeakMemory
from hmrec.paths import ProjectPaths, build_paths, calibration_output_paths, get_paths
from hmrec.platform_fields import PlatformFields
from hmrec.recall.chain_records import ChainRecord
from hmrec.recall.generate import FoldJob, generate_statistical_channel
from hmrec.recall.inputs import SharedTables
from hmrec.recall.locked import load_locked_recall, post_lock_folds
from hmrec.run_records import RunStamp

REPO_PATHS = get_paths()
BASE = load_base_config(REPO_PATHS.base_config_file)
CONFIG = load_calibration_config(REPO_PATHS.m10_calibration_config_file)
PREPROCESS = load_preprocess_config(REPO_PATHS.preprocess_config_file)
MANIFEST = load_fold_manifest(REPO_PATHS.fold_manifest_file)
LOCKED = load_locked_recall(REPO_PATHS)
COMMIT = "0123456789abcdef0123456789abcdef01234567"
PLATFORM = PlatformFields(
    "linux", "aarch64", "not loaded", "unknown", "not loaded", COMMIT, "c" * 64
)
R2 = "r2_repurchase"
ANCHOR_DAY = 400
# Customer 0 bought articles 30 and 40, customer 1 bought article 31 (before the anchor).
PURCHASES = [(0, 30, 370), (0, 40, 399), (1, 31, 395)]
# Query 1 = customer 0, query 2 = customer 1.
QUERIES = make_queries([(1, 0), (2, 1)], fold="V4")
GB = 1024**3


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _setup() -> RecallSetup:
    """The real configs and locks; the tables are not read (fold_job is replaced)."""
    empty = pl.DataFrame()
    return RecallSetup(
        preprocess=PREPROCESS,
        catalog=load_catalog_config(REPO_PATHS.catalog_config_file),
        manifest=MANIFEST,
        fold_names=tuple(fold.name for fold in post_lock_folds(MANIFEST)),
        locked=LOCKED,
        tables=SharedTables(empty.lazy(), empty, empty),
    )


def _job(output_paths: ProjectPaths) -> FoldJob:
    """The small R2 job of fold V4, writing where `output_paths` says."""
    setup = _setup()
    return FoldJob(
        paths=output_paths,
        base=BASE,
        preprocess=PREPROCESS,
        locked=LOCKED,
        fold=MANIFEST.get("V4"),
        fold_names=setup.fold_names,
        queries=QUERIES,
        inputs=make_recall_inputs(PURCHASES, ANCHOR_DAY),
    )


def _write_table(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> CalibrationContext:
    """A temporary project with a "stage-1" R2 result of V4, its labels and its records.

    Labels: query 1 bought {40, 99}, query 2 bought {31}. R2 returns {30, 40} for
    query 1 and {31} for query 2, so Recall@30 = (1/2 + 1/1) / 2 = 0.75.
    """
    paths = build_paths(BASE, tmp_path)
    generate_statistical_channel(_job(paths), R2)  # the stage-1 output
    labels = pl.DataFrame(
        {"query_id": [1, 1, 2], "fold": ["V4"] * 3, "article_idx": [40, 99, 31]},
        schema={"query_id": pl.Int32, "fold": pl.String, "article_idx": pl.Int32},
    )
    paths.labels_file("V4").parent.mkdir(parents=True)
    labels.write_parquet(paths.labels_file("V4"))
    _write_table(
        paths.m6_candidates_results_file,
        [{"fold": "V4", "scope": R2, "n_rows": "3", "recall": "0.75"}],
    )
    _write_table(
        paths.m6_runs_file,
        [{"fold": "V4", "step": R2, "generate_seconds": "0.25", "process_max_rss_gb": "1.0"}],
    )
    config = replace(
        CONFIG,
        artifact_sha256={"labels/V4.parquet": _sha256(paths.labels_file("V4"))},
        reference_sha256={f"candidates/V4/{R2}.parquet": _sha256(paths.candidates_file("V4", R2))},
    )
    context = CalibrationContext(
        paths=paths,
        main_paths=calibration_output_paths(paths, "linux-arm64", "run1", "main"),
        recorded_model_paths=calibration_output_paths(
            paths, "linux-arm64", "run1", "recorded_model"
        ),
        base=BASE,
        config=config,
        run="run1",
        platform_name="linux-arm64",
        stamp=RunStamp(COMMIT, False),
        started=0.0,
    )
    monkeypatch.setattr(context_module, "read_platform_fields", lambda repo_root: PLATFORM)
    monkeypatch.setattr(recall_items, "load_recall_setup", lambda context: _setup())
    monkeypatch.setattr(
        recall_items, "fold_job", lambda context, setup, fold_name, output: _job(output)
    )
    return context


def _by_quantity(measurements: list[Measurement]) -> dict[str, Measurement]:
    return {measurement.quantity: measurement for measurement in measurements}


def test_the_r2_item_reproduces_the_stage_1_result_and_writes_only_the_calibration_tree(
    context: CalibrationContext,
) -> None:
    stage1_file = context.paths.candidates_file("V4", R2)
    stage1_bytes = stage1_file.read_bytes()
    outcome = run_statistical_channel(context, R2)
    assert (outcome.item, outcome.scope) == ("A", R2)
    rows = _by_quantity(list(outcome.measurements))
    assert (rows["rows"].measured, rows["rows"].stage1) == (3, 3)
    assert rows["rank_rows_identical"].measured is True
    assert rows["rank_rows_identical"].within_tolerance is True  # R1-R4: "identical"
    assert rows["differing_rank_rows"].measured == 0
    assert rows["scores_bit_identical"].measured is True
    assert rows["max_abs_score_difference"].measured == 0.0
    assert rows["file_sha256_equal"].measured is True
    # The hand-computed recall, equal to the record: inside the tolerance.
    assert (rows["recall"].measured, rows["recall"].stage1) == (0.75, 0.75)
    assert rows["recall"].within_tolerance is True
    assert (rows["seconds"].stage1, rows["max_rss_gb"].stage1) == (0.25, 1.0)
    # The output lies in the calibration tree; the stage-1 file was not touched.
    written = context.main_paths.candidates_file("V4", R2)
    assert context.paths.calibration_artifacts_dir in written.parents
    assert pl.read_parquet(written).equals(pl.read_parquet(stage1_file))
    assert stage1_file.read_bytes() == stage1_bytes


def test_a_second_attempt_never_reuses_or_replaces_an_output(context: CalibrationContext) -> None:
    run_statistical_channel(context, R2)
    with pytest.raises(FileExistsError, match="never reuses or replaces an output"):
        run_statistical_channel(context, R2)


def test_a_changed_reference_or_label_file_stops_the_comparison(
    context: CalibrationContext,
) -> None:
    changed = replace(context.config, reference_sha256={f"candidates/V4/{R2}.parquet": "0" * 64})
    with pytest.raises(ValueError, match="sha256"):
        run_statistical_channel(replace(context, config=changed), R2)


def test_unknown_channels_and_starts_are_refused(context: CalibrationContext) -> None:
    with pytest.raises(ValueError, match="expected one of"):
        run_statistical_channel(context, "r6_twotower")
    with pytest.raises(ValueError, match="--start must be one of"):
        run_r6_predict(context, "base")


def test_require_new_output(tmp_path: Path) -> None:
    require_new_output(tmp_path / "missing.parquet")
    (tmp_path / "there.parquet").write_bytes(b"x")
    with pytest.raises(FileExistsError, match="never reuses or replaces"):
        require_new_output(tmp_path / "there.parquet")


def _candidates(rows: list[tuple[int, int, int, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "query_id": pl.Int32,
            "article_idx": pl.Int32,
            "rank": pl.Int32,
            "score": pl.Float32,
        },
        orient="row",
    )


def test_the_tolerance_is_identical_for_r1_to_r4_and_a_pair_share_for_r5_and_r6(
    context: CalibrationContext,
) -> None:
    recorded = _candidates([(1, 10, 1, 0.5), (1, 11, 2, 0.25), (2, 10, 1, 1.0), (2, 12, 2, 0.5)])
    # One of four recorded pairs is replaced: 75% of the recorded pairs are present.
    measured = _candidates([(1, 10, 1, 0.5), (1, 11, 2, 0.25), (2, 10, 1, 1.0), (2, 13, 2, 0.5)])
    comparison = compare_candidates(recorded, measured)
    statistical = _by_quantity(candidate_measurements(context, "A", "r4_itemcf", comparison))
    assert statistical["rank_rows_identical"].tolerance.startswith("identical")
    assert statistical["rank_rows_identical"].within_tolerance is False
    assert statistical["recorded_pairs_present_share"].within_tolerance is None
    for item, channel in (("B", "r5_wals"), ("C2", "r6_twotower")):
        learned = _by_quantity(candidate_measurements(context, item, channel, comparison))
        assert learned["rank_rows_identical"].within_tolerance is None
        share = learned["recorded_pairs_present_share"]
        assert (share.measured, share.tolerance, share.within_tolerance) == (0.75, ">= 0.99", False)
    # All pairs present: inside.
    same = _by_quantity(
        candidate_measurements(context, "B", "r5_wals", compare_candidates(recorded, recorded))
    )
    assert same["recorded_pairs_present_share"].within_tolerance is True


def test_resource_measurements_carry_the_stage_1_values_and_a_note() -> None:
    stage1 = {"generate_seconds": "18.25", "process_max_rss_gb": "4.5"}
    seconds, peak = resource_measurements(
        "C1", "r6_twotower", 20.0, ProcessPeakMemory(5 * GB, 0), stage1, "prediction only"
    )
    assert (seconds.quantity, seconds.measured, seconds.stage1) == ("seconds", 20.0, 18.25)
    assert (peak.quantity, peak.measured, peak.stage1) == ("max_rss_gb", 5.0, 4.5)
    assert seconds.source == "results/m6_runs.csv (prediction only)"
    assert seconds.within_tolerance is None  # time and memory have no tolerance


def _checkpoint(path: Path, weight: list[float], bias: list[float]) -> None:
    state = {"layer.weight": torch.tensor(weight), "layer.bias": torch.tensor(bias)}
    torch.save({"state_dict": state, "metadata": {"seconds": 1.0}}, path)


def test_parameter_difference_of_two_checkpoints(tmp_path: Path) -> None:
    _checkpoint(tmp_path / "recorded.pt", [1.0, 2.0, 3.0], [0.5])
    _checkpoint(tmp_path / "same.pt", [1.0, 2.0, 3.0], [0.5])
    # Two of four values differ: by 0.25 and by 1.0.
    _checkpoint(tmp_path / "other.pt", [1.0, 2.25, 3.0], [1.5])
    same = parameter_difference(tmp_path / "recorded.pt", tmp_path / "same.pt")
    assert (same.max_abs_difference, same.n_values, same.n_different_values) == (0.0, 4, 0)
    other = parameter_difference(tmp_path / "recorded.pt", tmp_path / "other.pt")
    assert (other.max_abs_difference, other.n_values, other.n_different_values) == (1.0, 4, 2)
    torch.save(
        {"state_dict": {"layer.weight": torch.tensor([1.0])}, "metadata": {}}, tmp_path / "x.pt"
    )
    with pytest.raises(ValueError, match="hold different tensors"):
        parameter_difference(tmp_path / "recorded.pt", tmp_path / "x.pt")


def _chain_record(sha256: str, train_loss: float, seconds: float) -> ChainRecord:
    return ChainRecord(
        fold="V3",
        anchor="2020-08-12",
        start_model="V2.pt",
        start_sha256="s" * 64,
        model="V3.pt",
        sha256=sha256,
        n_new_rows=10,
        n_replay_rows=10,
        epochs=2,
        steps=4,
        n_train_samples=20,
        n_alarm_samples=3,
        mean_train_loss=train_loss,
        alarm_loss=5.5,
        fine_tune_seconds=seconds,
        sampled_peak_rss_gb=1.0,
        git_commit=COMMIT,
        git_dirty=False,
        finished_at="2026-11-22T10:00:00",
    )


def test_fine_tune_measurements() -> None:
    recorded = _chain_record("a" * 64, 5.5, 50.0)
    measured = _chain_record("b" * 64, 5.75, 80.0)
    parameters = recall_items.ParameterDifference(0.125, 1000, 900)
    rows = _by_quantity(fine_tune_measurements(measured, recorded, parameters))
    assert (rows["mean_train_loss"].measured, rows["mean_train_loss"].stage1) == (5.75, 5.5)
    assert rows["start_sha256_equal"].measured is True
    assert rows["model_sha256_equal"].measured is False
    assert rows["max_abs_parameter_difference"].measured == 0.125
    assert (rows["different_parameter_values"].measured, rows["parameter_values"].measured) == (
        900,
        1000,
    )
    assert (rows["fine_tune_seconds"].measured, rows["fine_tune_seconds"].stage1) == (80.0, 50.0)
    assert all(row.item == "C2" and row.scope == "fine_tune" for row in rows.values())
    # The fine-tuning step has no tolerance of its own in docs/mlops.md 8.3.
    assert all(row.within_tolerance is None for row in rows.values())
