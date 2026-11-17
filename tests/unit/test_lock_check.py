"""Tests for hpo.lock_check: the locked channel must reproduce the recorded score exactly.

Tiny fold D1 (anchor day 400) with the REAL locked R2 config (decayed frequency,
lookback 365 days, quota 30). Purchase lines (customer, article, day):
    customer 0: (0, 30, 370), (0, 40, 399)        customer 1: (1, 31, 395)
Queries: q1 = customer 0, q2 = customer 1. Labels: q1 -> {31, 40}, q2 -> {99}.
R2 returns what each customer bought: q1 -> {30, 40}, q2 -> {31}.
    recall@30: q1 = |{40}| / 2 = 0.5, q2 = 0 / 1 = 0  ->  fold mean 0.25
"""

import dataclasses
from pathlib import Path

import polars as pl
import pytest
from conftest import make_queries, make_recall_inputs
from test_union_precheck import D1_CONTEXT

from hmrec.config import load_base_config
from hmrec.data.folds import load_fold_manifest
from hmrec.eval.report import FoldEvaluator
from hmrec.hpo.grid import load_phase_grid
from hmrec.hpo.lock_check import (
    LockCheckResult,
    check_locked_channel,
    read_recorded_trial,
    results_file_of,
)
from hmrec.hpo.runner import FoldContext
from hmrec.locks import ALL_CHANNELS
from hmrec.paths import get_paths
from hmrec.recall.build import build_statistical_channel
from hmrec.recall.locked import load_locked_recall

REPO = Path(__file__).resolve().parents[2]
PATHS = get_paths()
BASE = load_base_config(REPO / "configs" / "base.yaml")
GRID = load_phase_grid(PATHS.d_phase_grid_file)
D_FOLDS = [fold.name for fold in load_fold_manifest(PATHS.fold_manifest_file).by_phase("D")]
LOCKED = load_locked_recall(PATHS)
R2 = LOCKED.channel("r2_repurchase")
HAND_COMPUTED_RECALL = 0.25


def _tiny_fold() -> FoldContext:
    inputs = make_recall_inputs([(0, 30, 370), (0, 40, 399), (1, 31, 395)], 400)
    labels = pl.DataFrame(
        {"query_id": [1, 1, 2], "fold": ["D1"] * 3, "article_idx": [31, 40, 99]},
        schema={"query_id": pl.Int32, "fold": pl.String, "article_idx": pl.Int32},
    )
    queries = make_queries([(1, 0), (2, 1)])
    return FoldContext(D1_CONTEXT.fold, queries, inputs, FoldEvaluator("D1", labels))


def _check(path: Path, recorded_score: float) -> LockCheckResult:
    channel = build_statistical_channel("r2_repurchase", R2.config, BASE)
    return check_locked_channel(channel, R2, _tiny_fold(), recorded_score, path, BASE.memory)


def test_identical_score_passes_and_candidates_are_written(tmp_path: Path) -> None:
    path = tmp_path / "lock_check" / "D1" / "r2_repurchase.parquet"
    result = _check(path, HAND_COMPUTED_RECALL)
    assert result.score == HAND_COMPUTED_RECALL
    assert (result.channel, result.fold, result.trial_id) == ("r2_repurchase", "D1", R2.trial_id)
    assert (result.metric, result.n_rows) == ("recall@30", 3)
    written = pl.read_parquet(path).sort("query_id", "article_idx")
    assert written.select("query_id", "article_idx").rows() == [(1, 30), (1, 40), (2, 31)]


def test_any_difference_from_the_record_fails(tmp_path: Path) -> None:
    path = tmp_path / "r2.parquet"
    # One unit in the last place away from the hand-computed 0.25 is already a failure.
    almost = HAND_COMPUTED_RECALL + 2.0**-54
    assert almost != HAND_COMPUTED_RECALL
    with pytest.raises(ValueError, match="but trial f562d598b0d6 recorded"):
        _check(path, almost)
    # The candidates were written before the comparison, so they can be inspected.
    assert path.is_file()


def test_existing_candidates_are_never_replaced(tmp_path: Path) -> None:
    path = tmp_path / "r2.parquet"
    _check(path, HAND_COMPUTED_RECALL)
    with pytest.raises(FileExistsError, match="already exists"):
        _check(path, HAND_COMPUTED_RECALL)


# ---------------------------------------------------------------- recorded trials (real files)


def test_results_file_of_knows_only_the_two_files_with_locked_trials() -> None:
    assert results_file_of(PATHS, "results/d_phase.csv") == PATHS.d_phase_results_file
    assert results_file_of(PATHS, "results/m4_trials.csv") == PATHS.m4_trials_file
    with pytest.raises(ValueError, match="unknown results file"):
        results_file_of(PATHS, "results/v_phase.csv")


@pytest.mark.parametrize("channel", ALL_CHANNELS)
def test_every_locked_trial_is_found_in_its_results_file(channel: str) -> None:
    locked = LOCKED.channel(channel)
    record = read_recorded_trial(PATHS, locked, D_FOLDS, GRID)
    assert (record.channel, record.trial_id) == (channel, locked.trial_id)


def test_a_lock_that_disagrees_with_the_results_file_is_refused() -> None:
    stale_score = dataclasses.replace(R2.d_phase_record, pooled=R2.d_phase_record.pooled + 1e-9)
    with pytest.raises(ValueError, match="same config: True, same scores: False"):
        read_recorded_trial(
            PATHS, dataclasses.replace(R2, d_phase_record=stale_score), D_FOLDS, GRID
        )
    other_config = {**R2.config, "tau_days": 14.0}
    with pytest.raises(ValueError, match="same config: False, same scores: True"):
        read_recorded_trial(PATHS, dataclasses.replace(R2, config=other_config), D_FOLDS, GRID)
    with pytest.raises(ValueError, match="expected exactly one trial 000000000000"):
        read_recorded_trial(PATHS, dataclasses.replace(R2, trial_id="000000000000"), D_FOLDS, GRID)
    # R3's locked trial is in results/m4_trials.csv, not in results/d_phase.csv.
    r3 = LOCKED.channel("r3_sibling")
    wrong_source = dataclasses.replace(r3.d_phase_record, source="results/d_phase.csv")
    with pytest.raises(ValueError, match="found 0"):
        read_recorded_trial(
            PATHS, dataclasses.replace(r3, d_phase_record=wrong_source), D_FOLDS, GRID
        )
