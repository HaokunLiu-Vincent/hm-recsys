"""Tests for deepfm.rescore: a finished trial's Top-K is reproduced by a new scoring."""

import dataclasses
from pathlib import Path

import polars as pl
import pytest
import torch
from deepfm_example import N_QUERIES, TARGET, build_workspace

from hmrec.deepfm.records import deepfm_config_id
from hmrec.deepfm.rescore import compare_top_k, rescore_trial
from hmrec.deepfm.trial import AREA_V_PHASE, TrialInputs, TrialSpec, run_trial

SPEC = TrialSpec(AREA_V_PHASE, TARGET, 42, 2)


def _table(rows: list[tuple[int, int, int, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "query_id": pl.Int32,
            "rank": pl.Int32,
            "article_idx": pl.Int32,
            "score": pl.Float64,
        },
        orient="row",
    )


def test_compare_top_k_counts_rows_that_are_not_reproduced() -> None:
    saved = _table([(1, 1, 10, 0.5), (1, 2, 11, 0.25), (2, 1, 12, 0.75)])
    assert compare_top_k(saved, saved) == (0, 0.0)
    # Another article at a rank, and a score that moved by 0.125.
    other_article = _table([(1, 1, 10, 0.5), (1, 2, 99, 0.25), (2, 1, 12, 0.75)])
    assert compare_top_k(saved, other_article) == (1, 0.0)
    moved = _table([(1, 1, 10, 0.5), (1, 2, 11, 0.375), (2, 1, 12, 0.75)])
    assert compare_top_k(saved, moved) == (1, 0.125)
    # A saved row without a partner counts as different.
    assert compare_top_k(saved, saved.head(2)) == (1, 0.0)
    assert compare_top_k(saved, saved.clear()) == (3, 0.0)


def _standardised(inputs: TrialInputs, rows_per_block: int) -> TrialInputs:
    model = dataclasses.replace(inputs.config.model, numeric_input="standardised")
    config = dataclasses.replace(inputs.config, model=model, scoring_rows_per_block=rows_per_block)
    return dataclasses.replace(inputs, config=config)


@pytest.mark.parametrize("standardised", [False, True])
def test_a_finished_trial_is_reproduced_whatever_the_block_size(
    tmp_path: Path, standardised: bool
) -> None:
    torch.set_num_threads(1)
    inputs = build_workspace(tmp_path)
    if standardised:
        inputs = _standardised(inputs, 1_000_000)  # one pass: the scoring of before the change
    run_trial(inputs, SPEC)
    for rows_per_block in (1_000_000, 7, 1):
        again = dataclasses.replace(
            inputs, config=dataclasses.replace(inputs.config, scoring_rows_per_block=rows_per_block)
        )
        # The block size is not part of the configuration id: the same trial is found.
        assert deepfm_config_id(again.config) == deepfm_config_id(inputs.config)
        result = rescore_trial(again, SPEC)
        assert result.identical and result.n_different_rows == 0
        assert result.max_abs_score_difference == 0.0
        assert result.n_scored_rows == N_QUERIES * 12
        assert result.n_saved_rows == result.n_new_rows == N_QUERIES * 12


def test_a_changed_prediction_is_found_and_unfinished_or_t_trials_are_refused(
    tmp_path: Path,
) -> None:
    inputs = build_workspace(tmp_path)
    with pytest.raises(FileNotFoundError, match="no finished trial"):
        rescore_trial(inputs, SPEC)
    run_trial(inputs, SPEC)
    directory = inputs.paths.deepfm_trial_dir(
        SPEC.area, deepfm_config_id(inputs.config), SPEC.seed, SPEC.target_fold
    )
    saved = pl.read_parquet(directory / "predictions.parquet")
    # Move one saved score by the smallest step a float32 logit near 1 can take.
    changed = saved.with_columns(
        pl.when(pl.int_range(pl.len()) == 5)
        .then(pl.col("score") + 2.0**-23)
        .otherwise(pl.col("score"))
        .alias("score")
    )
    changed.write_parquet(directory / "predictions.parquet")
    result = rescore_trial(inputs, SPEC)
    assert not result.identical and result.n_different_rows == 1
    assert result.max_abs_score_difference == pytest.approx(2.0**-23)
    # The fold check comes first: a T fold is refused before any file is opened.
    with pytest.raises(ValueError, match="T1"):
        rescore_trial(inputs, dataclasses.replace(SPEC, target_fold="T1"))
