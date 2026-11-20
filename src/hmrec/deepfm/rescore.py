"""Score a finished trial's target week again and compare with what the trial saved (§16.10).

A check, not an evaluation. On 2026-11-20 the scoring was changed to use less
memory (hmrec.deepfm.model: the field embeddings are built in blocks). The change
must not move any score. This module takes a trial that was finished BEFORE the
change, loads its saved model and its saved transform, scores all candidates of
its target week with the CURRENT code and compares the Top-K with the trial's
predictions.parquet: the same articles at the same ranks with the same scores,
bit for bit.

No label is read, nothing is trained and nothing is written. The fold check of a
trial runs first, so a T fold is refused before any file is opened.
"""

from dataclasses import dataclass

import numpy as np
import polars as pl

from hmrec.deepfm.inputs import load_transform
from hmrec.deepfm.records import deepfm_config_id
from hmrec.deepfm.score import ARTICLE_IDX, QUERY_ID, RANK, SCORE
from hmrec.deepfm.trial import (
    DONE_FILE,
    EPOCH,
    PREDICTIONS_FILE,
    TRANSFORM_FILE,
    TrialInputs,
    TrialSpec,
    check_spec,
    load_final_model,
    score_target_fold,
)
from hmrec.logging_utils import get_logger

LOGGER = get_logger("deepfm.rescore")
KEYS = [QUERY_ID, RANK]


@dataclass(frozen=True)
class RescoreResult:
    """The comparison of a new scoring with a trial's saved predictions.

    Attributes:
        config_id: The trial's configuration.
        n_scored_rows: Candidate rows scored again.
        n_saved_rows: Top-K rows the trial saved.
        n_new_rows: Top-K rows of the new scoring.
        n_different_rows: Saved rows whose article or score differs in the new
            scoring (all rows if the two tables do not have the same keys).
        max_abs_score_difference: Largest absolute score difference over the
            rows both tables have (0.0 if every score is the same).
    """

    config_id: str
    n_scored_rows: int
    n_saved_rows: int
    n_new_rows: int
    n_different_rows: int
    max_abs_score_difference: float

    @property
    def identical(self) -> bool:
        """True if the new Top-K is exactly the saved one."""
        return self.n_different_rows == 0 and self.n_saved_rows == self.n_new_rows


def compare_top_k(saved: pl.DataFrame, new: pl.DataFrame) -> tuple[int, float]:
    """How a new Top-K table differs from a saved one.

    Args:
        saved: One row per (query_id, rank). Columns: query_id, rank,
            article_idx, score (float64).
        new: The same columns from the new scoring.

    Returns:
        (rows of `saved` that have no equal row in `new`, the largest absolute
        score difference over the (query_id, rank) both tables have).
    """
    joined = saved.select(*KEYS, ARTICLE_IDX, SCORE).join(
        new.select(
            *KEYS, pl.col(ARTICLE_IDX).alias("new_article"), pl.col(SCORE).alias("new_score")
        ),
        on=KEYS,
        how="left",
    )
    same = (pl.col(ARTICLE_IDX) == pl.col("new_article")) & (pl.col(SCORE) == pl.col("new_score"))
    # A row without a partner has nulls: it counts as different.
    n_different = joined.height - int(joined.select(same.fill_null(False).sum()).item())
    both = joined.filter(pl.col("new_score").is_not_null())
    if both.height == 0:
        return n_different, 0.0
    difference = np.abs(both[SCORE].to_numpy() - both["new_score"].to_numpy())
    return n_different, float(difference.max())


def rescore_trial(inputs: TrialInputs, spec: TrialSpec) -> RescoreResult:
    """Score a finished trial's target week again with the current code and compare.

    Args:
        inputs: The trial's configuration (the same as when it ran).
        spec: The trial.

    Raises:
        ValueError: If the fold check fails (e.g. a T fold).
        FileNotFoundError: If the trial is not finished.
    """
    check_spec(inputs, spec)  # before anything is read
    config_id = deepfm_config_id(inputs.config)
    directory = inputs.paths.deepfm_trial_dir(spec.area, config_id, spec.seed, spec.target_fold)
    if not (directory / DONE_FILE).is_file():
        raise FileNotFoundError(f"{directory} holds no finished trial")
    transform = load_transform(directory / TRANSFORM_FILE)
    model = load_final_model(directory, transform, inputs, spec.epochs)
    # reader=None: the wide table is verified, read and encoded again (no input cache).
    top, n_scored_rows = score_target_fold(inputs, spec.target_fold, transform, model)
    saved = pl.read_parquet(directory / PREDICTIONS_FILE).filter(pl.col(EPOCH) == spec.epochs)
    n_different, max_difference = compare_top_k(saved, top)
    LOGGER.info(
        "%s seed %d %s: %d rows scored again, %d saved Top-K rows, %d differ, largest "
        "score difference %r",
        config_id,
        spec.seed,
        spec.target_fold,
        n_scored_rows,
        saved.height,
        n_different,
        max_difference,
    )
    return RescoreResult(
        config_id, n_scored_rows, saved.height, top.height, n_different, max_difference
    )
