"""Repeatability: are two container runs of an item the same, bit for bit.

Items B (R5), C2 (R6 fine-tuning and its candidates) and D (LightGBM) are run a
second time, each in a fresh container, into the second run's own folders. The
functions here read the outputs of both runs and compare them bit by bit. They
compute nothing of the system and read no label.

The answer describes two runs on the SAME platform. It says nothing about the
laptop: a container result is never compared bit by bit with a laptop result.

What is compared, and why not simply the files' sha256: a Parquet file or a
checkpoint can differ in bytes that are not results (a checkpoint stores the
seconds its fine-tune took). So tables are compared by their values, vectors by
their arrays, models by their weights or their trees.

This module imports neither torch nor lightgbm at the top. Only
repeat_r6_fine_tune needs torch, and imports it inside the function, so the
LightGBM check can run in a process without torch.
"""

from pathlib import Path

import numpy as np
import polars as pl

from hmrec.calibration.compare import (
    arrays_bit_identical,
    leading_identical_trees,
    split_model_trees,
    tables_bit_identical,
)
from hmrec.calibration.context import MAIN_TREE, CalibrationContext
from hmrec.calibration.records import RepeatCheck
from hmrec.calibration.references import single_row
from hmrec.locks import R6_CHANNEL, load_ranker_lock
from hmrec.paths import ProjectPaths, calibration_output_paths

ITEM_B = "B"
ITEM_C2 = "C2"
ITEM_D = "D"
R5_CHANNEL = "r5_wals"
VECTOR_FILES = ("user_customer_idx", "user_vectors", "item_article_idx", "item_vectors")
RANKER_AREA = "m10_calibration"
FULL_EXPERIMENT = "full"
CHAIN_RESULT_COLUMNS = ("mean_train_loss", "alarm_loss", "steps", "n_new_rows", "n_replay_rows")


def run_paths(context: CalibrationContext, run: str) -> ProjectPaths:
    """The "main" output tree of one run on this platform."""
    return calibration_output_paths(context.paths, context.platform_name, run, MAIN_TREE)


def repeat_table(item: str, artifact: str, first: Path, second: Path) -> RepeatCheck:
    """Compare two Parquet files by their rows and values."""
    identical, detail = tables_bit_identical(pl.read_parquet(first), pl.read_parquet(second))
    return RepeatCheck(item, artifact, identical, detail)


def repeat_vectors(item: str, first_dir: Path, second_dir: Path) -> list[RepeatCheck]:
    """Compare the four arrays of two saved vector folders (hmrec.vectors)."""
    checks = []
    for name in VECTOR_FILES:
        identical, detail = arrays_bit_identical(
            np.load(first_dir / f"{name}.npy"), np.load(second_dir / f"{name}.npy")
        )
        checks.append(RepeatCheck(item, f"vectors: {name}", identical, detail))
    return checks


def repeat_r5(context: CalibrationContext, first_run: str, second_run: str) -> list[RepeatCheck]:
    """Item B twice: the R5 candidates and the saved vectors of the two runs."""
    fold = context.config.fold
    first, second = run_paths(context, first_run), run_paths(context, second_run)
    checks = [
        repeat_table(
            ITEM_B,
            "candidates (rows, ranks and scores)",
            first.candidates_file(fold, R5_CHANNEL),
            second.candidates_file(fold, R5_CHANNEL),
        )
    ]
    checks.extend(
        repeat_vectors(ITEM_B, first.wals_vectors_dir(fold), second.wals_vectors_dir(fold))
    )
    return checks


def repeat_model_weights(first_file: Path, second_file: Path) -> RepeatCheck:
    """Compare the weights of two Two-Tower checkpoints, tensor by tensor (needs torch)."""
    import torch

    first = torch.load(first_file, map_location="cpu", weights_only=True)["state_dict"]
    second = torch.load(second_file, map_location="cpu", weights_only=True)["state_dict"]
    artifact = "fine-tuned model (weights)"
    if list(first) != list(second):
        return RepeatCheck(ITEM_C2, artifact, False, "the two checkpoints hold different tensors")
    different = [name for name in first if not torch.equal(first[name], second[name])]
    if different:
        return RepeatCheck(ITEM_C2, artifact, False, f"tensors that differ: {different}")
    return RepeatCheck(ITEM_C2, artifact, True, f"{len(first)} tensors")


def repeat_chain_record(
    context: CalibrationContext, first_run: str, second_run: str
) -> RepeatCheck:
    """Compare the fine-tune numbers the two runs recorded (losses, steps, row counts)."""
    fold = context.config.fine_tune_fold
    rows = [
        single_row(context.paths.calibration_chain_file(context.platform_name, run), {"fold": fold})
        for run in (first_run, second_run)
    ]
    different = [name for name in CHAIN_RESULT_COLUMNS if rows[0][name] != rows[1][name]]
    detail = (
        f"columns that differ: {different}"
        if different
        else f"train loss {rows[0]['mean_train_loss']}, alarm loss {rows[0]['alarm_loss']}"
    )
    return RepeatCheck(ITEM_C2, "fine-tune record (losses, steps, rows)", not different, detail)


def repeat_r6(context: CalibrationContext, first_run: str, second_run: str) -> list[RepeatCheck]:
    """Item C2 twice: the fine-tuned model, its record, and the candidates it gives."""
    config = context.config
    first, second = run_paths(context, first_run), run_paths(context, second_run)
    tuned = config.fine_tune_fold
    checks = [
        repeat_model_weights(
            first.twotower_locked_chain_checkpoint_file(tuned),
            second.twotower_locked_chain_checkpoint_file(tuned),
        ),
        repeat_chain_record(context, first_run, second_run),
    ]
    for fold in (tuned, config.fold):
        checks.append(
            repeat_table(
                ITEM_C2,
                f"candidates of {fold} (rows, ranks and scores)",
                first.candidates_file(fold, R6_CHANNEL),
                second.candidates_file(fold, R6_CHANNEL),
            )
        )
    checks.extend(
        repeat_vectors(
            ITEM_C2,
            first.twotower_vectors_dir(config.fold),
            second.twotower_vectors_dir(config.fold),
        )
    )
    return checks


def repeat_ranker(
    context: CalibrationContext, first_run: str, second_run: str
) -> list[RepeatCheck]:
    """Item D twice: the trees of the two models, their Top-K and per-query metric values.

    The output paths hold the locked configuration id; it is read from the
    Ranker lock file (parsing the file loads neither LightGBM nor torch).
    """
    config_id = load_ranker_lock(context.paths.ranker_lock_file).selection.config_id
    key = (RANKER_AREA, FULL_EXPERIMENT, config_id, context.config.fold)
    first, second = run_paths(context, first_run), run_paths(context, second_run)
    first_trees = split_model_trees(first.ranker_model_file(*key).read_text(encoding="utf-8"))
    second_trees = split_model_trees(second.ranker_model_file(*key).read_text(encoding="utf-8"))
    identical = leading_identical_trees(first_trees, second_trees)
    same_model = identical == len(first_trees) == len(second_trees)
    detail = (
        f"{len(first_trees)} trees"
        if same_model
        else f"first differing tree: {identical} of {len(first_trees)} / {len(second_trees)}"
    )
    return [
        RepeatCheck(ITEM_D, "model (text of every tree)", same_model, detail),
        repeat_table(
            ITEM_D,
            "Top-12 of every query at every checkpoint (articles and scores)",
            first.ranker_predictions_file(*key),
            second.ranker_predictions_file(*key),
        ),
        repeat_table(
            ITEM_D,
            "per-query metric values at every checkpoint",
            first.ranker_per_query_file(*key),
            second.ranker_per_query_file(*key),
        ),
    ]
