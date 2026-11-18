"""Recall monitoring of the T folds (M8): the numbers M6 left for the final evaluation.

M6 monitored the candidates of every fold but read NO label of a T fold: for
T1-T4 it recorded only row counts (results/m6_candidates.csv). Here the same
numbers are computed WITH labels, by the same function
(hmrec.eval.monitor.build_monitor_rows):

    per channel   Recall@quota (the channel's selection metric), candidates per
                  query, the comparison with the mean of the four folds before
    union         union recall of the six channels, candidates per query after
                  dedup, the share of queries no channel hits

The rows go to results/t_phase_recall.csv (formal run) or
results/m8_rehearsal_recall.csv (rehearsal: the configured V fold, whose rows
must equal the ones M6 recorded for it). Nothing is selected or changed.

This module reads the candidate files, the query list and the lock files
directly and imports nothing from hmrec.recall, so it can run in a process that
has LightGBM loaded (AGENTS.md §5: no torch next to LightGBM).
"""

from pathlib import Path

import polars as pl

from hmrec.csv_records import read_rows
from hmrec.eval.final_eval import FormalRunPermit
from hmrec.eval.monitor import MONITOR_COLUMNS, FoldCandidates, build_monitor_rows
from hmrec.final.context import MODE_FORMAL, MODE_REHEARSAL, FinalContext
from hmrec.final.trial import target_evaluator, training_window
from hmrec.locks import (
    ALL_CHANNELS,
    R6_CHANNEL,
    LockedChannel,
    load_channels_lock,
    load_twotower_lock,
)
from hmrec.paths import ProjectPaths

CANDIDATE_COLUMNS = ["query_id", "article_idx", "rank"]
QUERY_COLUMNS = ["query_id", "fold"]


def recall_file(context: FinalContext, mode: str) -> Path:
    """The monitoring table of a mode."""
    if mode == MODE_REHEARSAL:
        return context.paths.m8_rehearsal_recall_file
    if mode == MODE_FORMAL:
        return context.paths.t_phase_recall_file
    raise ValueError(f"unknown mode {mode!r}")


def locked_channels(paths: ProjectPaths) -> dict[str, LockedChannel]:
    """Channel -> its lock (quota, selection metric, D-phase scores), in the order R1 ... R6."""
    channels = dict(load_channels_lock(paths.channels_lock_file).channels)
    channels[R6_CHANNEL] = load_twotower_lock(paths.twotower_lock_file).channel
    return {channel: channels[channel] for channel in ALL_CHANNELS}


def read_fold_candidates(context: FinalContext, fold_name: str) -> FoldCandidates:
    """The six candidate files and the query list of one fold, as the monitor reads them.

    Reads artifacts/candidates/{fold}/{channel}.parquet (query_id, article_idx,
    rank) and artifacts/queries/{fold}.parquet (query_id, fold). No label.
    """
    paths = context.paths
    fold = context.manifest.get(fold_name)
    return FoldCandidates(
        fold=fold_name,
        phase=fold.phase,
        queries=pl.read_parquet(paths.queries_file(fold_name), columns=QUERY_COLUMNS),
        candidates_by_channel={
            channel: pl.read_parquet(
                paths.candidates_file(fold_name, channel), columns=CANDIDATE_COLUMNS
            )
            for channel in ALL_CHANNELS
        },
        locked=locked_channels(paths),
        fold_names=context.manifest.names,
    )


def earlier_rows(context: FinalContext, mode: str) -> list[dict[str, str]]:
    """The monitoring rows a fold's recall is compared with.

    The F and V folds' rows from results/m6_candidates.csv, then the rows this
    mode has already written. (The T rows of the M6 table have no recall and are
    ignored by the comparison.)
    """
    m6_rows = read_rows(context.paths.m6_candidates_results_file, MONITOR_COLUMNS)
    return [*m6_rows, *read_rows(recall_file(context, mode), MONITOR_COLUMNS)]


def monitor_fold(
    context: FinalContext, mode: str, fold_name: str, permit: FormalRunPermit | None
) -> list[dict[str, str]]:
    """The monitoring rows of one target fold: one per channel, then the union.

    The mode's fold check runs first, then the labels come from the mode's
    guarded loader (hmrec.final.trial.target_evaluator).

    Returns:
        Rows with MONITOR_COLUMNS, ready to append to recall_file(context, mode).
    """
    training_window(context, mode, fold_name)  # the mode's fold check, before any label
    evaluator = target_evaluator(context, mode, fold_name, permit)
    return build_monitor_rows(
        read_fold_candidates(context, fold_name),
        evaluator,
        earlier_rows(context, mode),
        context.monitor,
        context.base.chunking.query_chunk_size,
        context.stamp,
    )
