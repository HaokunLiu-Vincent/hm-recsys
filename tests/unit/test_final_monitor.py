"""Tests for hmrec.final.monitor: the recall monitoring of M8 gives M6's numbers, with T labels."""

from pathlib import Path

import pytest
from final_example import COMMIT, N_QUERIES, NOW, copy_workspace

from hmrec.csv_records import append_row, read_rows
from hmrec.eval.final_eval import FormalRunPermit
from hmrec.eval.monitor import MONITOR_COLUMNS
from hmrec.final.context import MODE_FORMAL, MODE_REHEARSAL
from hmrec.final.monitor import (
    earlier_rows,
    locked_channels,
    monitor_fold,
    read_fold_candidates,
    recall_file,
)
from hmrec.locks import ALL_CHANNELS

PERMIT = FormalRunPermit(COMMIT, COMMIT, NOW, NOW)
RUN_COLUMNS = ("git_commit", "git_dirty", "finished_at")


def test_the_six_locked_channels_come_from_the_two_lock_files(tmp_path: Path) -> None:
    context = copy_workspace(tmp_path)
    channels = locked_channels(context.paths)
    assert tuple(channels) == ALL_CHANNELS  # R1 ... R6, in this order
    assert channels["r1_popularity"].quota == 100
    assert channels["r6_twotower"].channel == "r6_twotower"  # from twotower.lock.yaml


def test_fold_candidates_are_read_without_labels(tmp_path: Path) -> None:
    context = copy_workspace(tmp_path)
    context.paths.labels_file("T1").unlink()  # reading candidates needs no label
    data = read_fold_candidates(context, "T1")
    assert (data.fold, data.phase) == ("T1", "T")
    assert data.queries.height == N_QUERIES and data.queries.columns == ["query_id", "fold"]
    assert tuple(data.candidates_by_channel) == ALL_CHANNELS
    for candidates in data.candidates_by_channel.values():
        assert candidates.columns == ["query_id", "article_idx", "rank"]


def test_the_rehearsal_rows_equal_the_rows_m6_recorded(tmp_path: Path) -> None:
    context = copy_workspace(tmp_path)
    rows = monitor_fold(context, MODE_REHEARSAL, "V4", None)
    m6_rows = read_rows(context.paths.m6_candidates_results_file, MONITOR_COLUMNS)
    assert [row["scope"] for row in rows] == [*ALL_CHANNELS, "union"]
    for ours, theirs in zip(rows, m6_rows, strict=True):
        for column in MONITOR_COLUMNS:
            if column not in RUN_COLUMNS:
                assert ours[column] == theirs[column], (ours["scope"], column)
    assert all(row["labels_read"] == "True" for row in rows)


def test_t_folds_get_label_based_numbers_only_with_a_permit(tmp_path: Path) -> None:
    context = copy_workspace(tmp_path)
    rows = monitor_fold(context, MODE_FORMAL, "T1", PERMIT)
    union = rows[-1]
    assert (union["fold"], union["phase"], union["scope"]) == ("T1", "T", "union")
    # Every candidate of the tiny folds is returned by at least one channel and the label
    # is one of the 12 candidates, so the union of the six channels finds every label.
    assert float(union["recall"]) == 1.0 and float(union["zero_hit_query_share"]) == 0.0
    assert float(union["candidates_per_query_mean"]) == 12.0
    assert all(row["recall"] != "" for row in rows)
    with pytest.raises(TypeError, match="needs a FormalRunPermit"):
        monitor_fold(context, MODE_FORMAL, "T2", None)
    # A rehearsal cannot monitor a T fold, and the formal mode cannot monitor a V fold.
    with pytest.raises(ValueError, match="only the configured rehearsal fold"):
        monitor_fold(context, MODE_REHEARSAL, "T1", None)
    with pytest.raises(ValueError, match="predicts only T folds"):
        monitor_fold(context, MODE_FORMAL, "V4", PERMIT)


def test_a_t_fold_is_compared_with_the_four_folds_before_it(tmp_path: Path) -> None:
    context = copy_workspace(tmp_path)
    assert recall_file(context, MODE_FORMAL) == context.paths.t_phase_recall_file
    assert recall_file(context, MODE_REHEARSAL) == context.paths.m8_rehearsal_recall_file
    # The workspace's M6 table holds V4 only, so T1's four folds before (V1 ... V4) are not
    # all recorded: nothing is compared and nothing is flagged.
    first = monitor_fold(context, MODE_FORMAL, "T1", PERMIT)[0]
    assert first["previous_folds"] == "V1 V2 V3 V4" and first["previous_mean_recall"] == ""
    assert first["low_recall_flag"] == "False"
    for row in monitor_fold(context, MODE_FORMAL, "T1", PERMIT):
        append_row(context.paths.t_phase_recall_file, MONITOR_COLUMNS, row)
    # The rows written for T1 are part of the history T2 is compared with.
    history = earlier_rows(context, MODE_FORMAL)
    assert {row["fold"] for row in history} == {"V4", "T1"}
    second = monitor_fold(context, MODE_FORMAL, "T2", PERMIT)[0]
    assert second["previous_folds"] == "V2 V3 V4 T1"
