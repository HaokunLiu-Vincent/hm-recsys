"""Tests for hmrec.ranker.dataset: lgb.Sequence over sample files and the binary Dataset."""

from dataclasses import replace
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
import pytest
from ranker_example import build_workspace

from hmrec.hpo.v_trial import TrialInputs, build_and_save_dataset, plan_dataset
from hmrec.locks import file_sha256
from hmrec.ranker.dataset import (
    BINARY_FILE,
    RECORD_FILE,
    ParquetSequence,
    build_dataset,
    dataset_is_built,
    load_dataset,
    read_dataset_record,
    read_labels_and_groups,
    run_lengths,
    save_dataset,
)
from hmrec.ranker.train import dataset_params, training_params


@pytest.fixture(scope="module")
def inputs(tmp_path_factory: pytest.TempPathFactory) -> TrialInputs:
    return build_workspace(tmp_path_factory.mktemp("dataset"))


def _write(path: Path, query_ids: list[int], values: list[float]) -> None:
    pl.DataFrame(
        {
            "query_id": query_ids,
            "label": [1] + [0] * (len(query_ids) - 1),
            "a": values,
            "code": list(range(len(values))),
        },
        schema={"query_id": pl.Int32, "label": pl.Int8, "a": pl.Float32, "code": pl.Int16},
    ).write_parquet(path, row_group_size=3)


def test_sequence_rows_and_dtypes(tmp_path: Path) -> None:
    path = tmp_path / "sample.parquet"
    _write(path, [1, 1, 2, 2, 2, 3, 3, 3], [0.5, float("nan"), 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    sequence = ParquetSequence(path, ["a", "code"], n_rows=8, block_rows=3)
    assert len(sequence) == 8 and sequence.batch_size == 3
    row = sequence[4]
    # One row: float64 (LightGBM requires doubles for the bin sample) and its own memory.
    assert row.dtype == np.float64 and row.flags["OWNDATA"]
    assert row.tolist() == [4.0, 4.0]
    assert np.isnan(sequence[1][0])
    batch = sequence[3:6]
    assert batch.dtype == np.float32 and batch.shape == (3, 2)
    assert batch.tolist() == [[3.0, 3.0], [4.0, 4.0], [5.0, 5.0]]
    # A slice that crosses blocks, and the short last block.
    assert sequence[2:8][:, 1].tolist() == [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    assert sequence[6:8].shape == (2, 2)
    with pytest.raises(IndexError):
        sequence[8]
    with pytest.raises(TypeError):
        sequence[[0, 1]]
    with pytest.raises(ValueError, match="contiguous"):
        sequence[0:6:2]
    # Columns come back in the requested (model) order, not in file order.
    swapped = ParquetSequence(path, ["code", "a"], n_rows=8, block_rows=3)
    assert swapped[4].tolist() == [4.0, 4.0] and swapped[0:1].tolist() == [[0.0, 0.5]]


def test_sequence_refuses_a_file_whose_row_groups_are_not_the_blocks(tmp_path: Path) -> None:
    path = tmp_path / "sample.parquet"
    _write(path, [1, 1, 2, 2, 2, 3, 3, 3], [0.0] * 8)  # row groups of 3, 3, 2
    with pytest.raises(ValueError, match="do not match"):
        ParquetSequence(path, ["a"], n_rows=8, block_rows=4)[0]
    with pytest.raises(ValueError, match="do not match"):
        ParquetSequence(path, ["a"], n_rows=9, block_rows=3)[0]


def test_run_lengths() -> None:
    assert run_lengths(np.array([7, 7, 9, 9, 9, 12])).tolist() == [2, 3, 1]
    assert run_lengths(np.array([], dtype=np.int32)).tolist() == []


def test_groups_follow_the_row_order(tmp_path: Path) -> None:
    first, second = tmp_path / "f1.parquet", tmp_path / "f2.parquet"
    _write(first, [1, 1, 2, 2, 2], [0.0] * 5)
    _write(second, [7, 8, 8], [0.0] * 3)
    labels, groups = read_labels_and_groups([first, second])
    # Rows: q1 q1 | q2 q2 q2 | q7 | q8 q8 -> group sizes in exactly this order.
    assert groups.tolist() == [2, 3, 1, 2] and groups.dtype == np.int32
    assert labels.tolist() == [1, 0, 0, 0, 0, 1, 0, 0] and labels.dtype == np.int8
    assert groups.sum() == len(labels)
    # Files in the wrong (time) order: query_id decreases, groups would be wrong.
    with pytest.raises(ValueError, match="query_id decreases"):
        read_labels_and_groups([second, first])


def test_built_dataset_matches_the_sample_files(inputs: TrialInputs) -> None:
    plan = plan_dataset(inputs, "full", "V1", 2, "test-commit")
    assert [source.fold for source in plan.sources] == ["F1", "F2", "F3", "F4"]
    dataset, n_queries = build_dataset(
        plan.sources,
        plan.experiment.feature_names,
        plan.experiment.categorical_names,
        plan.params,
        inputs.ranker.sampling.row_group_rows,
    )
    samples = pl.concat([pl.read_parquet(source.sample_file) for source in plan.sources])
    assert dataset.num_data() == samples.height == plan.record.n_rows
    assert dataset.num_feature() == 71
    assert n_queries == samples["query_id"].n_unique() == 240
    group = dataset.get_group()
    label = dataset.get_label()
    assert group is not None and label is not None
    # LightGBM's groups are the run lengths of query_id in row order.
    assert np.asarray(group).tolist() == run_lengths(samples["query_id"].to_numpy()).tolist()
    assert np.asarray(label).tolist() == samples["label"].to_list()
    assert dataset.get_feature_name() == list(plan.experiment.feature_names)
    # Every query: 1 positive + 2 negatives (ratio 2).
    assert set(np.asarray(group).tolist()) == {3}


def test_binary_is_reproducible_and_reusable(inputs: TrialInputs, tmp_path: Path) -> None:
    plan = plan_dataset(inputs, "full", "V1", 2, "test-commit")
    record = build_and_save_dataset(plan, inputs.ranker.sampling.row_group_rows)
    assert record.n_queries == 240 and record.binary_sha256 == file_sha256(
        plan.directory / BINARY_FILE
    )
    assert dataset_is_built(plan.directory)
    # Built again somewhere else: the binary file is byte-identical.
    other = replace(plan, directory=tmp_path / "again")
    assert build_and_save_dataset(other, 64).binary_sha256 == record.binary_sha256
    # One binary serves different num_leaves / min_data_in_leaf (feature_pre_filter is off).
    expected = replace(plan.record, n_queries=record.n_queries)
    for num_leaves, min_data in ((4, 5), (8, 2)):
        tuned = replace(
            inputs.ranker.tuned_defaults, num_leaves=num_leaves, min_data_in_leaf=min_data
        )
        params = training_params(inputs.ranker, tuned, "col_wise", 2)
        booster = lgb.train(params, load_dataset(plan.directory, expected, params), 3)
        assert booster.num_trees() == 3
    with pytest.raises(FileExistsError, match="never replaced"):
        build_and_save_dataset(plan, 64)


def test_load_refuses_another_dataset(inputs: TrialInputs, tmp_path: Path) -> None:
    plan = replace(plan_dataset(inputs, "full", "V1", 4, "c"), directory=tmp_path / "ds")
    record = build_and_save_dataset(plan, 64)
    params = dataset_params(inputs.ranker, 2)
    expected = replace(plan.record, n_queries=record.n_queries)
    assert load_dataset(plan.directory, expected, params).num_data() == record.n_rows
    assert read_dataset_record(plan.directory) == record
    # Another sample file (by sha256) than the one the binary was built from.
    changed = replace(expected, sample_sha256={**expected.sample_sha256, "F1": "0" * 64})
    with pytest.raises(ValueError, match="other inputs than expected.*sample_sha256"):
        load_dataset(plan.directory, changed, params)
    with pytest.raises(ValueError, match="different: \\['ratio'\\]"):
        load_dataset(plan.directory, replace(expected, ratio=2), params)
    # The binary file itself was changed.
    with (plan.directory / BINARY_FILE).open("ab") as handle:
        handle.write(b"x")
    with pytest.raises(ValueError, match="sha256 is"):
        load_dataset(plan.directory, expected, params)


def test_half_written_dataset_is_an_error(tmp_path: Path) -> None:
    assert not dataset_is_built(tmp_path)
    (tmp_path / BINARY_FILE).write_bytes(b"half")
    with pytest.raises(ValueError, match="stopped half"):
        dataset_is_built(tmp_path)
    (tmp_path / RECORD_FILE).write_text("{}")
    assert dataset_is_built(tmp_path)


def test_save_needs_an_empty_folder(inputs: TrialInputs, tmp_path: Path) -> None:
    plan = plan_dataset(inputs, "full", "V1", 2, "c")
    (tmp_path / RECORD_FILE).write_text("{}")
    dataset, _ = build_dataset(plan.sources, plan.experiment.feature_names, [], plan.params, 64)
    with pytest.raises(FileExistsError):
        save_dataset(dataset, tmp_path, plan.record)
