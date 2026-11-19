"""Tests for deepfm.cache: the cache stores exactly what was encoded and is verified before use.

Unit part: a writer / reader round trip on tiny arrays, and what open_cache
refuses. Trial part (the tiny workspace of deepfm_example): a training that
reads the cache gives exactly the result of a training without any cache.
"""

import dataclasses
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch
from deepfm_example import TARGET, build_workspace

from hmrec.deepfm.cache import (
    CachedPart,
    CachedTraining,
    CacheWriter,
    open_cache,
    source_key,
)
from hmrec.deepfm.inputs import EncodedRows, InputTransform
from hmrec.deepfm.records import deepfm_config_id
from hmrec.deepfm.trial import (
    AREA_V_PHASE,
    TrialInputs,
    TrialSpec,
    cache_dir,
    cache_source,
    check_spec,
    run_trial,
)

SOURCE = {"target_fold": "V1", "numeric_input": "binned", "n_bins": 4}
SPEC = TrialSpec(AREA_V_PHASE, TARGET, 42, 2)


def _transform(numeric_input: str) -> InputTransform:
    statistics = np.array([0.5, 1.5], dtype=np.float64)
    return InputTransform(
        numeric_input=numeric_input,
        numeric_names=("a", "b"),
        categorical_names=("c",),
        categorical_sizes=(3,),
        n_bins=4,
        # 0.1 has no exact binary form: the round trip must still be exact.
        edges=(np.array([0.1, 0.7]), np.array([2.0])) if numeric_input == "binned" else (),
        clip_low=statistics if numeric_input != "binned" else np.zeros(0),
        clip_high=statistics + 1 if numeric_input != "binned" else np.zeros(0),
        mean=statistics / 3 if numeric_input != "binned" else np.zeros(0),
        std=statistics * 2 if numeric_input != "binned" else np.zeros(0),
    )


def _rows(n_rows: int, with_values: bool, seed: int) -> EncodedRows:
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, 5, size=(n_rows, 3)).astype(np.uint8)
    if not with_values:
        return EncodedRows(codes, None, None)
    values = rng.normal(size=(n_rows, 2)).astype(np.float32)
    return EncodedRows(codes, values, (values > 0).astype(np.uint8))


def _write(directory: Path, numeric_input: str = "binned") -> tuple[CachedTraining, CachedPart]:
    with_values = numeric_input != "binned"
    training = CachedTraining(
        _transform(numeric_input),
        _rows(6, with_values, 0),
        np.array([1, 0, 0, 1, 0, 0], dtype=np.uint8),
        np.array([7, 7, 7, 8, 8, 8], dtype=np.int32),
    )
    part = CachedPart(
        "part-00000.parquet",
        _rows(4, with_values, 1),
        np.array([20, 20, 21, 21], dtype=np.int32),
        np.array([3, 5, 3, 9], dtype=np.int32),
    )
    writer = CacheWriter(directory, SOURCE)
    writer.write_training(training)
    writer.write_part(part)
    writer.finish()
    return training, part


def _same_rows(one: EncodedRows, two: EncodedRows) -> bool:
    if (one.values is None) != (two.values is None):
        return False
    same = np.array_equal(one.codes, two.codes) and one.codes.dtype == two.codes.dtype
    if one.values is not None and two.values is not None:
        same = same and np.array_equal(one.values, two.values)
        same = same and np.array_equal(one.missing, two.missing)  # type: ignore[arg-type]  # both set
    return same


@pytest.mark.parametrize("numeric_input", ["binned", "standardised"])
def test_round_trip_gives_back_exactly_what_was_written(tmp_path: Path, numeric_input: str) -> None:
    directory = tmp_path / "V1_cache"
    training, part = _write(directory, numeric_input)
    reader = open_cache(directory, SOURCE)
    assert reader is not None and reader.parts == ("part-00000.parquet",)
    read = reader.read_training()
    assert _same_rows(read.rows, training.rows)
    assert np.array_equal(read.labels, training.labels) and read.labels.dtype == np.uint8
    assert np.array_equal(read.query_ids, training.query_ids) and read.query_ids.dtype == np.int32
    # The transform comes back bit for bit (floats are stored with their full repr).
    assert read.transform.numeric_names == ("a", "b") and read.transform.n_bins == 4
    for one, two in zip(read.transform.edges, training.transform.edges, strict=True):
        assert np.array_equal(one, two)
    assert np.array_equal(read.transform.std, training.transform.std)
    assert read.transform.layout == training.transform.layout
    read_part = reader.read_part("part-00000.parquet")
    assert _same_rows(read_part.rows, part.rows)
    assert read_part.article_idx.tolist() == [3, 5, 3, 9]
    with pytest.raises(ValueError, match="has no part"):
        reader.read_part("part-00009.parquet")


def test_no_cache_yet_is_none_and_a_cache_is_never_replaced(tmp_path: Path) -> None:
    directory = tmp_path / "V1_cache"
    assert open_cache(directory, SOURCE) is None
    _write(directory)
    with pytest.raises(FileExistsError, match="never replaced"):
        CacheWriter(directory, SOURCE)
    # An unfinished cache (no record yet) is invisible to readers and blocks a second writer.
    other = tmp_path / "V2_cache"
    CacheWriter(other, SOURCE)
    assert open_cache(other, SOURCE) is None
    with pytest.raises(FileExistsError, match="unfinished"):
        CacheWriter(other, SOURCE)


def test_a_cache_is_verified_before_it_is_used(tmp_path: Path) -> None:
    directory = tmp_path / "V1_cache"
    _write(directory)
    # Another source than the one it was written for.
    with pytest.raises(ValueError, match="another source"):
        open_cache(directory, {**SOURCE, "n_bins": 8})
    # One changed byte in one array.
    target = directory / "part-00000_codes.npy"
    content = bytearray(target.read_bytes())
    content[-1] ^= 1
    target.write_bytes(bytes(content))
    with pytest.raises(ValueError, match="sha256"):
        open_cache(directory, SOURCE)
    # A file that the record does not list, or a folder without a record.
    second = tmp_path / "V2_cache"
    _write(second)
    (second / "extra.npy").write_bytes(b"x")
    with pytest.raises(ValueError, match="other files than its record"):
        open_cache(second, SOURCE)
    third = tmp_path / "V3_cache"
    _write(third)
    (third / "cache.json").unlink()
    with pytest.raises(ValueError, match="not a finished cache"):
        open_cache(third, SOURCE)
    assert (
        source_key(SOURCE) != source_key({**SOURCE, "n_bins": 8}) and len(source_key(SOURCE)) == 12
    )


# ---------------------------------------------------------------- with a whole trial


def _without_cache(inputs: TrialInputs) -> TrialInputs:
    return dataclasses.replace(
        inputs, config=dataclasses.replace(inputs.config, cached_numeric_inputs=())
    )


def _trial_files(inputs: TrialInputs, spec: TrialSpec) -> Path:
    return inputs.paths.deepfm_trial_dir(
        spec.area, deepfm_config_id(inputs.config), spec.seed, spec.target_fold
    )


def test_a_cached_trial_equals_a_trial_without_cache(tmp_path: Path) -> None:
    torch.set_num_threads(1)
    cached_inputs = build_workspace(tmp_path / "cached")
    plain_inputs = _without_cache(build_workspace(tmp_path / "plain"))
    # Seed 42 writes the cache; seed 43 reads it. The plain workspace never has one.
    assert run_trial(cached_inputs, SPEC).input_cache == "written"
    other = dataclasses.replace(SPEC, seed=43)
    from_cache = run_trial(cached_inputs, other)
    plain = run_trial(plain_inputs, other)
    assert (from_cache.input_cache, plain.input_cache) == ("read", "off")
    assert from_cache.metrics[0].scores == plain.metrics[0].scores
    assert [e.mean_loss for e in from_cache.epochs] == [e.mean_loss for e in plain.epochs]
    one, two = _trial_files(cached_inputs, other), _trial_files(plain_inputs, other)
    assert pl.read_parquet(one / "predictions.parquet").equals(
        pl.read_parquet(two / "predictions.parquet")
    )
    assert (one / "transform.json").read_text() == (two / "transform.json").read_text()
    assert not plain_inputs.paths.deepfm_cache_dir(TARGET, "binned", "x").parent.exists()


def test_a_damaged_cache_stops_the_trial(tmp_path: Path) -> None:
    inputs = build_workspace(tmp_path)
    run_trial(inputs, SPEC)
    folds = check_spec(inputs, SPEC)
    source = cache_source(inputs, TARGET, folds)
    directory = cache_dir(inputs, TARGET, source)
    assert directory.name.startswith(f"{TARGET}_binned_") and (directory / "cache.json").is_file()
    # The key names the rows LightGBM trained on and the record of the wide table.
    assert sorted(source["sample_sha256"]) == ["F1", "F2", "F3", "F4"]
    recorded = json.loads((directory / "cache.json").read_text())
    assert recorded["source"] == source and recorded["parts"] == [
        "part-00000.parquet",
        "part-00001.parquet",
    ]
    codes = directory / "train_codes.npy"
    content = bytearray(codes.read_bytes())
    content[-1] ^= 1
    codes.write_bytes(bytes(content))
    with pytest.raises(ValueError, match="sha256"):
        run_trial(inputs, dataclasses.replace(SPEC, seed=43))


def test_only_the_listed_input_modes_are_cached(tmp_path: Path) -> None:
    inputs = build_workspace(tmp_path)
    assert inputs.config.cached_numeric_inputs == ("binned",)
    model = dataclasses.replace(inputs.config.model, numeric_input="standardised")
    standardised = dataclasses.replace(
        inputs, config=dataclasses.replace(inputs.config, model=model)
    )
    assert run_trial(standardised, SPEC).input_cache == "off"
    assert not (inputs.paths.deepfm_dir / "cache").exists()
