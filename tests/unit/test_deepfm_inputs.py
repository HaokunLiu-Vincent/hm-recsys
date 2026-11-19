"""Tests for deepfm.inputs: quantile bins, standardising, encoding (hand-computed)."""

import json
from pathlib import Path

import numpy as np
import pytest

from hmrec.deepfm.inputs import (
    BINNED,
    STANDARDISED,
    RawRows,
    bin_values,
    clip_and_standardise_statistics,
    encode_rows,
    fit_transform,
    load_transform,
    quantile_edges,
    save_transform,
    signed_log1p,
    standardise_values,
)

NAN = float("nan")


def _column(values: list[float]) -> np.ndarray:
    return np.array(values, dtype=np.float32)


def test_quantile_edges_hand_computed() -> None:
    # 1 ... 8 in 4 bins: the quantiles 0.25, 0.5, 0.75 of 8 values with linear
    # interpolation are at positions 1.75, 3.5, 5.25 -> 2.75, 4.5, 6.25.
    edges = quantile_edges(_column([8, 1, 7, 2, 6, 3, 5, 4]), 4)
    assert edges.tolist() == [2.75, 4.5, 6.25] and edges.dtype == np.float64


def test_missing_values_do_not_move_the_edges() -> None:
    with_missing = _column([8, NAN, 1, 7, 2, NAN, 6, 3, 5, 4])
    assert quantile_edges(with_missing, 4).tolist() == [2.75, 4.5, 6.25]
    # A feature without any value has no edge: every later value is bin 1.
    assert quantile_edges(_column([NAN, NAN]), 4).size == 0
    assert bin_values(_column([3.0, NAN]), np.zeros(0)).tolist() == [1, 0]


def test_equal_edges_are_merged() -> None:
    # Six zeros and two larger values: the quantiles 0.25 and 0.5 are both 0.
    edges = quantile_edges(_column([0, 0, 0, 0, 0, 0, 5, 9]), 4)
    assert edges.tolist() == [0.0, 1.25]


def test_bin_is_one_plus_the_edges_at_or_below_the_value() -> None:
    edges = np.array([2.75, 4.5, 6.25])
    values = _column([-100, 2.0, 2.75, 3, 4.5, 6.25, 100, NAN])
    # A value equal to an edge belongs to the bin above it; NaN is bin 0.
    assert bin_values(values, edges).tolist() == [1, 1, 2, 2, 3, 4, 4, 0]
    assert bin_values(values, edges).dtype == np.uint8


def test_signed_log1p_hand_computed() -> None:
    # g(x) = sign(x) * log(1 + |x|): g(0) = 0, g(e - 1) = 1, g(-(e - 1)) = -1, g(99) = log(100).
    e = float(np.e)
    g = signed_log1p(np.array([0.0, e - 1, -(e - 1), 99.0, NAN], dtype=np.float64))
    assert g[:4].tolist() == pytest.approx([0.0, 1.0, -1.0, np.log(100.0)])
    assert np.isnan(g[4]) and g.dtype == np.float64
    # It keeps the order of the values and shortens a long tail: 240,417 -> about 12.4.
    values = np.array([-50.0, -1.0, 0.0, 0.5, 3.0, 240417.0], dtype=np.float32)
    assert np.all(np.diff(signed_log1p(values)) > 0)
    assert signed_log1p(values)[-1] == pytest.approx(12.39, abs=0.01)


def test_standardise_statistics_hand_computed() -> None:
    # The statistics are those of g(x) = log(1 + x) here (all values >= 0), not of x.
    # x = 0 ... 100: g is increasing, so the 10th and 90th percentiles of g(x) are
    # g(10) = log(11) and g(90) = log(91). Clipping g(x) to them gives eleven log(11),
    # the values log(12) ... log(90) and eleven log(91).
    values = _column([*range(101), NAN])
    low, high, mean, std = clip_and_standardise_statistics(values, 10, 90)
    assert (low, high) == pytest.approx((np.log(11.0), np.log(91.0)))
    logs = [np.log(11.0)] * 11 + [np.log(float(x + 1)) for x in range(11, 90)] + [np.log(91.0)] * 11
    assert mean == pytest.approx(sum(logs) / 101)
    assert std == pytest.approx(np.array(logs).std())
    z, missing = standardise_values(_column([50, 0, 100, NAN]), low, high, mean, std)
    # g(0) is clipped up to log(11) and g(100) down to log(91).
    expected = [
        (np.log(51.0) - mean) / std,
        (np.log(11.0) - mean) / std,
        (np.log(91.0) - mean) / std,
    ]
    assert z.tolist() == pytest.approx([*expected, 0.0], rel=1e-6)
    assert missing.tolist() == [0, 0, 0, 1] and z.dtype == np.float32


def test_a_long_tail_no_longer_gives_a_huge_standardised_value() -> None:
    # 999 small values and one very large one. The 99.9th percentile lies between the
    # two, so the large value is only partly clipped. Without g(x) its z would be
    # above 25; with g(x) it stays below 10.
    values = _column([*([1.0] * 500), *([2.0] * 499), 240417.0])
    low, high, mean, std = clip_and_standardise_statistics(values, 0.1, 99.9)
    z, _ = standardise_values(values, low, high, mean, std)
    assert float(z.max()) < 10.0
    raw = np.clip(values.astype(np.float64), *np.percentile(values, [0.1, 99.9]))
    assert float((raw.max() - raw.mean()) / raw.std()) > 25.0


def test_constant_or_empty_feature_standardises_to_zero() -> None:
    assert clip_and_standardise_statistics(_column([NAN]), 0.1, 99.9) == (0.0, 0.0, 0.0, 0.0)
    low, high, mean, std = clip_and_standardise_statistics(_column([3, 3, 3]), 0.1, 99.9)
    # The statistics are those of g(x): g(3) = log(4).
    assert (mean, std) == (pytest.approx(np.log(4.0)), 0.0)
    z, missing = standardise_values(_column([3, 7, NAN]), low, high, mean, std)
    assert z.tolist() == [0.0, 0.0, 0.0] and missing.tolist() == [0, 0, 1]


TRAIN_NUMERIC = [[1, 10], [2, NAN], [3, 30], [4, 40], [5, 50], [6, 60], [7, 70], [8, 80]]
TRAIN = RawRows(
    numeric=np.array(TRAIN_NUMERIC, dtype=np.float32),
    categorical=np.array([[0], [1], [2], [1], [0], [2], [1], [2]], dtype=np.int16),
)
NAMES = (("a", "b"), ("c",), (3,))


def _fit(numeric_input: str):
    return fit_transform(TRAIN.numeric, *NAMES, numeric_input, 4, (10.0, 90.0))


def test_binned_rows_put_the_numeric_bins_first_then_the_codes() -> None:
    transform = _fit(BINNED)
    assert transform.layout.code_field_sizes == (5, 5, 3) and transform.layout.n_value_fields == 0
    assert transform.layout.n_fields == 3
    assert transform.edges[0].tolist() == [2.75, 4.5, 6.25]
    encoded = encode_rows(TRAIN, transform)
    assert encoded.codes.dtype == np.uint8 and encoded.values is None and encoded.missing is None
    assert encoded.codes[:, 0].tolist() == [1, 1, 2, 2, 3, 3, 4, 4]
    assert encoded.codes[1].tolist() == [1, 0, 1]  # row 2: feature b is missing -> bin 0
    assert encoded.codes[:, 2].tolist() == [0, 1, 2, 1, 0, 2, 1, 2]


def test_standardised_rows_keep_values_and_a_missing_indicator() -> None:
    transform = _fit(STANDARDISED)
    assert transform.layout.code_field_sizes == (3,) and transform.layout.n_value_fields == 2
    encoded = encode_rows(TRAIN, transform)
    assert encoded.codes.shape == (8, 1)
    assert encoded.values is not None and encoded.missing is not None
    assert encoded.values.shape == (8, 2) and encoded.values.dtype == np.float32
    assert encoded.missing[:, 1].tolist() == [0, 1, 0, 0, 0, 0, 0, 0]
    assert encoded.values[1, 1] == 0.0  # the missing value itself is 0
    assert float(encoded.values[:, 0].mean()) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("numeric_input", [BINNED, STANDARDISED])
def test_target_week_values_never_change_what_was_fitted(numeric_input: str) -> None:
    transform = _fit(numeric_input)
    before = json.dumps([e.tolist() for e in transform.edges] + [transform.mean.tolist()])
    # Target-week rows far outside the training range, and all missing.
    extreme = RawRows(
        numeric=np.array([[1e9, -1e9], [NAN, NAN], [-1e9, 1e9]], dtype=np.float32),
        categorical=np.array([[2], [0], [1]], dtype=np.int16),
    )
    encoded = encode_rows(extreme, transform)
    after = json.dumps([e.tolist() for e in transform.edges] + [transform.mean.tolist()])
    assert after == before
    # Fitting again on the same training rows gives the same statistics, whatever was encoded.
    again = _fit(numeric_input)
    assert json.dumps([e.tolist() for e in again.edges] + [again.mean.tolist()]) == before
    if numeric_input == BINNED:
        # Extreme values fall into the first or the last training bin; missing is bin 0.
        assert encoded.codes[:, :2].tolist() == [[4, 1], [0, 0], [1, 4]]
    else:
        assert encoded.values is not None
        # Clipped to the training percentiles, so no value explodes.
        assert float(np.abs(encoded.values).max()) < 3.0


def test_unknown_codes_and_bad_arguments_are_refused(tmp_path: Path) -> None:
    transform = _fit(BINNED)
    unknown = RawRows(TRAIN.numeric[:1], np.array([[3]], dtype=np.int16))
    with pytest.raises(ValueError, match="expected 0..2"):
        encode_rows(unknown, transform)
    with pytest.raises(ValueError, match="numeric_input must be"):
        fit_transform(TRAIN.numeric, *NAMES, "raw", 4, (10.0, 90.0))
    with pytest.raises(ValueError, match="1..256 codes"):
        fit_transform(TRAIN.numeric, ("a", "b"), ("c",), (300,), BINNED, 4, (10.0, 90.0))
    with pytest.raises(ValueError, match="n_bins must be >= 2"):
        quantile_edges(TRAIN.numeric[:, 0], 1)
    save_transform(transform, tmp_path / "transform.json")
    saved = json.loads((tmp_path / "transform.json").read_text())
    assert saved["edges"][0] == [2.75, 4.5, 6.25] and saved["numeric_names"] == ["a", "b"]
    with pytest.raises(FileExistsError, match="never replaced"):
        save_transform(transform, tmp_path / "transform.json")


@pytest.mark.parametrize("numeric_input", [BINNED, STANDARDISED])
def test_a_saved_transform_is_read_back_exactly(tmp_path: Path, numeric_input: str) -> None:
    transform = _fit(numeric_input)
    save_transform(transform, tmp_path / "transform.json")
    loaded = load_transform(tmp_path / "transform.json")
    assert (loaded.numeric_input, loaded.numeric_names, loaded.categorical_names) == (
        transform.numeric_input,
        transform.numeric_names,
        transform.categorical_names,
    )
    assert loaded.categorical_sizes == transform.categorical_sizes and loaded.n_bins == 4
    assert len(loaded.edges) == len(transform.edges)
    for one, two in zip(loaded.edges, transform.edges, strict=True):
        assert np.array_equal(one, two) and one.dtype == np.float64
    for name in ("clip_low", "clip_high", "mean", "std"):
        assert np.array_equal(getattr(loaded, name), getattr(transform, name))
    # So rows are encoded identically with the loaded transform.
    one, two = encode_rows(TRAIN, transform), encode_rows(TRAIN, loaded)
    assert np.array_equal(one.codes, two.codes)
    if one.values is not None and two.values is not None:
        assert np.array_equal(one.values, two.values)
