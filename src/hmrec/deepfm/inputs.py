"""Turn the 71 feature columns into DeepFM inputs (proposal §16.2, §16.3).

Every feature is one FIELD. Two kinds of fields reach the model:

    code fields    an integer code per row; the model looks up one embedding per
                   (field, code). The 8 categorical features (their stored codes,
                   0 = missing) and, with numeric_input = "binned", the 63 numeric
                   features after quantile binning.
    value fields   a float per row plus a 0/1 "missing" indicator; only with
                   numeric_input = "standardised".

Field order: the numeric features in the order of the features lock, then the
categorical features in that order. (The order only fixes which parameters
belong to which feature.)

Binned (the default). For each numeric feature, from its non-missing values in
the TRAINING rows of the window:

    edges = quantiles k / n_bins, k = 1 ... n_bins - 1   (numpy, linear interpolation,
            float64); equal edges are merged
    bin(x) = 1 + (number of edges <= x)                   -> 1 ... n_bins
    bin(NaN) = 0                                           (the missing bin)

Standardised. For each numeric feature, from its non-missing training values:

    lo, hi = the clip_low / clip_high percentiles; x is clipped to [lo, hi]
    mean, std of the clipped values; z = (clip(x) - mean) / std   (0 if std = 0)
    missing: z = 0 and the indicator m = 1

The statistics are FITTED on the training rows of one window only and then
applied unchanged to the target week: no value of the target week can move an
edge or a mean (tests/unit/test_deepfm_inputs.py).

Encoded codes are stored as uint8 (at most 256 values per field), values as
float32: 10.7 million training rows x 71 fields = 0.76 GB.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MISSING_CODE = 0
CODE_DTYPE = np.uint8
MAX_FIELD_SIZE = 256  # a code must fit in uint8
BINNED = "binned"
STANDARDISED = "standardised"


@dataclass(frozen=True)
class RawRows:
    """Feature values of some rows as they are stored in the wide table.

    Attributes:
        numeric: float32 [n_rows, n_numeric]; NaN = missing. Columns in lock order.
        categorical: int16 [n_rows, n_categorical]; code 0 = missing. Columns in lock order.
    """

    numeric: np.ndarray
    categorical: np.ndarray

    @property
    def n_rows(self) -> int:
        """Number of rows."""
        return int(self.numeric.shape[0])


@dataclass(frozen=True)
class EncodedRows:
    """Model inputs of some rows.

    Attributes:
        codes: uint8 [n_rows, n_code_fields]: one code per code field.
        values: float32 [n_rows, n_value_fields], or None if there are no value fields.
        missing: uint8 [n_rows, n_value_fields] (1 = missing), or None.
    """

    codes: np.ndarray
    values: np.ndarray | None
    missing: np.ndarray | None

    @property
    def n_rows(self) -> int:
        """Number of rows."""
        return int(self.codes.shape[0])


@dataclass(frozen=True)
class FieldLayout:
    """The fields of a model: how many codes each code field has, and the value fields.

    Attributes:
        code_field_sizes: Number of distinct codes of every code field, in field order.
        n_value_fields: Number of value fields (0 for binned inputs).
    """

    code_field_sizes: tuple[int, ...]
    n_value_fields: int

    @property
    def n_fields(self) -> int:
        """All fields: code fields plus value fields."""
        return len(self.code_field_sizes) + self.n_value_fields


@dataclass(frozen=True)
class InputTransform:
    """What was fitted on the training rows of one window.

    Attributes:
        numeric_input: "binned" or "standardised".
        numeric_names: The numeric features in column order.
        categorical_names: The categorical features in column order.
        categorical_sizes: Codes of every categorical field (its values + 1 for missing).
        n_bins: Quantile bins of the binned input.
        edges: Binned: one float64 array of bin edges per numeric feature
            (ascending, distinct, at most n_bins - 1). Standardised: empty.
        clip_low: Standardised: float64 [n_numeric] lower clip values; else empty.
        clip_high: Standardised: upper clip values.
        mean: Standardised: means of the clipped training values.
        std: Standardised: their standard deviations.
    """

    numeric_input: str
    numeric_names: tuple[str, ...]
    categorical_names: tuple[str, ...]
    categorical_sizes: tuple[int, ...]
    n_bins: int
    edges: tuple[np.ndarray, ...]
    clip_low: np.ndarray
    clip_high: np.ndarray
    mean: np.ndarray
    std: np.ndarray

    @property
    def layout(self) -> FieldLayout:
        """The model's fields for this transform (numeric fields first, then categorical)."""
        if self.numeric_input == BINNED:
            # n_bins quantile bins plus the missing bin.
            numeric_sizes = (self.n_bins + 1,) * len(self.numeric_names)
            return FieldLayout((*numeric_sizes, *self.categorical_sizes), 0)
        return FieldLayout(self.categorical_sizes, len(self.numeric_names))


# ---------------------------------------------------------------- binning


def quantile_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Bin edges of one numeric feature: the quantiles k / n_bins of its non-missing values.

    Args:
        values: float32 [n], the feature's TRAINING values; NaN = missing.
        n_bins: Number of quantile bins wanted (>= 2).

    Returns:
        float64 array of at most n_bins - 1 edges, ascending and distinct. Empty
        if every value is missing (then every non-missing value is bin 1).
    """
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}")
    present = values[~np.isnan(values)].astype(np.float64)
    if present.size == 0:
        return np.zeros(0, dtype=np.float64)
    probabilities = np.arange(1, n_bins, dtype=np.float64) / n_bins
    return np.unique(np.quantile(present, probabilities))


def bin_values(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Bin of every value: 0 for NaN, else 1 + (number of edges <= value).

    Args:
        values: float32 [n]; NaN = missing.
        edges: float64, ascending, from quantile_edges.

    Returns:
        uint8 [n], in 0 ... len(edges) + 1.
    """
    missing = np.isnan(values)
    # side="right": a value equal to an edge goes to the bin above the edge.
    bins = 1 + np.searchsorted(edges, values.astype(np.float64), side="right")
    bins[missing] = MISSING_CODE
    return bins.astype(CODE_DTYPE)


# ---------------------------------------------------------------- standardising


def clip_and_standardise_statistics(
    values: np.ndarray, low_percentile: float, high_percentile: float
) -> tuple[float, float, float, float]:
    """(lo, hi, mean, std) of one numeric feature from its non-missing training values.

    lo / hi are the two percentiles; mean / std are those of the values clipped
    to [lo, hi] (population std). A feature with no value at all gives
    (0, 0, 0, 0): its rows are all missing.
    """
    present = values[~np.isnan(values)].astype(np.float64)
    if present.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    low, high = np.percentile(present, [low_percentile, high_percentile])
    clipped = np.clip(present, low, high)
    return float(low), float(high), float(clipped.mean()), float(clipped.std())


def standardise_values(
    values: np.ndarray, low: float, high: float, mean: float, std: float
) -> tuple[np.ndarray, np.ndarray]:
    """Standardise one numeric feature: z = (clip(x) - mean) / std, plus its missing indicator.

    Returns:
        (z float32 [n], missing uint8 [n]). A missing value has z = 0; std = 0
        gives z = 0 everywhere (the feature is constant in the training rows).
    """
    missing = np.isnan(values)
    clipped = np.clip(values.astype(np.float64), low, high)
    z = (clipped - mean) / std if std > 0 else np.zeros_like(clipped)
    z[missing] = 0.0
    return z.astype(np.float32), missing.astype(CODE_DTYPE)


# ---------------------------------------------------------------- fit and apply


def check_categorical_sizes(sizes: tuple[int, ...]) -> None:
    """Fail unless every categorical field's codes fit in uint8."""
    too_large = [size for size in sizes if not 1 <= size <= MAX_FIELD_SIZE]
    if too_large:
        raise ValueError(f"a code field must have 1..{MAX_FIELD_SIZE} codes, got {too_large}")


def fit_transform(
    numeric: np.ndarray,
    numeric_names: tuple[str, ...],
    categorical_names: tuple[str, ...],
    categorical_sizes: tuple[int, ...],
    numeric_input: str,
    n_bins: int,
    clip_percentiles: tuple[float, float],
) -> InputTransform:
    """Fit the input transform on the TRAINING rows of one window.

    Args:
        numeric: float32 [n_train_rows, n_numeric]; NaN = missing.
        numeric_names: Its columns.
        categorical_names: The categorical features.
        categorical_sizes: Codes per categorical field (values + 1).
        numeric_input: "binned" or "standardised".
        n_bins: Quantile bins (binned).
        clip_percentiles: (low, high) percentiles (standardised).

    Raises:
        ValueError: On an unknown numeric_input or a field too large for uint8.
    """
    if numeric.shape[1] != len(numeric_names):
        raise ValueError(f"numeric has {numeric.shape[1]} columns, expected {len(numeric_names)}")
    if numeric_input not in (BINNED, STANDARDISED):
        raise ValueError(f"numeric_input must be binned or standardised, got {numeric_input!r}")
    check_categorical_sizes(categorical_sizes)
    n_numeric = len(numeric_names)
    edges: tuple[np.ndarray, ...] = ()
    statistics = np.zeros((4, 0), dtype=np.float64)
    if numeric_input == BINNED:
        edges = tuple(quantile_edges(numeric[:, column], n_bins) for column in range(n_numeric))
    else:
        columns = [
            clip_and_standardise_statistics(numeric[:, column], *clip_percentiles)
            for column in range(n_numeric)
        ]
        statistics = np.array(columns, dtype=np.float64).T.reshape(4, n_numeric)
    return InputTransform(
        numeric_input=numeric_input,
        numeric_names=numeric_names,
        categorical_names=categorical_names,
        categorical_sizes=categorical_sizes,
        n_bins=n_bins,
        edges=edges,
        clip_low=statistics[0],
        clip_high=statistics[1],
        mean=statistics[2],
        std=statistics[3],
    )


def encode_categorical(categorical: np.ndarray, sizes: tuple[int, ...]) -> np.ndarray:
    """The stored categorical codes as uint8, after checking that each is a known code.

    Raises:
        ValueError: If a code is negative or not below its field's size.
    """
    if categorical.shape[1] != len(sizes):
        raise ValueError(f"categorical has {categorical.shape[1]} columns, expected {len(sizes)}")
    for column, size in enumerate(sizes):
        codes = categorical[:, column]
        if codes.size and (codes.min() < 0 or codes.max() >= size):
            raise ValueError(
                f"categorical column {column} has codes {codes.min()}..{codes.max()}, "
                f"expected 0..{size - 1}"
            )
    return categorical.astype(CODE_DTYPE)


def encode_rows(rows: RawRows, transform: InputTransform) -> EncodedRows:
    """Apply a fitted transform to rows (training rows or the target week's candidates).

    Returns:
        Binned: codes uint8 [n, n_numeric + n_categorical] (numeric bins first).
        Standardised: codes uint8 [n, n_categorical], values float32 and missing
        uint8 [n, n_numeric].
    """
    n_numeric = len(transform.numeric_names)
    categorical = encode_categorical(rows.categorical, transform.categorical_sizes)
    if transform.numeric_input == BINNED:
        codes = np.empty((rows.n_rows, n_numeric + categorical.shape[1]), dtype=CODE_DTYPE)
        for column in range(n_numeric):
            codes[:, column] = bin_values(rows.numeric[:, column], transform.edges[column])
        codes[:, n_numeric:] = categorical
        return EncodedRows(codes, None, None)
    values = np.empty((rows.n_rows, n_numeric), dtype=np.float32)
    missing = np.empty((rows.n_rows, n_numeric), dtype=CODE_DTYPE)
    for column in range(n_numeric):
        values[:, column], missing[:, column] = standardise_values(
            rows.numeric[:, column],
            float(transform.clip_low[column]),
            float(transform.clip_high[column]),
            float(transform.mean[column]),
            float(transform.std[column]),
        )
    return EncodedRows(categorical, values, missing)


def save_transform(transform: InputTransform, path: Path) -> None:
    """Write a fitted transform as JSON (a record of what a training used). Never replaced."""
    if path.exists():
        raise FileExistsError(f"{path} already exists; a transform is never replaced")
    content = {
        "numeric_input": transform.numeric_input,
        "numeric_names": list(transform.numeric_names),
        "categorical_names": list(transform.categorical_names),
        "categorical_sizes": list(transform.categorical_sizes),
        "n_bins": transform.n_bins,
        "edges": [edges.tolist() for edges in transform.edges],
        "clip_low": transform.clip_low.tolist(),
        "clip_high": transform.clip_high.tolist(),
        "mean": transform.mean.tolist(),
        "std": transform.std.tolist(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content), encoding="utf-8")
