"""Per-feature summary of a fold's wide table: results/m6_features.csv.

For every feature of every fold: how many rows are missing, and the smallest,
largest and mean value of the rest. The table is written chunk by chunk, so
the numbers are accumulated chunk by chunk too (FeatureStats.update).

    numeric feature (float32)   missing = NaN; min / max / mean over the other rows
    category code (int16)       missing = code 0; min / max over ALL codes; no mean
                                (the mean of category codes means nothing)

These rows are a report: a sanity check of ranges and missing rates. Nothing
reads them to choose or change a feature (proposal §8.7).
"""

from dataclasses import dataclass
from datetime import datetime

import polars as pl

from hmrec.config import FeatureSpec
from hmrec.run_records import RunStamp

STATS_COLUMNS: tuple[str, ...] = (
    "fold",
    "feature",
    "group",
    "dtype",
    "categorical",
    "n_rows",
    "n_missing",
    "missing_rate",
    "min",
    "max",
    "mean",
    "git_commit",
    "git_dirty",
    "finished_at",
)
MISSING_CODE = 0
NOT_COMPUTED = ""


@dataclass
class _Running:
    """Running numbers of one feature (float64, summed chunk by chunk in chunk order)."""

    n_rows: int = 0
    n_missing: int = 0
    total: float = 0.0
    smallest: float | None = None
    largest: float | None = None


def _number(value: object) -> float:
    """A polars min / max result as a float; fails if it is not a number."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"expected a number, got {value!r}")
    return float(value)


def _merge(current: float | None, new: float | None, pick_larger: bool) -> float | None:
    """The smaller / larger of two optional numbers (None = no value yet)."""
    if new is None:
        return current
    if current is None:
        return new
    return max(current, new) if pick_larger else min(current, new)


class FeatureStats:
    """Accumulates the summary of every feature over the chunks of one fold."""

    def __init__(self, features: tuple[FeatureSpec, ...]) -> None:
        """Start with empty numbers for each feature."""
        self.features = features
        self._running = {feature.name: _Running() for feature in features}

    def update(self, wide: pl.DataFrame) -> None:
        """Add one chunk of the wide table (one row per candidate, all feature columns)."""
        for feature in self.features:
            column = wide[feature.name]
            running = self._running[feature.name]
            if feature.categorical:
                present = column  # every code counts for min / max, also the missing code 0
                n_missing = int((column == MISSING_CODE).sum())
            else:
                present = column.filter(column.is_not_nan())
                n_missing = column.len() - present.len()
                running.total += float(present.cast(pl.Float64).sum())
            running.n_rows += column.len()
            running.n_missing += n_missing
            if present.len() > 0:
                smallest, largest = _number(present.min()), _number(present.max())
                running.smallest = _merge(running.smallest, smallest, pick_larger=False)
                running.largest = _merge(running.largest, largest, pick_larger=True)

    def rows(self, fold: str, stamp: RunStamp) -> list[dict[str, str]]:
        """One row per feature for results/m6_features.csv (floats with repr())."""
        finished_at = datetime.now().isoformat(timespec="seconds")
        rows: list[dict[str, str]] = []
        for feature in self.features:
            running = self._running[feature.name]
            n_present = running.n_rows - running.n_missing
            has_mean = not feature.categorical and n_present > 0
            rows.append(
                {
                    "fold": fold,
                    "feature": feature.name,
                    "group": feature.group,
                    "dtype": feature.dtype,
                    "categorical": str(feature.categorical),
                    "n_rows": str(running.n_rows),
                    "n_missing": str(running.n_missing),
                    "missing_rate": repr(running.n_missing / running.n_rows)
                    if running.n_rows
                    else NOT_COMPUTED,
                    "min": NOT_COMPUTED if running.smallest is None else repr(running.smallest),
                    "max": NOT_COMPUTED if running.largest is None else repr(running.largest),
                    "mean": repr(running.total / n_present) if has_mean else NOT_COMPUTED,
                    "git_commit": stamp.git_commit,
                    "git_dirty": str(stamp.git_dirty),
                    "finished_at": finished_at,
                }
            )
        return rows
