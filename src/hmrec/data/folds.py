"""Fold dataclass; read and validate configs/fold_manifest.yaml (proposal §3).

A fold is one label week:
    - inputs (history) use t_dat < anchor;
    - labels use anchor <= t_dat <= label_end (label_end is INCLUSIVE),
      which is the same as the half-open window [anchor, anchor + 7d).

Validation (all must hold, otherwise ValueError):
    - every week covers exactly `week_length_days` days;
    - weeks are sorted, adjacent (next anchor = previous label_end + 1 day) and
      do not overlap;
    - the last label day equals `last_label_date` (2020-09-22);
    - phases appear in `phase_order` (D, F, V, T), `folds_per_phase` folds each,
      and fold names are phase + 1-based index (D1 ... D4, F1 ... T4).
"""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from hmrec.config import (
    check_keys,
    get_date,
    get_int,
    get_mapping,
    get_str,
    get_str_list,
    load_yaml,
    require_positive,
)


@dataclass(frozen=True)
class Fold:
    """One label week.

    Attributes:
        name: Fold name, e.g. "D1".
        phase: One of "D", "F", "V", "T".
        anchor: First day of the label week. History uses t_dat < anchor.
        label_end: Last day of the label week, inclusive.
    """

    name: str
    phase: str
    anchor: date
    label_end: date

    @property
    def label_start(self) -> date:
        """First label day (inclusive). Same as the anchor."""
        return self.anchor

    @property
    def label_end_exclusive(self) -> date:
        """First day AFTER the label week, for half-open filters t_dat < label_end_exclusive."""
        return self.label_end + timedelta(days=1)

    @property
    def label_days(self) -> int:
        """Number of days in the label week, counting both ends."""
        return (self.label_end - self.anchor).days + 1


@dataclass(frozen=True)
class FoldChecks:
    """Invariants the manifest must satisfy, read from its `checks` block.

    Attributes:
        week_length_days: Days per label week (7).
        folds_per_phase: Folds in each phase (4).
        phase_order: Phases in time order ("D", "F", "V", "T").
        last_label_date: Last label day of the last fold (2020-09-22).
    """

    week_length_days: int
    folds_per_phase: int
    phase_order: tuple[str, ...]
    last_label_date: date


@dataclass(frozen=True)
class FoldManifest:
    """All folds in time order.

    Attributes:
        folds: Folds sorted by anchor, D1 first and T4 last.
    """

    folds: tuple[Fold, ...]

    def get(self, name: str) -> Fold:
        """Return the fold called `name`, e.g. "V2"."""
        for fold in self.folds:
            if fold.name == name:
                return fold
        known = [fold.name for fold in self.folds]
        raise KeyError(f"unknown fold {name!r}; known folds: {known}")

    def by_phase(self, phase: str) -> tuple[Fold, ...]:
        """Return the folds of one phase in time order; phase "D" gives D1..D4."""
        folds = tuple(fold for fold in self.folds if fold.phase == phase)
        if not folds:
            raise KeyError(f"no folds for phase {phase!r}")
        return folds

    @property
    def names(self) -> tuple[str, ...]:
        """Fold names in time order."""
        return tuple(fold.name for fold in self.folds)


# ---------------------------------------------------------------- parsing


def parse_fold_checks(raw: Mapping[str, Any]) -> FoldChecks:
    """Parse the `checks` block of the manifest."""
    where = "fold_manifest.checks"
    check_keys(
        raw, {"week_length_days", "folds_per_phase", "phase_order", "last_label_date"}, where
    )
    checks = FoldChecks(
        week_length_days=get_int(raw, "week_length_days", where),
        folds_per_phase=get_int(raw, "folds_per_phase", where),
        phase_order=get_str_list(raw, "phase_order", where),
        last_label_date=get_date(raw, "last_label_date", where),
    )
    require_positive(checks.week_length_days, f"{where}.week_length_days")
    require_positive(checks.folds_per_phase, f"{where}.folds_per_phase")
    return checks


def parse_fold(raw: Mapping[str, Any], position: int) -> Fold:
    """Parse one entry of the `folds` list.

    Args:
        raw: Mapping with keys name, phase, anchor, label_end.
        position: 0-based position in the list, only used in error messages.
    """
    where = f"fold_manifest.folds[{position}]"
    check_keys(raw, {"name", "phase", "anchor", "label_end"}, where)
    return Fold(
        name=get_str(raw, "name", where),
        phase=get_str(raw, "phase", where),
        anchor=get_date(raw, "anchor", where),
        label_end=get_date(raw, "label_end", where),
    )


def parse_fold_manifest(raw: Mapping[str, Any]) -> tuple[FoldManifest, FoldChecks]:
    """Parse the manifest content without validating it.

    Args:
        raw: Top-level mapping with keys checks and folds.

    Returns:
        The manifest (folds in file order) and its checks.
    """
    check_keys(raw, {"checks", "folds"}, "fold_manifest")
    checks = parse_fold_checks(get_mapping(raw, "checks", "fold_manifest"))
    raw_folds = raw["folds"]
    if not isinstance(raw_folds, list) or not raw_folds:
        raise ValueError("fold_manifest.folds: expected a non-empty list")
    folds = []
    for position, raw_fold in enumerate(raw_folds):
        if not isinstance(raw_fold, Mapping):
            raise ValueError(f"fold_manifest.folds[{position}]: expected a mapping")
        folds.append(parse_fold(raw_fold, position))
    return FoldManifest(folds=tuple(folds)), checks


# ---------------------------------------------------------------- validation


def check_week_lengths(folds: tuple[Fold, ...], week_length_days: int) -> None:
    """Fail if any fold does not cover exactly `week_length_days` days."""
    for fold in folds:
        if fold.label_days != week_length_days:
            raise ValueError(
                f"fold {fold.name}: label week {fold.anchor}..{fold.label_end} has "
                f"{fold.label_days} days, expected {week_length_days}"
            )


def check_adjacent_without_overlap(folds: tuple[Fold, ...]) -> None:
    """Fail if two consecutive folds overlap or leave a gap.

    Consecutive folds must satisfy next.anchor == previous.label_end + 1 day.
    Overlap is reported separately from a gap so the message says which one it is.
    """
    for previous, current in zip(folds, folds[1:], strict=False):
        if current.anchor <= previous.label_end:
            raise ValueError(
                f"folds {previous.name} and {current.name} overlap or are out of order: "
                f"{current.name} starts {current.anchor}, {previous.name} ends {previous.label_end}"
            )
        if current.anchor != previous.label_end_exclusive:
            raise ValueError(
                f"gap between folds {previous.name} and {current.name}: {current.name} "
                f"starts {current.anchor}, expected {previous.label_end_exclusive}"
            )


def check_last_label_date(folds: tuple[Fold, ...], last_label_date: date) -> None:
    """Fail if the last fold does not end on `last_label_date`."""
    last_fold = folds[-1]
    if last_fold.label_end != last_label_date:
        raise ValueError(
            f"last fold {last_fold.name} ends {last_fold.label_end}, expected {last_label_date}"
        )


def check_names_and_phases(
    folds: tuple[Fold, ...], phase_order: tuple[str, ...], folds_per_phase: int
) -> None:
    """Fail unless folds are exactly D1..D4, F1..F4, V1..V4, T1..T4 in this order.

    The expected names are built from `phase_order` and `folds_per_phase`, so
    this also checks the total count, the phase of every fold and uniqueness.
    """
    expected = [
        (phase, f"{phase}{index}")
        for phase in phase_order
        for index in range(1, folds_per_phase + 1)
    ]
    actual = [(fold.phase, fold.name) for fold in folds]
    if actual != expected:
        raise ValueError(f"fold (phase, name) list is {actual}, expected {expected}")


def validate_fold_manifest(manifest: FoldManifest, checks: FoldChecks) -> None:
    """Run all manifest checks. See the module docstring for the list.

    Raises:
        ValueError: On the first check that fails.
    """
    check_names_and_phases(manifest.folds, checks.phase_order, checks.folds_per_phase)
    check_week_lengths(manifest.folds, checks.week_length_days)
    check_adjacent_without_overlap(manifest.folds)
    check_last_label_date(manifest.folds, checks.last_label_date)


def load_fold_manifest(path: Path) -> FoldManifest:
    """Read, parse and validate configs/fold_manifest.yaml.

    Args:
        path: Path to the manifest (normally ProjectPaths.fold_manifest_file).

    Returns:
        The validated manifest, folds in time order.
    """
    manifest, checks = parse_fold_manifest(load_yaml(path))
    validate_fold_manifest(manifest, checks)
    return manifest
