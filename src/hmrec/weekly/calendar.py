"""Which week an anchor date is, for the weekly pipeline (docs/mlops.md §6.3).

The anchor is the pipeline's only clock. A week is the 7 days that start at an
anchor; the pipeline numbers weeks from the first fold of
configs/fold_manifest.yaml, one week every 7 days:

    week ordinal = (anchor - anchor of the first fold) / 7        D1 = 0 ... T4 = 15

A week that is a fold of the manifest keeps the fold's name and phase ("V1",
phase "V"), so the stage-1 functions find what they expect: the same query ids
(query_id = ordinal x offset + position, as hmrec.data.queries numbers them), the
same names in the files. A week AFTER the last fold has no fold: it is named
"W" + its anchor ("W20200923") and has the phase "S". "S" is none of D, F, V, T,
so every stage-1 guard that checks a phase refuses such a week; it has no labels
in the data.

The fold manifest is only read here; nothing is changed in it.

This module reads no data and imports neither torch nor lightgbm.
"""

from dataclasses import dataclass
from datetime import date, timedelta

from hmrec.config import INT32_MAX
from hmrec.data.folds import Fold, FoldManifest

DAYS_PER_WEEK = 7
# The phase of a week after the last fold of the manifest (it is not D, F, V or T).
LATER_PHASE = "S"
LATER_NAME_PREFIX = "W"


@dataclass(frozen=True)
class Week:
    """One week of the pipeline's calendar.

    Attributes:
        fold: Name, phase, anchor and last day of the week, in the shape the
            stage-1 functions take. A manifest fold for a week of the manifest.
        ordinal: Weeks since the first fold of the manifest (D1 = 0).
        in_manifest: True if the week is a fold of configs/fold_manifest.yaml.
    """

    fold: Fold
    ordinal: int
    in_manifest: bool

    @property
    def name(self) -> str:
        """The week's name, e.g. "V1" or "W20200923"."""
        return self.fold.name

    @property
    def anchor(self) -> date:
        """First day of the week."""
        return self.fold.anchor

    @property
    def end_exclusive(self) -> date:
        """First day after the week: the anchor of the next week."""
        return self.fold.anchor + timedelta(days=DAYS_PER_WEEK)


def week_at(anchor: date, manifest: FoldManifest) -> Week:
    """The week that starts at `anchor`.

    Args:
        anchor: First day of the week.
        manifest: All folds in time order.

    Raises:
        ValueError: If the anchor is before the first fold, or is not a whole
            number of weeks after it (every anchor is a Wednesday of the fold grid).
    """
    first_anchor = manifest.folds[0].anchor
    days = (anchor - first_anchor).days
    if days < 0 or days % DAYS_PER_WEEK != 0:
        raise ValueError(
            f"{anchor} is not an anchor of the weekly grid: anchors are {first_anchor} plus a "
            f"whole number of weeks"
        )
    ordinal = days // DAYS_PER_WEEK
    if ordinal < len(manifest.folds):
        fold = manifest.folds[ordinal]
        if fold.anchor != anchor:
            raise ValueError(f"fold {fold.name} has anchor {fold.anchor}, expected {anchor}")
        return Week(fold, ordinal, in_manifest=True)
    name = f"{LATER_NAME_PREFIX}{anchor:%Y%m%d}"
    label_end = anchor + timedelta(days=DAYS_PER_WEEK - 1)
    return Week(Fold(name, LATER_PHASE, anchor, label_end), ordinal, in_manifest=False)


def week_named(name: str, manifest: FoldManifest) -> Week:
    """The week of a manifest fold given by name, e.g. "V1"."""
    return week_at(manifest.get(name).anchor, manifest)


def previous_week(week: Week, manifest: FoldManifest) -> Week:
    """The week right before `week`.

    Raises:
        ValueError: If `week` is the first week of the grid.
    """
    return week_at(week.anchor - timedelta(days=DAYS_PER_WEEK), manifest)


def weeks_before(week: Week, manifest: FoldManifest, n_weeks: int) -> tuple[Week, ...]:
    """The `n_weeks` weeks right before `week`, in time order.

    Example: the 4 weeks before V2 are F2, F3, F4, V1 (the Ranker's training
    window, proposal §9.2).

    Raises:
        ValueError: If n_weeks <= 0 or the grid has fewer weeks before `week`.
    """
    if n_weeks <= 0:
        raise ValueError(f"n_weeks must be > 0, got {n_weeks}")
    if week.ordinal < n_weeks:
        raise ValueError(
            f"week {week.name} has only {week.ordinal} weeks before it, need {n_weeks}"
        )
    anchors = [week.anchor - timedelta(days=DAYS_PER_WEEK * back) for back in range(n_weeks, 0, -1)]
    return tuple(week_at(anchor, manifest) for anchor in anchors)


def weeks_from(first: Week, last: Week, manifest: FoldManifest) -> tuple[Week, ...]:
    """Every week from `first` to `last`, both included, in time order.

    Raises:
        ValueError: If `last` is before `first`.
    """
    if last.ordinal < first.ordinal:
        raise ValueError(f"week {last.name} is before week {first.name}")
    count = last.ordinal - first.ordinal + 1
    return tuple(
        week_at(first.anchor + timedelta(days=DAYS_PER_WEEK * step), manifest)
        for step in range(count)
    )


def query_id_base(week: Week, query_id_fold_offset: int) -> int:
    """The query_id of the first query of a week: ordinal x offset.

    For a fold of the manifest this is the base hmrec.data.queries uses, so a
    week's queries have the same ids as in stage 1. The sampling seed of a query
    is (seed, query_id) (hmrec.ranker.sampling), so equal ids matter.

    Raises:
        ValueError: If a full week of ids from this base would not fit in int32.
    """
    base = week.ordinal * query_id_fold_offset
    if base + query_id_fold_offset - 1 > INT32_MAX:
        raise ValueError(
            f"week {week.name} (ordinal {week.ordinal}) x query_id_fold_offset "
            f"{query_id_fold_offset} does not fit in int32"
        )
    return base
