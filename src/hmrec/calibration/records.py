"""The result tables of the M10 calibration (results/m10_calibration/*.csv).

Three append-only tables, all new in stage 3. No stage-1 table is written.

    comparisons.csv     one row per measured quantity of an item: the value
                        measured in the container, the stage-1 value, their
                        difference, the provisional tolerance and whether the
                        value is inside it
    runs.csv            one row per calibration process: what ran, how long, the
                        true memory peak (ru_maxrss)
    repeatability.csv   one row per artifact of two container runs: are the two
                        the same, bit for bit

Every row carries the run name, the commit, and the platform fields of
hmrec.platform_fields (OS, architecture, OpenMP library and version, BLAS, image
commit, code-tree hash): a container result is never bit-identical to the
laptop by construction, so a row must say what it was computed on.

"inside" / "outside" always refers to the PROVISIONAL tolerance of
docs/mlops.md §8.3 (configs/m10_calibration.yaml). It is a description of the
row, not a decision; the user decides the tolerance after the calibration.

This module imports neither torch nor lightgbm: both sides write these tables.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from hmrec.csv_records import append_row
from hmrec.platform_fields import PLATFORM_COLUMNS, PlatformFields

INSIDE = "inside"
OUTSIDE = "outside"
NO_TOLERANCE = ""
NOT_RECORDED = ""

IDENTITY_COLUMNS: tuple[str, ...] = ("run", "git_commit", "git_dirty", *PLATFORM_COLUMNS)

COMPARISON_COLUMNS: tuple[str, ...] = (
    "item",
    "scope",
    "quantity",
    "measured",
    "stage1",
    "difference",
    "relative_difference",
    "provisional_tolerance",
    "within_tolerance",
    "stage1_source",
    *IDENTITY_COLUMNS,
    "finished_at",
)

RUN_COLUMNS: tuple[str, ...] = (
    "item",
    "scope",
    "item_seconds",
    "total_seconds",
    "sampled_peak_rss_gb",
    "item_max_rss_gb",
    "process_max_rss_gb",
    "num_threads",
    "usable_cpus",
    "memory_limit_gb",
    "detail",
    *IDENTITY_COLUMNS,
    "finished_at",
)

REPEAT_COLUMNS: tuple[str, ...] = (
    "item",
    "artifact",
    "first_run",
    "second_run",
    "bit_identical",
    "detail",
    *IDENTITY_COLUMNS,
    "finished_at",
)

Number = int | float


@dataclass(frozen=True)
class RunIdentity:
    """Which run, code and platform a row belongs to.

    Attributes:
        run: Name of the container run, e.g. "run1".
        git_commit: The commit of the code (inside an image: the image's commit).
        git_dirty: Uncommitted changes when the process started (always False in an image).
        platform: What the process ran on, read after the work was done.
    """

    run: str
    git_commit: str
    git_dirty: bool
    platform: PlatformFields

    def columns(self) -> dict[str, str]:
        """The identity as table columns, in the order of IDENTITY_COLUMNS."""
        return {
            "run": self.run,
            "git_commit": self.git_commit,
            "git_dirty": str(self.git_dirty),
            **self.platform.columns(),
        }


@dataclass(frozen=True)
class Measurement:
    """One measured quantity of a calibration item, next to its stage-1 value.

    Attributes:
        item: The calibration item: "A", "B", "C1", "C2", "union" or "D".
        scope: What inside the item, e.g. a channel name, "fine_tune", "500 trees".
        quantity: Name of the quantity, e.g. "recall" or "recorded_pairs_present_share".
        measured: The value measured in the container.
        stage1: The stage-1 value it is compared with; None if stage 1 recorded none.
        tolerance: The provisional tolerance in words, e.g. "relative difference
            <= 0.01"; "" if docs/mlops.md §8.3 names none for this quantity.
        within_tolerance: Inside the provisional tolerance or not; None if there
            is none.
        source: Where the stage-1 value comes from, e.g. "results/m6_candidates.csv".
    """

    item: str
    scope: str
    quantity: str
    measured: Number | str
    stage1: Number | str | None
    tolerance: str = NO_TOLERANCE
    within_tolerance: bool | None = None
    source: str = ""


def value_text(value: Number | str | None) -> str:
    """A value as table text: floats with repr() (they read back exactly), None as ""."""
    if value is None:
        return NOT_RECORDED
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    return str(value)


def is_number(value: Number | str | None) -> bool:
    """True for an int or a float (a bool is a flag, not a number, and gives no difference)."""
    return isinstance(value, int | float) and not isinstance(value, bool)


def difference_texts(measured: Number | str, stage1: Number | str | None) -> tuple[str, str]:
    """(measured - stage1, (measured - stage1) / stage1) as text.

    Both are "" unless both values are numbers; the relative difference is ""
    when the stage-1 value is 0.
    """
    if not (isinstance(measured, int | float) and isinstance(stage1, int | float)):
        return "", ""
    if not (is_number(measured) and is_number(stage1)):
        return "", ""
    difference = measured - stage1
    relative = "" if stage1 == 0 else repr(float(difference) / float(stage1))
    return value_text(difference), relative


def within_text(within_tolerance: bool | None) -> str:
    """The words inside / outside, or an empty text when the quantity has no tolerance."""
    if within_tolerance is None:
        return NO_TOLERANCE
    return INSIDE if within_tolerance else OUTSIDE


def now_text() -> str:
    """The current local time to the second, ISO format."""
    return datetime.now().isoformat(timespec="seconds")


def comparison_row(
    measurement: Measurement, identity: RunIdentity, finished_at: str
) -> dict[str, str]:
    """One row of comparisons.csv (all values as text)."""
    difference, relative = difference_texts(measurement.measured, measurement.stage1)
    return {
        "item": measurement.item,
        "scope": measurement.scope,
        "quantity": measurement.quantity,
        "measured": value_text(measurement.measured),
        "stage1": value_text(measurement.stage1),
        "difference": difference,
        "relative_difference": relative,
        "provisional_tolerance": measurement.tolerance,
        "within_tolerance": within_text(measurement.within_tolerance),
        "stage1_source": measurement.source,
        **identity.columns(),
        "finished_at": finished_at,
    }


def append_comparisons(
    path: Path, measurements: Sequence[Measurement], identity: RunIdentity
) -> None:
    """Append one row per measurement to comparisons.csv; all rows get the same time."""
    finished_at = now_text()
    for measurement in measurements:
        append_row(path, COMPARISON_COLUMNS, comparison_row(measurement, identity, finished_at))


@dataclass(frozen=True)
class ProcessRecord:
    """What one calibration process did (one row of runs.csv).

    Attributes:
        item: The calibration item, e.g. "B".
        scope: What inside the item, e.g. "r5_wals".
        item_seconds: Wall time of the item's own computation (the span stage 1
            timed for the same step), without input checks and comparisons.
        total_seconds: Wall time of the whole process up to the row.
        sampled_peak_rss_gb: SAMPLED peak RSS during the computation (a lower bound).
        item_max_rss_gb: ru_maxrss right after the computation, BEFORE the
            comparison loaded the stage-1 files: the true peak of the item itself,
            comparable with the stage-1 process.
        process_max_rss_gb: ru_maxrss at the end of the process (true peak
            including the comparison).
        num_threads: num_threads of configs/base.yaml.
        usable_cpus: CPUs this process may use (the container's CPU limit).
        memory_limit_gb: The container's memory limit in GB; 0.0 if there is none.
        detail: Short free text.
    """

    item: str
    scope: str
    item_seconds: float
    total_seconds: float
    sampled_peak_rss_gb: float
    item_max_rss_gb: float
    process_max_rss_gb: float
    num_threads: int
    usable_cpus: float
    memory_limit_gb: float
    detail: str


def run_row(record: ProcessRecord, identity: RunIdentity) -> dict[str, str]:
    """One row of runs.csv (all values as text)."""
    return {
        "item": record.item,
        "scope": record.scope,
        "item_seconds": repr(record.item_seconds),
        "total_seconds": repr(record.total_seconds),
        "sampled_peak_rss_gb": repr(record.sampled_peak_rss_gb),
        "item_max_rss_gb": repr(record.item_max_rss_gb),
        "process_max_rss_gb": repr(record.process_max_rss_gb),
        "num_threads": str(record.num_threads),
        "usable_cpus": repr(record.usable_cpus),
        "memory_limit_gb": repr(record.memory_limit_gb),
        "detail": record.detail,
        **identity.columns(),
        "finished_at": now_text(),
    }


def append_run(path: Path, record: ProcessRecord, identity: RunIdentity) -> None:
    """Append the row of a process to runs.csv."""
    append_row(path, RUN_COLUMNS, run_row(record, identity))


@dataclass(frozen=True)
class RepeatCheck:
    """One artifact of two container runs, compared bit by bit.

    Attributes:
        item: The calibration item: "B", "C2" or "D".
        artifact: What was compared, e.g. "candidates (rows and scores)".
        bit_identical: True if the two runs gave the same bits.
        detail: What differs, or how much was compared.
    """

    item: str
    artifact: str
    bit_identical: bool
    detail: str


def append_repeat_checks(
    path: Path,
    checks: Sequence[RepeatCheck],
    first_run: str,
    second_run: str,
    identity: RunIdentity,
) -> None:
    """Append one row per check to repeatability.csv."""
    finished_at = now_text()
    for check in checks:
        row = {
            "item": check.item,
            "artifact": check.artifact,
            "first_run": first_run,
            "second_run": second_run,
            "bit_identical": str(check.bit_identical),
            "detail": check.detail,
            **identity.columns(),
            "finished_at": finished_at,
        }
        append_row(path, REPEAT_COLUMNS, row)
