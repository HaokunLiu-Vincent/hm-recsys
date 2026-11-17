"""One row per M6 process in results/m6_runs.csv: what ran, how long, how much memory.

Every (fold, step) of candidate generation runs in its own process (step = a
channel name, or "monitor"). As its LAST action the process appends one row
here. Because the process does one step only, its process max RSS is the true
memory peak of that step (hmrec.memory: ru_maxrss never goes down, so a
process that ran several steps could only report the largest of them).

Two memory columns, never to be mixed up (see hmrec.memory):
    sampled_peak_rss_gb   sampled during the generate step only; a lower bound
    process_max_rss_gb    ru_maxrss of the whole process; the true peak
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from hmrec.csv_records import append_row
from hmrec.memory import ProcessPeakMemory

RUN_COLUMNS: tuple[str, ...] = (
    "fold",
    "step",
    "status",
    "n_rows",
    "generate_seconds",
    "total_seconds",
    "sampled_peak_rss_gb",
    "process_max_rss_gb",
    "children_max_rss_gb",
    "checked_rows",
    "detail",
    "git_commit",
    "git_dirty",
    "finished_at",
)

GENERATED = "generated"
SKIPPED = "skipped"
FINE_TUNE_ONLY = "fine_tune_only"
STATUSES = (GENERATED, SKIPPED, FINE_TUNE_ONLY)


@dataclass(frozen=True)
class RunStamp:
    """The code state of a run.

    Attributes:
        git_commit: HEAD when the run started.
        git_dirty: Uncommitted code / config changes when the run started.
    """

    git_commit: str
    git_dirty: bool


@dataclass(frozen=True)
class StepResult:
    """What one step of one fold did.

    Attributes:
        fold: Fold name, e.g. "F1".
        step: A channel name ("r4_itemcf") or "monitor".
        status: "generated" (new output written), "skipped" (the output
            already existed and was left untouched) or "fine_tune_only" (R6:
            the candidates existed, only the missing fine-tune was done).
        n_rows: Rows of the step's main output (candidate rows; monitor rows).
        generate_seconds: Wall time of the main computation (0 when skipped).
        sampled_peak_rss_gb: SAMPLED peak RSS during that computation (a lower bound).
        checked_rows: Rows covered by the step's consistency check (0 = the
            step has no such check, or it was skipped).
        detail: Short free text, e.g. the R6 fine-tune losses.
    """

    fold: str
    step: str
    status: str
    n_rows: int
    generate_seconds: float
    sampled_peak_rss_gb: float
    checked_rows: int
    detail: str

    def __post_init__(self) -> None:
        """Only the three known statuses are allowed."""
        if self.status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}, got {self.status!r}")


def run_row(
    result: StepResult, total_seconds: float, peak: ProcessPeakMemory, stamp: RunStamp
) -> dict[str, str]:
    """One row of results/m6_runs.csv (all values as text; floats with repr()).

    Args:
        result: What the step did.
        total_seconds: Wall time of the whole process up to this point.
        peak: The process's true peak memory, read as the last step of the run.
        stamp: The code state of the run.
    """
    return {
        "fold": result.fold,
        "step": result.step,
        "status": result.status,
        "n_rows": str(result.n_rows),
        "generate_seconds": repr(result.generate_seconds),
        "total_seconds": repr(total_seconds),
        "sampled_peak_rss_gb": repr(result.sampled_peak_rss_gb),
        "process_max_rss_gb": repr(peak.process_max_rss_gb),
        "children_max_rss_gb": repr(peak.children_max_rss_gb),
        "checked_rows": str(result.checked_rows),
        "detail": result.detail,
        "git_commit": stamp.git_commit,
        "git_dirty": str(stamp.git_dirty),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }


def append_run_row(path: Path, row: dict[str, str]) -> None:
    """Append one row to results/m6_runs.csv (and the header if the file is new)."""
    append_row(path, RUN_COLUMNS, row)


def check_memory_limit(peak: ProcessPeakMemory, limit_gb: float, label: str) -> None:
    """Fail if the process's TRUE peak RSS is above the limit (proposal §1.3: 8 GB).

    Called after the run row is written, so the run is on record; the error
    then stops a driver that runs the steps one after another.

    Raises:
        MemoryError: If process_max_rss_gb > limit_gb.
    """
    if peak.process_max_rss_gb > limit_gb:
        raise MemoryError(
            f"{label}: process max RSS {peak.process_max_rss_gb:.2f} GB is above the limit "
            f"of {limit_gb:.2f} GB"
        )
