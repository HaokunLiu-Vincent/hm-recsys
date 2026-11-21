"""Check that every metric in the MLflow store equals the value in its source.

MLflow is a view; the result tables and the run manifests are the record
(docs/mlops.md §7.3, goal S2). This check reads the runs back from the store
and, for EVERY metric of every run:

    1. finds the metric's source tag ("source.<key>": file, row, column);
    2. reads that row of the source file again, with the small reader of this
       module (not with the code that built the runs);
    3. requires the two float64 values to be EQUAL, not close: SQLite stores a
       float64 as it is, and the tables hold repr() texts.

It also compares the store with the runs the view should hold: no run missing,
no run that is not expected, and the same metric keys in each.

The functions here take the runs as plain data (hmrec.tracking.store.LoggedRun),
so they need no MLflow.
"""

import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hmrec.tracking.records import (
    SOURCE_TAG_PREFIX,
    MetricSource,
    TrackedRun,
    select_row,
    source_from_text,
)
from hmrec.tracking.store import LoggedRun

JSON_SUFFIX = ".json"


@dataclass(frozen=True)
class CheckResult:
    """The outcome of a check.

    Attributes:
        n_runs: Runs read from the store.
        n_metrics: Metric values compared with their source.
        problems: One line per difference; empty if the view equals its sources.
    """

    n_runs: int
    n_metrics: int
    problems: tuple[str, ...]

    @property
    def passed(self) -> bool:
        """True if nothing differs."""
        return not self.problems


def _source_rows(repo_root: Path, source: MetricSource) -> list[dict[str, Any]]:
    """The rows of a source that a metric may come from.

    A CSV table: its rows. A JSON manifest: the list under `section`, or the
    top-level object as one row when `section` is "".
    """
    path = repo_root / source.file
    if not path.is_file():
        raise ValueError(f"source file not found: {source.file}")
    if path.suffix != JSON_SUFFIX:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    content = json.loads(path.read_text(encoding="utf-8"))
    if source.section == "":
        return [content]
    return list(content[source.section])


def source_value(repo_root: Path, source: MetricSource) -> float:
    """The value at a metric's source, read from the file now.

    Raises:
        ValueError: If the file, the row or the column is missing, or the row is
            not unique.
    """
    rows = _source_rows(repo_root, source)
    row = select_row(rows, source.where, source.file)
    if source.column not in row:
        raise ValueError(f"{source.file}: no column {source.column!r}")
    return float(row[source.column])


def _metric_problem(repo_root: Path, run: LoggedRun, key: str, value: float) -> str | None:
    """What is wrong with one logged metric, or None if it equals its source."""
    where = f"{run.experiment} / {run.record_id} / {key}"
    tag = run.tags.get(SOURCE_TAG_PREFIX + key)
    if tag is None:
        return f"{where}: the metric has no source tag"
    try:
        expected = source_value(repo_root, source_from_text(tag))
    except ValueError as error:
        return f"{where}: {error}"
    if value != expected:
        return f"{where}: MLflow has {value!r}, the source has {expected!r}"
    return None


def _coverage_problems(logged: Sequence[LoggedRun], expected: Sequence[TrackedRun]) -> list[str]:
    """Runs or metric keys that are missing from the store, or in it without being expected."""
    problems: list[str] = []
    in_store = {(run.experiment, run.record_id): run for run in logged}
    if len(in_store) != len(logged):
        problems.append("the store holds two runs with the same record id in one experiment")
    wanted = {(run.experiment, run.record_id): run for run in expected}
    for identity in sorted(set(wanted) - set(in_store)):
        problems.append(f"{identity[0]} / {identity[1]}: expected, but not in the store")
    for identity in sorted(set(in_store) - set(wanted)):
        problems.append(f"{identity[0]} / {identity[1]}: in the store, but not expected")
    for identity in sorted(set(wanted) & set(in_store)):
        logged_keys = set(in_store[identity].metrics)
        wanted_keys = {metric.key for metric in wanted[identity].metrics}
        if logged_keys != wanted_keys:
            difference = sorted(logged_keys ^ wanted_keys)
            problems.append(f"{identity[0]} / {identity[1]}: metric keys differ: {difference}")
    return problems


def check_logged_runs(
    logged: Sequence[LoggedRun], expected: Sequence[TrackedRun], repo_root: Path
) -> CheckResult:
    """Compare the runs of the store with their sources and with the expected runs.

    Args:
        logged: The runs read from the store (hmrec.tracking.store.read_experiment_runs).
        expected: The runs the view should hold (hmrec.tracking.records).
        repo_root: The repository root (source files are relative to it).

    Returns:
        How many runs and metric values were compared, and every difference.
    """
    problems = _coverage_problems(logged, expected)
    n_metrics = 0
    for run in logged:
        for key, value in sorted(run.metrics.items()):
            n_metrics += 1
            problem = _metric_problem(repo_root, run, key, value)
            if problem is not None:
                problems.append(problem)
    return CheckResult(n_runs=len(logged), n_metrics=n_metrics, problems=tuple(problems))
