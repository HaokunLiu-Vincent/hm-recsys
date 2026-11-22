"""Which runs the MLflow view holds, built from result tables and run manifests.

Three kinds of runs (configs/m11_versioning.yaml names their experiments):

    stage-1 weekly records   one run per week V1-V4 and T1-T4: the recall of each
                             channel, the union recall, and MAP@12 of the Ranker
                             and of the RRF baseline, as stage 1 recorded them.
    M10 calibration          one run per (calibration run, item, scope) of
                             results/m10_calibration/comparisons.csv.
    weekly pipeline          one run per run manifest (from M12 on).

The first two are IMPORTED RECORDS: they were made before the weekly pipeline
existed and are copied into the view so that it is useful now. Every run carries
the tag hmrec.record_kind ("imported_record" or "pipeline_run"), and the
experiments of imported records say "(imported)" in their names.

Reading the stage-1 result tables of the T weeks reads recorded numbers, not
labels: no label file is opened here.

Every metric carries its SOURCE: the file, the row and the column its value was
read from. hmrec.tracking.check reads the source again, on its own, and compares.

This module needs no MLflow and no model library.
"""

import csv
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from hmrec.data.folds import Fold
from hmrec.locks import ALL_CHANNELS, RankerLock
from hmrec.run_manifest import RunManifest

IMPORTED_RECORD = "imported_record"
PIPELINE_RUN = "pipeline_run"

# Tags every run of the view carries.
RECORD_KIND_TAG = "hmrec.record_kind"
RECORD_ID_TAG = "hmrec.record_id"
# The tag that holds the source of the metric <key>: SOURCE_TAG_PREFIX + key.
SOURCE_TAG_PREFIX = "source."

# MLflow allows letters, digits, "_", "-", ".", " ", ":" and "/" in a metric name.
NOT_ALLOWED_IN_KEY = re.compile(r"[^A-Za-z0-9_\-. :/]")
AT_SIGN = "@"
AT_WORD = "_at_"

# Scope of the union row in the recall tables (hmrec.eval.monitor).
UNION_SCOPE = "union"
# The rows of the metric tables (hmrec.hpo.v_records): the Ranker with all six channels,
# and the RRF baseline (which has no trees).
FULL_EXPERIMENT = "full"
RRF_BASELINE = "rrf_baseline"
RRF_NUM_TREES = "0"
MAP_COLUMN = "map@12"
RECALL_COLUMN = "recall"

# Columns of results/m10_calibration/comparisons.csv that become metrics, with the
# suffix of the metric key ("" = the measured value itself).
CALIBRATION_VALUE_COLUMNS = (
    ("measured", ""),
    ("stage1", ".stage1"),
    ("relative_difference", ".relative_difference"),
)
# Columns of results/m10_calibration/runs.csv that become metrics of a process.
CALIBRATION_PROCESS_COLUMNS = (
    "item_seconds",
    "total_seconds",
    "item_max_rss_gb",
    "process_max_rss_gb",
)
CALIBRATION_GROUP = ("run", "item", "scope")
# Sections of a run manifest a metric can come from ("" = the top level).
MANIFEST_METRICS = "metrics"
MANIFEST_STEPS = "steps"
MANIFEST_TOP = ""
DAYS_PER_WEEK = 7


@dataclass(frozen=True)
class MetricSource:
    """Where a metric value was read from.

    Attributes:
        file: The source file, relative to the repository root, with "/".
        section: "" for a CSV table and for the top level of a JSON manifest;
            for a list inside a manifest its key ("metrics", "steps").
        where: Column -> text that select EXACTLY ONE row; empty for the top
            level of a manifest.
        column: The column (or key) that holds the value.
    """

    file: str
    section: str
    where: tuple[tuple[str, str], ...]
    column: str

    def to_text(self) -> str:
        """The source as the text of an MLflow tag (JSON)."""
        content = {
            "file": self.file,
            "section": self.section,
            "where": dict(self.where),
            "column": self.column,
        }
        return json.dumps(content)


def source_from_text(text: str) -> MetricSource:
    """Parse the text of a source tag (MetricSource.to_text).

    Raises:
        ValueError: If the text is not such a source.
    """
    content = json.loads(text)
    if not isinstance(content, dict) or sorted(content) != ["column", "file", "section", "where"]:
        raise ValueError(f"not a metric source: {text!r}")
    where = tuple((str(key), str(value)) for key, value in content["where"].items())
    return MetricSource(
        file=content["file"], section=content["section"], where=where, column=content["column"]
    )


@dataclass(frozen=True)
class TrackedMetric:
    """One metric of a run of the view.

    Attributes:
        key: The MLflow metric name.
        value: The value (float64).
        source: Where the value was read from.
    """

    key: str
    value: float
    source: MetricSource


@dataclass(frozen=True)
class TrackedRun:
    """One run of the view.

    Attributes:
        experiment: Name of the MLflow experiment.
        record_id: A stable, unique id of the record, e.g. "stage1/V1". A record
            that is already in the store is not logged again.
        run_name: The name shown in the MLflow UI.
        record_kind: "imported_record" or "pipeline_run".
        params: MLflow parameters (text).
        tags: MLflow tags (text), without the tags this module adds itself.
        metrics: The metrics with their sources.
    """

    experiment: str
    record_id: str
    run_name: str
    record_kind: str
    params: dict[str, str]
    tags: dict[str, str]
    metrics: tuple[TrackedMetric, ...]

    def all_tags(self) -> dict[str, str]:
        """Every tag of the run: its own, the record kind and id, and one source per metric."""
        tags = dict(self.tags)
        tags[RECORD_KIND_TAG] = self.record_kind
        tags[RECORD_ID_TAG] = self.record_id
        for metric in self.metrics:
            tags[SOURCE_TAG_PREFIX + metric.key] = metric.source.to_text()
        return tags


# ---------------------------------------------------------------- small helpers


def metric_key(name: str) -> str:
    """A metric name MLflow accepts: "@" becomes "_at_", other odd characters "_".

    E.g. "map@12" -> "map_at_12".
    """
    return NOT_ALLOWED_IN_KEY.sub("_", name.replace(AT_SIGN, AT_WORD))


def as_number(text: str) -> float | None:
    """The text as a finite float, or None if it is not one ("", "True", "none", "nan")."""
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def read_table(path: Path) -> list[dict[str, str]]:
    """All rows of a CSV result table as dicts (column -> text).

    Raises:
        FileNotFoundError: If the table does not exist.
    """
    if not path.is_file():
        raise FileNotFoundError(f"result table not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def select_row(
    rows: Sequence[Mapping[str, Any]], where: Sequence[tuple[str, str]], what: str
) -> Mapping[str, Any]:
    """The ONE row whose columns have the given texts.

    Raises:
        ValueError: If no row or more than one row matches. A metric must come
            from exactly one recorded row.
    """
    matching = [row for row in rows if all(str(row.get(key)) == value for key, value in where)]
    if len(matching) != 1:
        raise ValueError(f"{what}: expected one row with {dict(where)}, found {len(matching)}")
    return matching[0]


def check_unique_keys(run: TrackedRun) -> TrackedRun:
    """Return the run after checking that no metric key repeats.

    Raises:
        ValueError: If two metrics got the same key (two names that differ only
            in characters MLflow does not allow).
    """
    keys = [metric.key for metric in run.metrics]
    if len(set(keys)) != len(keys):
        repeated = sorted({key for key in keys if keys.count(key) > 1})
        raise ValueError(f"{run.record_id}: metric key(s) {repeated} repeat")
    return run


def _relative(repo_root: Path, path: Path) -> str:
    """A path relative to the repository root, with "/" (absolute if it lies outside)."""
    if path.is_relative_to(repo_root):
        return path.relative_to(repo_root).as_posix()
    return path.as_posix()


def _table_metric(
    key: str,
    rows: Sequence[Mapping[str, str]],
    file: str,
    where: tuple[tuple[str, str], ...],
    column: str,
) -> TrackedMetric:
    """A metric read from one row of a CSV table.

    Raises:
        ValueError: If the row is not unique or the value is not a finite number.
    """
    row = select_row(rows, where, file)
    value = as_number(row[column])
    if value is None:
        raise ValueError(f"{file}: {column} of the row {dict(where)} is not a number")
    return TrackedMetric(key=key, value=value, source=MetricSource(file, "", where, column))


# ---------------------------------------------------------------- stage-1 weekly records


@dataclass(frozen=True)
class WeeklyTables:
    """The stage-1 tables that hold the weekly records of some folds.

    Attributes:
        recall_file: Recall of each channel and of the union, one row per (fold, scope).
        metrics_file: Ranker and baseline metrics, one row per
            (experiment, config_id, fold, num_trees).
    """

    recall_file: Path
    metrics_file: Path


def stage1_weekly_run(
    fold: Fold, tables: WeeklyTables, lock: RankerLock, experiment: str, repo_root: Path
) -> TrackedRun:
    """The imported record of one week: recall per channel, union recall and MAP@12.

    Args:
        fold: The week.
        tables: The recall table and the metric table that hold the week.
        lock: The Ranker lock: the id and the number of trees of the locked
            configuration select the Ranker's row.
        experiment: Name of the MLflow experiment.
        repo_root: The repository root (source paths are relative to it).

    Raises:
        ValueError: If a row is missing, not unique, or not a number.
    """
    recall_rows = read_table(tables.recall_file)
    metric_rows = read_table(tables.metrics_file)
    recall_file = _relative(repo_root, tables.recall_file)
    metrics_file = _relative(repo_root, tables.metrics_file)
    config_id = lock.selection.config_id
    ranker_where = (
        ("experiment", FULL_EXPERIMENT),
        ("config_id", config_id),
        ("fold", fold.name),
        ("num_trees", str(lock.model.num_trees)),
    )
    baseline_where = (
        ("experiment", RRF_BASELINE),
        ("config_id", RRF_BASELINE),
        ("fold", fold.name),
        ("num_trees", RRF_NUM_TREES),
    )
    metrics = [
        _table_metric(
            f"recall.{channel}",
            recall_rows,
            recall_file,
            (("fold", fold.name), ("scope", channel)),
            RECALL_COLUMN,
        )
        for channel in ALL_CHANNELS
    ]
    union_where = (("fold", fold.name), ("scope", UNION_SCOPE))
    metrics.append(
        _table_metric("union_recall", recall_rows, recall_file, union_where, RECALL_COLUMN)
    )
    metrics.append(
        _table_metric("map_at_12.ranker", metric_rows, metrics_file, ranker_where, MAP_COLUMN)
    )
    metrics.append(
        _table_metric(
            "map_at_12.rrf_baseline", metric_rows, metrics_file, baseline_where, MAP_COLUMN
        )
    )
    ranker_row = select_row(metric_rows, ranker_where, metrics_file)
    run = TrackedRun(
        experiment=experiment,
        record_id=f"stage1/{fold.name}",
        run_name=f"{fold.name} stage-1 record (imported)",
        record_kind=IMPORTED_RECORD,
        params={
            "fold": fold.name,
            "phase": fold.phase,
            "anchor": fold.anchor.isoformat(),
            "label_end": fold.label_end.isoformat(),
            "ranker_config_id": config_id,
            "ranker_num_trees": str(lock.model.num_trees),
            "n_queries": ranker_row["n_queries"],
        },
        tags={
            "hmrec.note": "Imported from the stage-1 result tables. Not a pipeline run.",
            "hmrec.platform": "darwin-arm64 (the laptop, stage 1)",
            "hmrec.recall_table": recall_file,
            "hmrec.metrics_table": metrics_file,
            "hmrec.ranker_git_commit": ranker_row["git_commit"],
            "hmrec.recorded_at": ranker_row["finished_at"],
        },
        metrics=tuple(metrics),
    )
    return check_unique_keys(run)


# ---------------------------------------------------------------- M10 calibration


def _group_of(row: Mapping[str, str]) -> tuple[str, str, str]:
    """(run, item, scope) of a calibration row."""
    return (row["run"], row["item"], row["scope"])


def _calibration_metrics(
    rows: Sequence[Mapping[str, str]], group: tuple[str, str, str], file: str
) -> tuple[list[TrackedMetric], dict[str, str]]:
    """(metrics, text values) of the comparison rows of one (run, item, scope).

    A numeric value becomes a metric; a measured value that is not a number
    ("True", "deterministic") becomes a parameter, so it is still visible.
    """
    metrics: list[TrackedMetric] = []
    texts: dict[str, str] = {}
    for row in rows:
        if _group_of(row) != group:
            continue
        where = (*zip(CALIBRATION_GROUP, group, strict=True), ("quantity", row["quantity"]))
        for column, suffix in CALIBRATION_VALUE_COLUMNS:
            value = as_number(row[column])
            if value is not None:
                key = metric_key(row["quantity"]) + suffix
                metrics.append(TrackedMetric(key, value, MetricSource(file, "", where, column)))
        if as_number(row["measured"]) is None:
            texts[f"measured.{metric_key(row['quantity'])}"] = row["measured"]
    return metrics, texts


def _process_metrics(
    rows: Sequence[Mapping[str, str]], group: tuple[str, str, str], file: str
) -> list[TrackedMetric]:
    """Time and memory of the calibration process of one (run, item, scope), if it has a row."""
    where = tuple(zip(CALIBRATION_GROUP, group, strict=True))
    if not any(_group_of(row) == group for row in rows):
        return []
    return [
        _table_metric(f"process.{column}", rows, file, where, column)
        for column in CALIBRATION_PROCESS_COLUMNS
    ]


def calibration_runs(
    comparisons_file: Path, runs_file: Path, experiment: str, repo_root: Path
) -> list[TrackedRun]:
    """The imported records of the M10 calibration, one per (run, item, scope).

    Args:
        comparisons_file: results/m10_calibration/comparisons.csv (one row per quantity).
        runs_file: results/m10_calibration/runs.csv (one row per process).
        experiment: Name of the MLflow experiment.
        repo_root: The repository root.

    Returns:
        The runs in the order their first row appears in the comparison table.
    """
    comparison_rows = read_table(comparisons_file)
    process_rows = read_table(runs_file)
    comparisons = _relative(repo_root, comparisons_file)
    processes = _relative(repo_root, runs_file)
    groups = list(dict.fromkeys(_group_of(row) for row in comparison_rows))
    tracked: list[TrackedRun] = []
    for group in groups:
        run_name, item, scope = group
        first = next(row for row in comparison_rows if _group_of(row) == group)
        metrics, texts = _calibration_metrics(comparison_rows, group, comparisons)
        metrics.extend(_process_metrics(process_rows, group, processes))
        run = TrackedRun(
            experiment=experiment,
            record_id=f"m10/{run_name}/{item}/{scope}",
            run_name=f"{run_name} item {item} {scope} (imported)",
            record_kind=IMPORTED_RECORD,
            params={"run": run_name, "item": item, "scope": scope, **texts},
            tags={
                "hmrec.note": "Imported from the M10 calibration tables. Not a pipeline run.",
                "hmrec.platform": f"{first['os']}-{first['architecture']}",
                "hmrec.git_commit": first["git_commit"],
                "hmrec.image_commit": first["image_commit"],
                "hmrec.code_tree_sha256": first["code_tree_sha256"],
                "hmrec.openmp_version": first["openmp_version"],
                "hmrec.recorded_at": first["finished_at"],
            },
            metrics=tuple(metrics),
        )
        tracked.append(check_unique_keys(run))
    return tracked


# ---------------------------------------------------------------- weekly pipeline


def _week_prefix(manifest: RunManifest, week: date) -> str:
    """Which of the tick's two weeks a metric describes: "opened_week" or "closed_week".

    Raises:
        ValueError: For any other week. A tick opens the week of its anchor and
            closes the week before it (docs/mlops.md §6.1); it describes no other.
    """
    if week == manifest.anchor:
        return "opened_week"
    if week == manifest.anchor - timedelta(days=DAYS_PER_WEEK):
        return "closed_week"
    raise ValueError(
        f"{manifest.run_id}: a metric describes week {week}, which is neither the tick's "
        f"anchor {manifest.anchor} nor the week before it"
    )


def pipeline_run(
    manifest: RunManifest,
    manifest_file: Path,
    experiment: str,
    repo_root: Path,
    record_prefix: str = "pipeline",
) -> TrackedRun:
    """The run of the view for one tick of the weekly pipeline.

    Args:
        manifest: The tick's manifest (hmrec.run_manifest.read_run_manifest).
        manifest_file: The file it was read from.
        experiment: Name of the MLflow experiment.
        repo_root: The repository root.
        record_prefix: First part of the record id. Two states can hold a run
            of the same id (the replay on the laptop and the replay in the
            cloud both have "2020-07-29-eval-a1"); the prefix keeps their
            records apart.
    """
    file = _relative(repo_root, manifest_file)
    metrics = [
        TrackedMetric(
            key=f"{_week_prefix(manifest, metric.week)}/{metric_key(metric.name)}",
            value=metric.value,
            source=MetricSource(
                file,
                MANIFEST_METRICS,
                (("name", metric.name), ("week", metric.week.isoformat())),
                "value",
            ),
        )
        for metric in manifest.metrics
    ]
    for step in manifest.steps:
        where = (("name", step.name),)
        for column in ("seconds", "process_max_rss_gb"):
            key = f"step.{metric_key(step.name)}.{column}"
            value = float(getattr(step, column))
            metrics.append(
                TrackedMetric(key, value, MetricSource(file, MANIFEST_STEPS, where, column))
            )
    for column in ("total_seconds", "driver_max_rss_gb", "max_process_rss_gb"):
        value = float(getattr(manifest, column))
        metrics.append(TrackedMetric(column, value, MetricSource(file, MANIFEST_TOP, (), column)))
    code = manifest.code
    run = TrackedRun(
        experiment=experiment,
        record_id=f"{record_prefix}/{manifest.run_id}",
        run_name=manifest.run_id,
        record_kind=PIPELINE_RUN,
        params={
            "run_id": manifest.run_id,
            "anchor": manifest.anchor.isoformat(),
            "mode": manifest.mode,
            "status": manifest.status,
            "n_inputs": str(len(manifest.inputs)),
            "n_outputs": str(len(manifest.outputs)),
        },
        tags={
            "hmrec.manifest": file,
            "hmrec.git_commit": code.git_commit,
            "hmrec.git_dirty": str(code.git_dirty),
            "hmrec.code_tree_sha256": code.code_tree_sha256,
            "hmrec.image_digest": code.image_digest or "unknown",
            "hmrec.platform": manifest.platform.name,
            "hmrec.failure": manifest.failure,
            "hmrec.started_at_utc": manifest.started_at_utc.isoformat(timespec="seconds"),
            "hmrec.finished_at_utc": manifest.finished_at_utc.isoformat(timespec="seconds"),
        },
        metrics=tuple(metrics),
    )
    return check_unique_keys(run)
