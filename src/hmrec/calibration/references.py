"""The stage-1 records a calibration compares with, and the check of its inputs.

Stage-1 tables are READ here and never written (results/*.csv are in the image,
owned by root; AGENTS.md §3.3). Each function returns the one row that belongs
to a fold and a step; no row or more than one row is an error, because a
comparison against the wrong record would be worse than none.

This module imports neither torch nor lightgbm.
"""

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path

from hmrec.config import CalibrationConfig
from hmrec.locks import check_file_sha256
from hmrec.logging_utils import get_logger
from hmrec.paths import ProjectPaths

LOGGER = get_logger("calibration.references")

M8_REHEARSAL_AREA = "m8_rehearsal"
FULL_EXPERIMENT = "full"
REHEARSAL_DATASET_STEP = "rehearsal_dataset_full"


def read_table(path: Path) -> list[dict[str, str]]:
    """All rows of a stage-1 result table as dicts (column -> text).

    Raises:
        FileNotFoundError: If the table does not exist.
    """
    if not path.is_file():
        raise FileNotFoundError(f"stage-1 result table not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def single_row(path: Path, where: Mapping[str, str]) -> dict[str, str]:
    """The ONE row of a table whose columns have the given values.

    Args:
        path: A stage-1 result table.
        where: Column -> the text it must have, e.g. {"fold": "V4", "step": "r5_wals"}.

    Raises:
        ValueError: If no row or more than one row matches.
    """
    rows = [
        row for row in read_table(path) if all(row[name] == value for name, value in where.items())
    ]
    if len(rows) != 1:
        raise ValueError(f"{path}: expected exactly one row with {dict(where)}, found {len(rows)}")
    return rows[0]


def stage1_run_row(paths: ProjectPaths, fold: str, step: str) -> dict[str, str]:
    """The row of results/m6_runs.csv of one step of one fold (run time, memory peak)."""
    return single_row(paths.m6_runs_file, {"fold": fold, "step": step})


def stage1_candidates_row(paths: ProjectPaths, fold: str, scope: str) -> dict[str, str]:
    """The row of results/m6_candidates.csv of one channel (or "union") of one fold."""
    return single_row(paths.m6_candidates_results_file, {"fold": fold, "scope": scope})


def _require_rehearsal_area(config: CalibrationConfig) -> None:
    """Only the M8 rehearsal is known as a Ranker reference."""
    if config.ranker_reference_area != M8_REHEARSAL_AREA:
        raise ValueError(
            f"m10_calibration.ranker_reference_area must be {M8_REHEARSAL_AREA!r}, "
            f"got {config.ranker_reference_area!r}"
        )


def stage1_ranker_trial(
    paths: ProjectPaths, config: CalibrationConfig, config_id: str
) -> dict[str, str]:
    """The trial row of the stage-1 Ranker record of the calibration fold.

    results/m8_rehearsal.csv: the locked configuration trained on the fold's
    window to the locked number of trees (times, memory peak, model sha256).
    """
    _require_rehearsal_area(config)
    where = {"experiment": FULL_EXPERIMENT, "config_id": config_id, "fold": config.fold}
    return single_row(paths.m8_rehearsal_results_file, where)


def stage1_ranker_metrics(
    paths: ProjectPaths, config: CalibrationConfig, config_id: str
) -> dict[int, dict[str, float]]:
    """Number of trees -> {metric -> value} of the stage-1 Ranker record of the fold.

    Read from results/m8_rehearsal_metrics.csv (float64 values written with repr()).

    Raises:
        ValueError: If the record has no row, or two rows for one number of trees.
    """
    _require_rehearsal_area(config)
    by_trees: dict[int, dict[str, float]] = {}
    for row in read_table(paths.m8_rehearsal_metrics_file):
        key = (row["experiment"], row["config_id"], row["fold"])
        if key != (FULL_EXPERIMENT, config_id, config.fold):
            continue
        num_trees = int(row["num_trees"])
        if num_trees in by_trees:
            raise ValueError(
                f"{paths.m8_rehearsal_metrics_file} has two rows for {key} at {num_trees} trees"
            )
        by_trees[num_trees] = {metric: float(row[metric]) for metric in config.metrics}
    if not by_trees:
        raise ValueError(f"{paths.m8_rehearsal_metrics_file} has no row for {config_id}")
    return by_trees


def stage1_dataset_row(paths: ProjectPaths, config: CalibrationConfig) -> dict[str, str]:
    """The row of results/m8_runs.csv of the Dataset build of the stage-1 Ranker record."""
    _require_rehearsal_area(config)
    return single_row(paths.m8_runs_file, {"fold": config.fold, "step": REHEARSAL_DATASET_STEP})


def verify_files(root: Path, expected_sha256: Mapping[str, str], names: Sequence[str]) -> None:
    """Fail unless each named file under `root` has the sha256 the config records.

    Called before an item reads its inputs and before a comparison reads a
    stage-1 output: the calibration must measure the platform, not a changed file.

    Args:
        root: data/processed/ or artifacts/.
        expected_sha256: Relative path -> sha256 (configs/m10_calibration.yaml).
        names: The relative paths to check.

    Raises:
        KeyError: If a name has no recorded sha256.
        FileNotFoundError: If a file does not exist.
        ValueError: If a file's sha256 differs.
    """
    for name in names:
        if name not in expected_sha256:
            raise KeyError(f"configs/m10_calibration.yaml records no sha256 for {name!r}")
        check_file_sha256(root / name, expected_sha256[name])
        LOGGER.info("input verified: %s (sha256 %s)", root / name, expected_sha256[name])
