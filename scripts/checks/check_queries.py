"""Independent cross-check of artifacts/queries/ and artifacts/labels/ (see README.md).

Usage:
    uv run python scripts/checks/check_queries.py [--folds D1 D2 ...]

For every fold, the label week is re-derived here with plain datetime
arithmetic, WITHOUT hmrec.data.asof, hmrec.data.queries or hmrec.data.day_index,
so a bug in those modules cannot hide itself. This is a deliberate, documented
exception to the AGENTS.md rules on date filtering and day/date conversion; it
applies only to scripts/checks/.

Checks per fold (any failure raises ValueError):
    1. query_id lies in [ordinal * offset, (ordinal + 1) * offset), is consecutive
       from ordinal * offset, fits in int32, and the fold has < offset queries.
    2. fold and anchor columns hold the fold's name and anchor.
    3. labels == distinct (customer_idx, article_idx) bought in [anchor, anchor + 7d).
    4. query customers == distinct buyers in that week.
Plus: the last fold's label week ends on the manifest's last_label_date.
Logs a per-fold table and the folds whose query count is > 30% from the mean.
"""

# Thread setup must run before polars is imported (see hmrec.threads).
from hmrec.threads import check_polars_thread_pool, configure_polars_threads

configure_polars_threads()

import argparse  # noqa: E402
import logging  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from datetime import date, timedelta  # noqa: E402

import polars as pl  # noqa: E402
import yaml  # noqa: E402

from hmrec.config import (  # noqa: E402
    INT32_MAX,
    load_base_config,
    load_preprocess_config,
    load_queries_config,
)
from hmrec.data.metadata import scan_processed_parquet  # noqa: E402
from hmrec.logging_utils import get_logger, setup_logging  # noqa: E402
from hmrec.paths import ProjectPaths, get_paths  # noqa: E402

LOGGER = get_logger("check_queries")
WEEK_DAYS = 7
OUTLIER_SHARE = 0.30  # report folds whose query count is more than 30% from the mean
PAIR = ["customer_idx", "article_idx"]


@dataclass(frozen=True)
class ManifestFold:
    """One fold as read directly from the YAML (not via hmrec.data.folds)."""

    ordinal: int
    name: str
    anchor: date
    label_end: date


@dataclass(frozen=True)
class FoldReport:
    """Counts of one fold after all its checks passed."""

    name: str
    anchor: date
    n_queries: int
    n_labels: int
    first_label_day: date
    last_label_day: date


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", nargs="+", help="fold names; default: all folds")
    return parser.parse_args()


def read_manifest(paths: ProjectPaths) -> tuple[list[ManifestFold], date]:
    """Read fold names and dates straight from fold_manifest.yaml."""
    raw = yaml.safe_load(paths.fold_manifest_file.read_text(encoding="utf-8"))
    folds = [
        ManifestFold(ordinal, entry["name"], entry["anchor"], entry["label_end"])
        for ordinal, entry in enumerate(raw["folds"])
    ]
    return folds, raw["checks"]["last_label_date"]


def require(condition: bool, message: str) -> None:
    """Raise ValueError(message) unless condition holds (works under python -O)."""
    if not condition:
        raise ValueError(message)


def week_purchases(transactions: pl.DataFrame, fold: ManifestFold, day_zero: date) -> pl.DataFrame:
    """Return purchase lines with anchor <= date < anchor + 7d, computed independently."""
    require(
        fold.label_end == fold.anchor + timedelta(days=WEEK_DAYS - 1), f"{fold.name}: not 7 days"
    )
    # Independent re-derivation of the day index (see module docstring).
    start_day = (fold.anchor - day_zero).days
    end_day_exclusive = start_day + WEEK_DAYS
    day = pl.col("t_dat_day")
    return transactions.filter((day >= start_day) & (day < end_day_exclusive))


def check_query_ids(queries: pl.DataFrame, fold: ManifestFold, offset: int) -> None:
    """Check 1 and 2: query_id range, continuity, int32 fit, fold and anchor columns."""
    first_id = fold.ordinal * offset
    expected_ids = list(range(first_id, first_id + queries.height))
    require(queries.height < offset, f"{fold.name}: {queries.height} queries >= offset {offset}")
    require(queries["query_id"].to_list() == expected_ids, f"{fold.name}: query_id not consecutive")
    require(first_id + queries.height - 1 <= INT32_MAX, f"{fold.name}: query_id exceeds int32")
    require(bool((queries["fold"] == fold.name).all()), f"{fold.name}: wrong fold column")
    require(bool((queries["anchor"] == fold.anchor).all()), f"{fold.name}: wrong anchor column")


def check_labels(
    queries: pl.DataFrame, labels: pl.DataFrame, week: pl.DataFrame, fold: ManifestFold
) -> None:
    """Check 3 and 4: labels and query customers equal what was bought that week."""
    require(bool((labels["fold"] == fold.name).all()), f"{fold.name}: wrong fold in labels")
    decoded = labels.join(queries.select("query_id", "customer_idx"), on="query_id")
    require(decoded.height == labels.height, f"{fold.name}: labels with unknown query_id")
    actual_pairs = decoded.select(PAIR).sort(PAIR)
    expected_pairs = week.select(PAIR).unique().sort(PAIR)
    require(actual_pairs.equals(expected_pairs), f"{fold.name}: labels != pairs bought that week")
    buyers = week["customer_idx"].unique().sort()
    require(queries["customer_idx"].sort().equals(buyers), f"{fold.name}: queries != buyers")


def label_day_range(week: pl.DataFrame, day_zero: date) -> tuple[date, date]:
    """Return the first and last purchase date of the label week."""
    min_day = week["t_dat_day"].min()
    max_day = week["t_dat_day"].max()
    require(isinstance(min_day, int) and isinstance(max_day, int), "label week is empty")
    assert isinstance(min_day, int) and isinstance(max_day, int)  # narrows the type for pyright
    return day_zero + timedelta(days=min_day), day_zero + timedelta(days=max_day)


def check_fold(
    paths: ProjectPaths, transactions: pl.DataFrame, fold: ManifestFold, day_zero: date, offset: int
) -> FoldReport:
    """Run all checks of one fold and return its counts."""
    queries = pl.read_parquet(paths.queries_file(fold.name))
    labels = pl.read_parquet(paths.labels_file(fold.name))
    week = week_purchases(transactions, fold, day_zero)
    check_query_ids(queries, fold, offset)
    check_labels(queries, labels, week, fold)
    first_day, last_day = label_day_range(week, day_zero)
    return FoldReport(fold.name, fold.anchor, queries.height, labels.height, first_day, last_day)


def log_summary(reports: list[FoldReport]) -> None:
    """Log the per-fold table and the folds far from the mean query count."""
    mean_queries = sum(report.n_queries for report in reports) / len(reports)
    for report in reports:
        LOGGER.info(
            "%s anchor=%s queries=%7d labels=%7d labels/query=%.2f days=%s..%s vs_mean=%+.1f%%",
            report.name,
            report.anchor,
            report.n_queries,
            report.n_labels,
            report.n_labels / report.n_queries,
            report.first_label_day,
            report.last_label_day,
            (report.n_queries / mean_queries - 1) * 100,
        )
    outliers = [r.name for r in reports if abs(r.n_queries / mean_queries - 1) > OUTLIER_SHARE]
    LOGGER.info("mean queries per fold = %.1f", mean_queries)
    LOGGER.info("folds with query count > 30%% from the mean: %s", outliers)


def main() -> None:
    """Check the selected folds; raise on the first failure."""
    args = parse_args()
    paths = get_paths()
    setup_logging(logging.INFO)
    check_polars_thread_pool(load_base_config(paths.base_config_file).num_threads)
    preprocess_config = load_preprocess_config(paths.preprocess_config_file)
    offset = load_queries_config(paths.queries_config_file).query_id_fold_offset
    folds, last_label_date = read_manifest(paths)
    transactions = (
        scan_processed_parquet(
            paths.transactions_file, paths.processed_metadata_file, preprocess_config
        )
        .select("customer_idx", "article_idx", "t_dat_day")
        .collect()
    )

    selected = [fold for fold in folds if args.folds is None or fold.name in args.folds]
    reports = [
        check_fold(paths, transactions, fold, preprocess_config.day_zero, offset)
        for fold in selected
    ]
    if selected and selected[-1].name == folds[-1].name:
        last_day = reports[-1].last_label_day
        require(last_day == last_label_date, f"last label day {last_day} != {last_label_date}")
        LOGGER.info("last fold %s ends on %s", folds[-1].name, last_day)
    log_summary(reports)
    LOGGER.info("ALL CHECKS PASSED for %d folds", len(reports))


if __name__ == "__main__":
    main()
