"""A tiny hand-made workspace for the tests of the diagnostics; not a test module itself.

Nothing of the project's data is read: every file lives under a temporary folder.
The fold dates are the real ones (configs/fold_manifest.yaml, 16 folds).

EVERY fold has the same four queries (query ids differ per fold):

    query  customer        labels   R1 candidates (rank 1, 2, 3)   R2     days since last purchase
    q0     1               {1, 2}   [1, 5, 6]                      [2]    3     -> recent_purchase
    q1     2               {1}      [5, 6, 7]                      -      NaN   -> no_history
    q2     3               {4, 8}   [5, 4, 6]                      -      45    -> other
    q3     100 + ordinal   {5}      [5, 6, 1]                      -      30    -> recent_purchase

R3 ... R6 return nothing. Customers 1, 2, 3 are in every fold; customer
100 + ordinal only in its own fold (ordinal = position of the fold, D1 = 0).

Per query, by hand (RRF with k = 60, Top-12; AP@12 divides by min(12, |labels|)):

    q0  RRF order [1, 2, 5, 6] (1 and 2 tie at 1/61 with best_rank 1 -> article_idx
        ascending). Hits at ranks 1 and 2: AP = (1/1 + 2/2) / 2 = 1.     union recall 2/2 = 1
    q1  [5, 6, 7], no hit: AP = 0.                                       union recall 0
    q2  [5, 4, 6], hit at rank 2: AP = (1/2) / 2 = 0.25.                 union recall 1/2
    q3  [5, 6, 1], hit at rank 1: AP = 1.                                union recall 1

    RRF baseline  MAP@12 = (1 + 0 + 0.25 + 1) / 4 = 0.5625
                  Recall@12 = (1 + 0 + 0.5 + 1) / 4 = 0.625
    union recall  0.625
    "Ranker"      saved per-query AP 1, 0, 0.5, 1 -> MAP@12 0.625 (recorded Recall@12 0.625)

Label pairs of a week: article 1 has 2 (q0, q1); articles 2, 4, 5, 8 have 1 each;
6 pairs, 5 articles.

Purchase lines (data/processed/transactions.parquet), for the first sale day:

    article 1, 4   day 0                      (old in every fold)
    article 2      10 days before V1's anchor (not yet sold for D1 ... F3; new for F4, V1, V2,
                                               V3: 3, 10, 17, 24 days; old from V4: 31 days)
    article 5      ON the anchor day of T1    (label-week data for T1: NOT history;
                                               7 days before T2's anchor: new for T2)
    article 8      never                      (first sale in the label week, always)
"""

from pathlib import Path
from typing import Any

import polars as pl
import yaml

from hmrec.config import (
    load_base_config,
    load_preprocess_config,
    parse_diagnostics_config,
)
from hmrec.csv_records import append_row
from hmrec.data.day_index import date_to_day_index
from hmrec.data.folds import Fold, load_fold_manifest
from hmrec.data.metadata import build_processed_metadata, write_processed_metadata
from hmrec.diagnostics.context import DiagnosticsContext, RankerIdentity
from hmrec.eval.monitor import MONITOR_COLUMNS
from hmrec.feature_files import FeatureFiles, describe_part, write_feature_files
from hmrec.final.goals import GOAL_COLUMNS
from hmrec.fusion.rrf import best_rank_expr, rrf_score_expr
from hmrec.fusion.union import RANK_COLUMNS
from hmrec.hpo.v_records import MetricRow, append_metric_row, results_files
from hmrec.locks import ALL_CHANNELS
from hmrec.paths import ProjectPaths, build_paths
from hmrec.run_records import RunStamp

REPO = Path(__file__).resolve().parents[2]
MANIFEST = load_fold_manifest(REPO / "configs" / "fold_manifest.yaml")
PREPROCESS = load_preprocess_config(REPO / "configs" / "preprocess.yaml")
COMMIT = "c" * 40  # the commit the diagnostics "run at"
FORMAL_COMMIT = "f" * 40  # the commit of the finished formal run
NOW = "2026-11-19T10:00:00"
SCRIPT = "scripts/m8_diagnostics.py"
METRICS = ("map@12", "recall@12", "ndcg@12")
CONFIG_ID = "cfg123456789"
LOCKED_TREES = 10
OTHER_TREES = 5  # rows at another number of trees, which the diagnostics must ignore
RRF_K = 60
NAN = float("nan")
DAYS = "user_days_since_last_purchase"
WIDE_COLUMNS = ("query_id", "article_idx", *RANK_COLUMNS, "rrf_score", "best_rank", DAYS)
RESULT_PHASES = ("V", "T")
AREA_OF_PHASE = {"V": "v_phase", "T": "t_phase"}

# The hand-computed numbers of every fold (module docstring).
UNION_RECALL = 0.625
RANKER_MAP = 0.625
RANKER_RECALL = 0.625
BASELINE_MAP = 0.5625
BASELINE_RECALL = 0.625
RANKER_AP = (1.0, 0.0, 0.5, 1.0)  # q0 ... q3
N_QUERIES = 4

LABELS = ({1, 2}, {1}, {4, 8}, {5})
R1_CANDIDATES = ([1, 5, 6], [5, 6, 7], [5, 4, 6], [5, 6, 1])
R2_CANDIDATES = ([2], [], [], [])
DAYS_SINCE_LAST_PURCHASE = (3.0, NAN, 45.0, 30.0)

TINY_BASE = {
    "paths": {"data_root": "data", "artifacts_root": "artifacts", "results_root": "results"},
    "seed": 0,
    "num_threads": 2,
    "memory": {"limit_gb": 8.0, "sample_interval_seconds": 0.1},
    "chunking": {
        "query_chunk_size": 10,
        "item_chunk_size": 10,
        "dense_score_query_chunk_size": 10,
        "twotower_score_query_chunk_size": 10,
        "pair_chunk_size": 10,
    },
}
TINY_DIAGNOSTICS = {
    "decomposition_phases": ["V", "T"],
    "concentration_phases": ["D", "F", "V", "T"],
    "concentration": {"top_article_counts": [1, 2], "new_article_days": 28},
    "query_group_phases": ["V", "T"],
    "query_groups": {"recent_purchase_days": 30},
    "bootstrap_phases": ["T", "V"],
    "bootstrap": {"n_replicates": 200, "seed": 7, "confidence_level": 0.95},
}


def ordinal(fold: Fold) -> int:
    """Position of the fold in the manifest (D1 = 0 ... T4 = 15)."""
    return MANIFEST.names.index(fold.name)


def query_ids(fold: Fold) -> list[int]:
    """The four query ids of a fold: 1000 * (ordinal + 1) + 0 ... 3."""
    base = 1000 * (ordinal(fold) + 1)
    return [base + position for position in range(N_QUERIES)]


def customers(fold: Fold) -> list[int]:
    """The customers of q0 ... q3 (module docstring)."""
    return [1, 2, 3, 100 + ordinal(fold)]


def anchor_day(fold_name: str) -> int:
    """Day index of a fold's anchor."""
    return date_to_day_index(MANIFEST.get(fold_name).anchor, PREPROCESS.day_zero)


# ---------------------------------------------------------------- files of one fold


def write_queries_and_labels(paths: ProjectPaths, fold: Fold) -> None:
    """artifacts/queries/{fold}.parquet and artifacts/labels/{fold}.parquet."""
    ids = query_ids(fold)
    queries = pl.DataFrame(
        {"query_id": ids, "fold": [fold.name] * N_QUERIES, "customer_idx": customers(fold)},
        schema={"query_id": pl.Int32, "fold": pl.String, "customer_idx": pl.Int32},
    ).with_columns(pl.lit(fold.anchor, dtype=pl.Date).alias("anchor"))
    pairs = [(ids[q], article) for q in range(N_QUERIES) for article in sorted(LABELS[q])]
    labels = pl.DataFrame(
        {
            "query_id": [pair[0] for pair in pairs],
            "fold": [fold.name] * len(pairs),
            "article_idx": [pair[1] for pair in pairs],
        },
        schema={"query_id": pl.Int32, "fold": pl.String, "article_idx": pl.Int32},
    )
    for path, table in (
        (paths.queries_file(fold.name), queries),
        (paths.labels_file(fold.name), labels),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        table.write_parquet(path)


def channel_candidates(fold: Fold, channel: str) -> pl.DataFrame:
    """One channel's candidates of a fold: query_id, article_idx, rank (int32)."""
    lists = {"r1_popularity": R1_CANDIDATES, "r2_repurchase": R2_CANDIDATES}.get(
        channel, ([], [], [], [])
    )
    rows = [
        (query_id, article, rank)
        for query_id, articles in zip(query_ids(fold), lists, strict=True)
        for rank, article in enumerate(articles, start=1)
    ]
    return pl.DataFrame(
        {
            "query_id": [row[0] for row in rows],
            "article_idx": [row[1] for row in rows],
            "rank": [row[2] for row in rows],
        },
        schema={"query_id": pl.Int32, "article_idx": pl.Int32, "rank": pl.Int32},
    )


def wide_table(fold: Fold) -> pl.DataFrame:
    """The fold's wide table: one row per (query, candidate) with WIDE_COLUMNS."""
    keys = ["query_id", "article_idx"]
    by_channel = {channel: channel_candidates(fold, channel) for channel in ALL_CHANNELS}
    wide = pl.concat([table.select(keys) for table in by_channel.values()]).unique()
    for channel, rank_column in zip(ALL_CHANNELS, RANK_COLUMNS, strict=True):
        ranks = by_channel[channel].select(
            *keys, pl.col("rank").cast(pl.Float32).alias(rank_column)
        )
        wide = wide.join(ranks, on=keys, how="left")
    days = pl.DataFrame(
        {"query_id": query_ids(fold), DAYS: list(DAYS_SINCE_LAST_PURCHASE)},
        schema={"query_id": pl.Int32, DAYS: pl.Float32},
    )
    wide = wide.with_columns(pl.col(RANK_COLUMNS).fill_null(NAN)).with_columns(
        rrf_score_expr(RANK_COLUMNS, RRF_K), best_rank_expr(RANK_COLUMNS)
    )
    return wide.join(days, on="query_id").select(WIDE_COLUMNS).sort(keys)


def write_candidates_and_features(paths: ProjectPaths, fold: Fold) -> None:
    """The six candidate files and the wide table in two parts, with features.json."""
    for channel in ALL_CHANNELS:
        path = paths.candidates_file(fold.name, channel)
        path.parent.mkdir(parents=True, exist_ok=True)
        channel_candidates(fold, channel).write_parquet(path)
    wide, ids = wide_table(fold), query_ids(fold)
    directory = paths.features_dir(fold.name)
    directory.mkdir(parents=True, exist_ok=True)
    described = []
    # Two parts; a query is never split over two parts (q0, q1 | q2, q3).
    for number, part_ids in enumerate((ids[:2], ids[2:])):
        path = directory / f"part-{number:05d}.parquet"
        wide.filter(pl.col("query_id").is_in(part_ids)).write_parquet(path)
        described.append(describe_part(path))
    record = FeatureFiles(
        fold.name, WIDE_COLUMNS, (), tuple(described), wide.height, N_QUERIES, {"source": "test"}
    )
    write_feature_files(directory, record)


def write_ranker_per_query(paths: ProjectPaths, fold: Fold) -> None:
    """The Ranker's saved per-query values: rows at 5 trees (to ignore) and at 10 trees."""
    rows = []
    for trees, values in ((OTHER_TREES, (0.0, 0.0, 0.0, 0.0)), (LOCKED_TREES, RANKER_AP)):
        for query_id, value in zip(query_ids(fold), values, strict=True):
            rows.append((query_id, fold.name, value, value, value, trees))
    table = pl.DataFrame(
        rows,
        schema={
            "query_id": pl.Int32,
            "fold": pl.String,
            "map@12": pl.Float64,
            "recall@12": pl.Float64,
            "ndcg@12": pl.Float64,
            "num_trees": pl.Int32,
        },
        orient="row",
    )
    area = AREA_OF_PHASE[fold.phase]
    path = paths.ranker_per_query_file(area, "full", CONFIG_ID, fold.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.write_parquet(path)


# ---------------------------------------------------------------- result tables


def _metric_row(experiment: str, config_id: str, fold: str, trees: int, scores: tuple) -> MetricRow:
    return MetricRow(
        experiment, config_id, fold, trees, N_QUERIES, dict(zip(METRICS, scores, strict=True)), None
    )


def write_metric_rows(paths: ProjectPaths, fold: Fold) -> None:
    """The fold's rows of the area's metric table: baseline, Ranker at 5 and at 10 trees."""
    metrics_file = results_files(paths, AREA_OF_PHASE[fold.phase])[1]
    rows = (
        _metric_row(
            "rrf_baseline", "rrf_baseline", fold.name, 0, (BASELINE_MAP, BASELINE_RECALL, 0.7)
        ),
        _metric_row("full", CONFIG_ID, fold.name, OTHER_TREES, (0.0, 0.0, 0.0)),
        _metric_row("full", CONFIG_ID, fold.name, LOCKED_TREES, (RANKER_MAP, RANKER_RECALL, 0.8)),
    )
    for row in rows:
        append_metric_row(metrics_file, row, METRICS, FORMAL_COMMIT, NOW)


def monitor_row(fold: Fold, scope: str, recall: str) -> dict[str, str]:
    """A monitoring row with only the cells the diagnostics read filled in."""
    row = dict.fromkeys(MONITOR_COLUMNS, "")
    row.update(
        fold=fold.name, phase=fold.phase, scope=scope, n_queries=str(N_QUERIES), recall=recall
    )
    return row


def write_monitor_rows(paths: ProjectPaths, fold: Fold) -> None:
    """The union row (and one channel row) of a fold, as M6 and M8 recorded them.

    Like the real tables: results/m6_candidates.csv has the V folds with recall
    and the T folds WITHOUT recall; results/t_phase_recall.csv has the T folds.
    """
    m6_file, t_file = paths.m6_candidates_results_file, paths.t_phase_recall_file
    if fold.phase == "V":
        append_row(m6_file, MONITOR_COLUMNS, monitor_row(fold, "r1_popularity", "0.5"))
        append_row(m6_file, MONITOR_COLUMNS, monitor_row(fold, "union", repr(UNION_RECALL)))
    else:
        append_row(m6_file, MONITOR_COLUMNS, monitor_row(fold, "union", ""))
        append_row(t_file, MONITOR_COLUMNS, monitor_row(fold, "union", repr(UNION_RECALL)))


def write_goal_rows(paths: ProjectPaths, goals: tuple[str, ...] = ("G1", "G2", "G3")) -> None:
    """results/t_phase_goals.csv: the verdict of the finished formal run."""
    for goal in goals:
        row = dict.fromkeys(GOAL_COLUMNS, "x")
        row.update(goal=goal, met="True", git_commit=FORMAL_COMMIT, decided_at=NOW)
        append_row(paths.t_phase_goals_file, GOAL_COLUMNS, row)


# ---------------------------------------------------------------- processed data


def transaction_lines() -> list[tuple[int, int, int]]:
    """(customer_idx, article_idx, t_dat_day) of the purchase lines (module docstring)."""
    return [
        (1, 1, 0),
        (1, 1, 300),  # a later sale of article 1: its FIRST sale stays day 0
        (2, 4, 0),
        (3, 2, anchor_day("V1") - 10),
        (3, 5, anchor_day("T1")),
    ]


def write_transactions(paths: ProjectPaths, lines: list[tuple[int, int, int]]) -> None:
    """data/processed/transactions.parquet and metadata.json for the real preprocess config."""
    table = pl.DataFrame(
        lines,
        schema={"customer_idx": pl.Int32, "article_idx": pl.Int32, "t_dat_day": pl.Int16},
        orient="row",
    )
    paths.transactions_file.parent.mkdir(parents=True, exist_ok=True)
    table.write_parquet(paths.transactions_file)
    metadata = build_processed_metadata(PREPROCESS, {"transactions": table.height})
    write_processed_metadata(paths.processed_metadata_file, metadata)


# ---------------------------------------------------------------- the workspace


def make_context(
    paths: ProjectPaths,
    dirty: bool = False,
    changed: tuple[str, ...] = (),
    diagnostics: dict[str, Any] | None = None,
) -> DiagnosticsContext:
    """The context of a process at COMMIT.

    Args:
        paths: The workspace's paths.
        dirty: True = the working tree has uncommitted changes.
        changed: What changed_existing reports (M8 files changed since the formal run).
        diagnostics: Raw content of m8_diagnostics.yaml; default TINY_DIAGNOSTICS.
    """
    return DiagnosticsContext(
        paths=paths,
        base=load_base_config(paths.base_config_file),
        config=parse_diagnostics_config(diagnostics or TINY_DIAGNOSTICS),
        preprocess=PREPROCESS,
        manifest=MANIFEST,
        ranker=RankerIdentity(CONFIG_ID, LOCKED_TREES, 12, WIDE_COLUMNS),
        metrics=METRICS,
        stamp=RunStamp(COMMIT, dirty),
        changed_existing=lambda commit, protected: list(changed),
    )


def build_diag_workspace(root: Path) -> DiagnosticsContext:
    """Write the whole workspace under `root` and return its context (clean tree, at COMMIT)."""
    base_file = root / "configs" / "base.yaml"
    base_file.parent.mkdir(parents=True, exist_ok=True)
    base_file.write_text(yaml.safe_dump(TINY_BASE, sort_keys=False), encoding="utf-8")
    paths = build_paths(load_base_config(base_file), root)
    for fold in MANIFEST.folds:
        write_queries_and_labels(paths, fold)
        if fold.phase in RESULT_PHASES:
            write_candidates_and_features(paths, fold)
            write_ranker_per_query(paths, fold)
            write_metric_rows(paths, fold)
            write_monitor_rows(paths, fold)
    write_goal_rows(paths)
    write_transactions(paths, transaction_lines())
    return make_context(paths)


def file_snapshot(root: Path) -> dict[str, bytes]:
    """Relative path -> content of every file under `root` (to see what a run wrote)."""
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
