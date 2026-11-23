"""A small synthetic workspace for the tests of the weekly pipeline's driver.

    workspace = build_workspace(tmp_path, shard_customers=2)

gives project paths whose data, artifacts and results lie under tmp_path, with a
copy of the repository's configs/ (so the calendar is the real one: the first
tick is V1, anchor 2020-07-29) and a tiny bootstrap state: the chain model
"after F4" and the monitoring history of F1-F4.

FakeSteps stands in for the step processes. For every step it writes small but
well-formed files into the step's staging folder, exactly the files the driver
reads (ingest.json, step.json, the model record, the chain record, feature
statistics, output signals, the closed week's metrics), and nothing is computed.
A step named in `fail` raises instead, like a process that ended with an error.
Nothing of the project's data is read.
"""

import csv
import json
import shutil
from dataclasses import asdict, dataclass, field, replace
from datetime import date, timedelta
from pathlib import Path

import polars as pl
from run_manifest_example import COMMIT, DRIVER_PLATFORM, TREE

from hmrec.config import load_base_config
from hmrec.features.stats import STATS_COLUMNS
from hmrec.locks import file_sha256
from hmrec.paths import ProjectPaths, build_paths
from hmrec.run_manifest import EVAL_MODE, MetricRecord, RunCode
from hmrec.run_records import RunStamp
from hmrec.weekly import driver
from hmrec.weekly.context import OutputFile, TickContext, load_tick_context, write_step_record
from hmrec.weekly.data_steps import IngestRecord
from hmrec.weekly.driver import StepCall
from hmrec.weekly.outputs import OutputSignals, WeekEvaluation
from hmrec.weekly.steps import (
    DATASET,
    EVALUATE,
    FEATURES,
    FEATURES_SHARED,
    FINE_TUNE,
    INGEST,
    MERGE_SCORES,
    PUBLISH,
    QUERIES,
    RECALL,
    SAMPLE,
    SCORE,
    SHARD_FEATURES,
    SHARD_SCORE,
    TRAIN,
    staging_dir,
)

REPO = Path(__file__).resolve().parents[2]
STAMP = RunStamp(COMMIT, False)
V1 = date(2020, 7, 29)
V2 = date(2020, 8, 5)
V3 = date(2020, 8, 12)
FEATURES_OF_EXAMPLE = ("pop_rank", "user_age")
CHANNELS = ("r1_popularity", "r2_repurchase", "r3_sibling", "r4_itemcf", "r5_wals", "r6_twotower")
BOOTSTRAP_FOLDS = ("F1", "F2", "F3", "F4")
N_CUSTOMERS = 5
WEEK_ROWS = 1000


def build_workspace(root: Path, shard_customers: int = 2) -> ProjectPaths:
    """Create the workspace under `root` and return its project paths.

    Args:
        root: An empty folder (tmp_path).
        shard_customers: Written into the copied configs/m12_pipeline.yaml, so
            that a serve tick of N_CUSTOMERS customers has several shards.
    """
    shutil.copytree(REPO / "configs", root / "configs")
    pipeline_file = root / "configs" / "m12_pipeline.yaml"
    text = pipeline_file.read_text(encoding="utf-8")
    pipeline_file.write_text(
        text.replace("shard_customers: 70000", f"shard_customers: {shard_customers}"),
        encoding="utf-8",
    )
    paths = build_paths(load_base_config(root / "configs" / "base.yaml"), root)
    monitoring = paths.bootstrap_dir / "monitoring"
    monitoring.mkdir(parents=True)
    model = paths.bootstrap_dir / "models" / "twotower" / "chain_locked" / "F4.pt"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"chain model after F4")
    _write_csv(
        monitoring / "m6_twotower_chain.csv",
        ["fold", "model", "sha256"],
        [{"fold": "F4", "model": "F4.pt", "sha256": file_sha256(model)}],
    )
    _write_csv(
        monitoring / "m6_candidates.csv",
        ["fold", "scope", "recall", "candidates_per_query_mean"],
        [
            {"fold": fold, "scope": scope, "recall": "0.10", "candidates_per_query_mean": "300.0"}
            for fold in BOOTSTRAP_FOLDS
            for scope in (*CHANNELS, "union")
        ],
    )
    _write_csv(
        monitoring / "m6_features.csv",
        list(STATS_COLUMNS),
        [feature_row(fold, name) for fold in BOOTSTRAP_FOLDS for name in FEATURES_OF_EXAMPLE],
    )
    return paths


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def feature_row(fold: str, feature: str, missing_rate: float = 0.2, mean: float = 10.0) -> dict:
    """One row with the columns of results/m6_features.csv."""
    row = dict.fromkeys(STATS_COLUMNS, "")
    row.update(
        fold=fold,
        feature=feature,
        n_rows="100",
        n_missing=str(round(missing_rate * 100)),
        missing_rate=repr(missing_rate),
        mean=repr(mean),
    )
    return row


def patch_driver_environment(monkeypatch) -> None:
    """Replace the two readers of the driver that need a real repository or image.

    read_run_code asks git (the workspace is not a repository) and
    read_platform_fields verifies the code tree inside an image (the workspace is
    not the image's code). Both are tested on their own.
    """
    code = RunCode(COMMIT, False, TREE, None)
    monkeypatch.setattr(driver, "read_run_code", lambda repo_root, results_root: code)
    monkeypatch.setattr(driver, "read_platform_fields", lambda repo_root: DRIVER_PLATFORM)


@dataclass
class FakeSteps:
    """A StepRunner that writes the files a real step would leave (module docstring).

    Attributes:
        paths: The workspace's project paths.
        fail: Names of steps that raise instead of writing.
        new_week_rows: Purchase lines the fake C1 reports for the new week.
        union_recall: Union recall the fake C2 reports.
        candidates_per_query: What the fake feature step reports as candidates per
            query (the fake scoring step always reports 300).
        calls: The names of the steps that were started, in order.
    """

    paths: ProjectPaths
    fail: set[str] = field(default_factory=set)
    new_week_rows: int = WEEK_ROWS
    union_recall: float = 0.30
    candidates_per_query: float = 300.0
    calls: list[str] = field(default_factory=list)

    def __call__(self, call: StepCall) -> None:
        """Run one fake step."""
        self.calls.append(call.step.name)
        context = load_tick_context(
            self.paths, call.anchor, call.mode, call.run_id, STAMP, call.step
        )
        folder = staging_dir(context.layout, call.step)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "partial.txt").write_text("written before the step ended", encoding="utf-8")
        if call.step.name in self.fail:
            failure = context.layout.step_failure_file(call.run_id, call.step.name)
            failure.parent.mkdir(parents=True, exist_ok=True)
            failure.write_text("ValueError: a check of the fake step failed\n", encoding="utf-8")
            raise RuntimeError("the step's process ended with exit code 1")
        files, metrics = self._write(context, call, folder)
        record = {
            "name": call.step.name,
            "seconds": 1.5,
            "process_max_rss_gb": 0.25,
            "platform": DRIVER_PLATFORM.columns(),
            "detail": f"fake {call.step.kind}",
            "metrics": [
                {"name": m.name, "week": m.week.isoformat(), "value": m.value} for m in metrics
            ],
            "files": [{"role": file.role, "name": file.name} for file in files],
        }
        write_step_record(folder, record)

    def _write(
        self, context: TickContext, call: StepCall, folder: Path
    ) -> tuple[list[OutputFile], list[MetricRecord]]:
        kind, step, tick, closing = call.step.kind, call.step, context.tick, context.closing
        view = context.view
        if kind == INGEST:
            anchor = tick.anchor
            record = IngestRecord(
                anchor=anchor.isoformat(),
                n_rows=10 * WEEK_ROWS,
                first_date="2018-09-20",
                last_date=(anchor - timedelta(days=1)).isoformat(),
                new_week_rows=self.new_week_rows,
                new_week_first_date=(anchor - timedelta(days=7)).isoformat(),
                new_week_last_date=(anchor - timedelta(days=1)).isoformat(),
                expected_first_date=(anchor - timedelta(days=7)).isoformat(),
                expected_last_date=(anchor - timedelta(days=1)).isoformat(),
                previous_week_rows=(WEEK_ROWS,) * 4,
            )
            (folder / "ingest.json").write_text(json.dumps(asdict(record)), encoding="utf-8")
            (folder / "transactions.parquet").write_bytes(b"rows before the anchor")
            return [OutputFile("transactions_before_anchor", "transactions.parquet")], []
        if kind == QUERIES:
            queries = pl.DataFrame(
                {"query_id": list(range(N_CUSTOMERS)), "customer_idx": list(range(N_CUSTOMERS))},
                schema={"query_id": pl.Int32, "customer_idx": pl.Int32},
            )
            queries.write_parquet(folder / "queries.parquet")
            return [OutputFile("queries", "queries.parquet")], []
        if kind == RECALL:
            (folder / "candidates.parquet").write_bytes(step.channel.encode())
            if context.mode == EVAL_MODE:
                return [OutputFile(f"candidates/{step.channel}", "candidates.parquet")], []
            return [], []
        if kind in (FEATURES, SHARD_FEATURES):
            (folder / "features.json").write_text("{}", encoding="utf-8")
            _write_csv(
                folder / "feature_stats.csv",
                list(STATS_COLUMNS),
                [feature_row(tick.name, name) for name in FEATURES_OF_EXAMPLE],
            )
            # Like the real steps of the first replay tick: the feature step and the
            # scoring step both report the candidates per query, with the same value.
            metrics = [
                MetricRecord("structure/feature_rows", tick.anchor, 1500.0),
                MetricRecord("structure/deduplicated_candidates", tick.anchor, 1500.0),
                MetricRecord("structure/duplicate_keys", tick.anchor, 0.0),
                MetricRecord("output/candidates_per_query", tick.anchor, self.candidates_per_query),
            ]
            return [], (metrics if kind == FEATURES else [])
        if kind == FEATURES_SHARED:
            (folder / "item_table.parquet").write_bytes(b"articles")
            return [], []
        if kind == DATASET:
            (folder / "dataset.json").write_text("{}", encoding="utf-8")
            return [OutputFile("ranker_dataset_record", "dataset.json")], []
        if kind == TRAIN:
            (folder / "model.txt").write_text(f"model of {tick.name}", encoding="utf-8")
            record = {"sha256": file_sha256(folder / "model.txt")}
            (folder / "model.json").write_text(json.dumps(record), encoding="utf-8")
            return [OutputFile("ranker_model", "model.txt")], []
        if kind == SHARD_SCORE:
            (folder / "top.parquet").write_bytes(b"top of a shard")
            features = view.for_shard(step.shard or 0).features_dir(tick.name)
            shutil.copyfile(features / "feature_stats.csv", folder / "feature_stats.csv")
            return [], []
        if kind in (SCORE, MERGE_SCORES):
            (folder / "top.parquet").write_bytes(b"top of the week")
            signals = OutputSignals(N_CUSTOMERS, 300.0, 0.5, 40, 100, 7)
            (folder / "signals.json").write_text(json.dumps(asdict(signals)), encoding="utf-8")
            metrics = [
                MetricRecord("output/distinct_articles", tick.anchor, 40.0),
                MetricRecord("output/candidates_per_query", tick.anchor, 300.0),
            ]
            return [OutputFile("top_k", "top.parquet")], metrics
        if kind == PUBLISH:
            (folder / "top.parquet").write_bytes(b"prediction set")
            (folder / "manifest.json").write_text("{}", encoding="utf-8")
            return [OutputFile("prediction_set", "top.parquet")], []
        if kind == EVALUATE:
            evaluation = WeekEvaluation(
                week=closing.name,
                anchor=closing.anchor.isoformat(),
                n_queries=N_CUSTOMERS,
                n_label_pairs=8,
                channel_recall_metric=dict.fromkeys(CHANNELS, "recall@100"),
                channel_recall=dict.fromkeys(CHANNELS, 0.10),
                union_recall=self.union_recall,
                zero_hit_query_share=0.4,
                ranker={"map@12": 0.03},
                rrf_baseline={"map@12": 0.02},
                hit_value=1.25,
            )
            (folder / "metrics.json").write_text(json.dumps(asdict(evaluation)), encoding="utf-8")
            metric = MetricRecord("map@12", closing.anchor, 0.03)
            return [OutputFile("closed_week/metrics", "metrics.json")], [metric]
        if kind == SAMPLE:
            (folder / "r40.parquet").write_bytes(b"sample")
            return [OutputFile("closed_week/sample", "r40.parquet")], []
        if kind == FINE_TUNE:
            model = folder / f"{closing.name}.pt"
            model.write_text(f"chain model after {closing.name}", encoding="utf-8")
            row = {"fold": closing.name, "model": model.name, "sha256": file_sha256(model)}
            _write_csv(folder / "chain_record.csv", ["fold", "model", "sha256"], [row])
            return [OutputFile("r6_chain_model", model.name)], []
        raise ValueError(f"the fake runner has no step of kind {kind!r}")


def with_step(steps: FakeSteps, **changes) -> FakeSteps:
    """A copy of a fake runner with some settings changed (the call list is shared)."""
    return replace(steps, **changes)


# ---------------------------------------------------------------- tiny processed data


def write_processed(
    paths: ProjectPaths,
    lines: list[tuple[int, int, int]],
    n_customers: int,
    n_articles: int,
    customers_without_age: tuple[int, ...] = (),
) -> None:
    """Write tiny data/processed/ tables (with a matching metadata.json) into a workspace.

    Args:
        paths: The workspace's project paths.
        lines: (customer_idx, article_idx, t_dat_day) purchase lines, any order;
            they are written sorted by (t_dat_day, customer_idx) like the real table.
            Every line has the price 0.5.
        n_customers: Customers 0 .. n_customers - 1 (in age bucket AGE_18_24).
        n_articles: Articles 0 .. n_articles - 1 of the catalog.
        customers_without_age: Customers in the "age missing" bucket instead
            (the fallback list of the service is their R1 list, M14).
    """
    from conftest import make_transactions
    from hmrec.config import load_preprocess_config
    from hmrec.data.metadata import build_processed_metadata, write_processed_metadata

    preprocess = load_preprocess_config(paths.preprocess_config_file)
    paths.processed_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(lines, key=lambda line: (line[2], line[0]))
    make_transactions(ordered).write_parquet(paths.transactions_file)
    buckets = [
        preprocess.unknown_age_bucket if customer in customers_without_age else "AGE_18_24"
        for customer in range(n_customers)
    ]
    pl.DataFrame(
        {"customer_idx": list(range(n_customers)), "age_bucket": buckets},
        schema={"customer_idx": pl.Int32, "age_bucket": pl.String},
    ).write_parquet(paths.customers_file)
    pl.DataFrame(
        {"article_idx": list(range(n_articles)), "product_code": list(range(n_articles))},
        schema={"article_idx": pl.Int32, "product_code": pl.Int32},
    ).write_parquet(paths.articles_file)
    # The id maps: a prediction set names them with their sha256 (hmrec.weekly.outputs).
    pl.DataFrame(
        {
            "customer_idx": list(range(n_customers)),
            "customer_id": [f"{customer:064x}" for customer in range(n_customers)],
        },
        schema={"customer_idx": pl.Int32, "customer_id": pl.String},
    ).write_parquet(paths.customer_id_map_file)
    pl.DataFrame(
        {
            "article_idx": list(range(n_articles)),
            "article_id": [f"{article:010d}" for article in range(n_articles)],
        },
        schema={"article_idx": pl.Int32, "article_id": pl.String},
    ).write_parquet(paths.article_id_map_file)
    metadata = build_processed_metadata(preprocess, {"transactions": len(lines)})
    write_processed_metadata(paths.processed_metadata_file, metadata)


def write_r1_candidates(path: Path, lists: dict[int, list[int]]) -> None:
    """Write an R1 candidates file: every query with its list, rank 1 first.

    Args:
        path: The candidates file (its folder is created).
        lists: query_id -> the article_idx of its list, best first.
    """
    rows = [
        (query, rank, article)
        for query, articles in lists.items()
        for rank, article in enumerate(articles, start=1)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        rows,
        schema={"query_id": pl.Int32, "rank": pl.Int32, "article_idx": pl.Int32},
        orient="row",
    ).write_parquet(path)


def tick_context(paths: ProjectPaths, anchor: date, mode: str = EVAL_MODE) -> TickContext:
    """The context of a tick whose steps write straight into the final folders (no staging)."""
    return load_tick_context(paths, anchor, mode, "test-run", STAMP)
