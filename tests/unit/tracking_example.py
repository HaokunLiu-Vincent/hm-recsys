"""Small result tables for the tests of hmrec.tracking, written into a temporary project.

Week V1 with hand-written values (three decimals, easy to read back):

    recall table    r1 0.101, r2 0.102, r3 0.103, r4 0.104, r5 0.105, r6 0.106, union 0.25
    metric table    the locked Ranker (config id and trees from the Ranker lock) 0.03;
                    the same configuration with 100 trees 0.02 (must NOT be picked);
                    another configuration 0.5 (must NOT be picked);
                    the RRF baseline 0.018
"""

from pathlib import Path

from hmrec.data.folds import Fold, load_fold_manifest
from hmrec.locks import ALL_CHANNELS, RankerLock, load_ranker_lock
from hmrec.tracking.records import TrackedRun, WeeklyTables, stage1_weekly_run
from hmrec.tracking.store import LoggedRun

REPO = Path(__file__).resolve().parents[2]
EXPERIMENT = "stage1 (test)"
COMMIT = "c" * 40
CHANNEL_RECALL = {
    channel: 0.101 + 0.001 * position for position, channel in enumerate(ALL_CHANNELS)
}


def ranker_lock() -> RankerLock:
    """The repository's Ranker lock (configs/ is in the image too)."""
    return load_ranker_lock(REPO / "configs" / "locks" / "ranker.lock.yaml")


def fold_v1() -> Fold:
    """Week V1 of the repository's fold manifest."""
    return load_fold_manifest(REPO / "configs" / "fold_manifest.yaml").get("V1")


def write_weekly_tables(root: Path) -> WeeklyTables:
    """Write the two tables of the module docstring under root/results/."""
    lock = ranker_lock()
    config_id, trees = lock.selection.config_id, lock.model.num_trees
    results = root / "results"
    results.mkdir(parents=True, exist_ok=True)
    recall_lines = ["fold,scope,recall"]
    recall_lines += [f"V1,{channel},{value!r}" for channel, value in CHANNEL_RECALL.items()]
    recall_lines += ["V1,union,0.25", "V2,union,0.99"]
    (results / "recall.csv").write_text("\r\n".join(recall_lines) + "\r\n", encoding="utf-8")
    header = "experiment,config_id,fold,num_trees,n_queries,map@12,git_commit,finished_at"
    metric_lines = [
        header,
        f"rrf_baseline,rrf_baseline,V1,0,1000,0.018,{COMMIT},2026-11-18T13:17:22",
        f"full,{config_id},V1,100,1000,0.02,{COMMIT},2026-11-18T13:20:00",
        f"full,{config_id},V1,{trees},1000,0.03,{COMMIT},2026-11-18T13:25:00",
        f"full,aaaaaaaaaaaa,V1,{trees},1000,0.5,{COMMIT},2026-11-18T13:30:00",
        f"full,{config_id},V2,{trees},1000,0.9,{COMMIT},2026-11-18T13:35:00",
    ]
    (results / "metrics.csv").write_text("\r\n".join(metric_lines) + "\r\n", encoding="utf-8")
    return WeeklyTables(results / "recall.csv", results / "metrics.csv")


def weekly_run(root: Path) -> TrackedRun:
    """The imported record of V1 built from the tables of write_weekly_tables."""
    return stage1_weekly_run(fold_v1(), write_weekly_tables(root), ranker_lock(), EXPERIMENT, root)


def as_logged(run: TrackedRun) -> LoggedRun:
    """A run as it would come back from the store after a correct sync."""
    return LoggedRun(
        experiment=run.experiment,
        record_id=run.record_id,
        mlflow_run_id="0" * 32,
        metrics={metric.key: metric.value for metric in run.metrics},
        tags=run.all_tags(),
    )
