"""Tests for hmrec.diagnostics.context: the configs and the identity of the locked Ranker."""

import shutil
from pathlib import Path

import pytest
from diag_example import build_diag_workspace
from final_example import LOCKED_TREES, copy_workspace
from ranker_example import WIDE_COLUMNS

from hmrec.diagnostics.context import check_metrics, load_diagnostics_context
from hmrec.ranker.runtime import read_lightgbm_runtime
from hmrec.run_records import RunStamp

REPO = Path(__file__).resolve().parents[2]


def test_the_context_takes_the_ranker_identity_from_the_lock(tmp_path: Path) -> None:
    # The tiny M8 workspace has a consistent Ranker lock; add the two configs it lacks.
    final_context = copy_workspace(tmp_path)
    paths = final_context.paths
    for name in ("m8_diagnostics.yaml", "preprocess.yaml"):
        shutil.copyfile(REPO / "configs" / name, paths.configs_dir / name)
    context = load_diagnostics_context(paths, RunStamp("c" * 40, False), read_lightgbm_runtime())
    assert context.ranker.config_id == final_context.locked.config_id
    assert context.ranker.num_trees == LOCKED_TREES
    assert context.ranker.top_k == final_context.locked.config.top_k == 12
    assert context.ranker.wide_columns == WIDE_COLUMNS
    assert context.metrics == ("map@12", "recall@12", "ndcg@12")
    assert (context.map_metric, context.recall_metric) == ("map@12", "recall@12")
    assert context.config.bootstrap_replicates == 1000
    assert context.preprocess.day_zero.isoformat() == "2018-09-20"


def test_folds_of_gives_the_folds_of_the_phases_in_time_order(tmp_path: Path) -> None:
    context = build_diag_workspace(tmp_path)
    assert [fold.name for fold in context.folds_of(("T", "V"))] == [
        "V1",
        "V2",
        "V3",
        "V4",
        "T1",
        "T2",
        "T3",
        "T4",
    ]
    assert len(context.folds_of(("D", "F", "V", "T"))) == 16


def test_the_main_metric_must_be_a_map_with_its_recall() -> None:
    check_metrics(("map@12", "recall@12", "ndcg@12"))
    with pytest.raises(ValueError, match="map@K as the main metric"):
        check_metrics(("ndcg@12", "recall@12"))
    with pytest.raises(ValueError, match="recall@K next to it"):
        check_metrics(("map@12", "recall@20"))
