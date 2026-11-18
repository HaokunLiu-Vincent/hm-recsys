"""Tests for hmrec.final.context: the configs of M8 and that they fit together."""

from dataclasses import replace
from pathlib import Path

import pytest
import yaml
from final_example import COMMIT, load_context, write_configs

from hmrec.config import load_yaml
from hmrec.final.context import (
    MODE_FORMAL,
    MODE_REHEARSAL,
    area_of,
    check_final_config,
    load_final_context,
)
from hmrec.ranker.runtime import LightGBMRuntime, read_lightgbm_runtime
from hmrec.run_records import RunStamp


def test_areas_of_the_two_modes() -> None:
    assert area_of(MODE_REHEARSAL) == "m8_rehearsal"
    assert area_of(MODE_FORMAL) == "t_phase"
    with pytest.raises(ValueError, match="mode must be one of"):
        area_of("dry_run")


def test_the_context_uses_the_locked_ranker_everywhere(tmp_path: Path) -> None:
    context = load_context(write_configs(tmp_path))
    # The shared trial code reads inputs.ranker: it must be the LOCKED configuration.
    assert context.inputs.ranker is context.locked.config
    assert context.inputs.ranker.num_trees == context.locked.num_trees == 10
    assert context.inputs.ranker.sampling.ratios == (2,)  # only the locked ratio
    assert context.inputs.metrics == ("map@12", "recall@12", "ndcg@12")
    assert context.config.target_folds == ("T1", "T2", "T3", "T4")
    assert context.config.rehearsal_target_fold == "V4"
    assert (context.monitor.previous_folds, context.monitor.max_relative_recall_drop) == (4, 0.1)
    assert context.stamp == RunStamp(COMMIT, False)


def test_edits_of_ranker_yaml_do_not_reach_the_context(tmp_path: Path) -> None:
    paths = write_configs(tmp_path)
    before = load_context(paths)
    raw = load_yaml(paths.ranker_config_file)
    raw["tuned_defaults"]["num_leaves"] = 8
    raw["num_trees"] = 40
    raw["train_params"]["learning_rate"] = 0.3
    paths.ranker_config_file.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    after = load_context(paths)
    assert after.locked.config == before.locked.config
    assert after.locked.tuned.num_leaves == 4 and after.locked.num_trees == 10


def test_another_runtime_stops_the_context_from_loading(tmp_path: Path) -> None:
    paths = write_configs(tmp_path)
    real = read_lightgbm_runtime()
    other = LightGBMRuntime("0.0.1", real.libomp_path, real.libomp_version)
    with pytest.raises(ValueError, match="does not match configs/locks/ranker.lock.yaml"):
        load_final_context(paths, RunStamp(COMMIT, False), other)


def test_config_must_fit_the_manifest_and_the_lock(tmp_path: Path) -> None:
    context = load_context(write_configs(tmp_path))
    config, locked, manifest = context.config, context.locked, context.manifest
    check_final_config(config, locked, manifest)  # the real file: no error
    # A V fold as a target of the formal run.
    with pytest.raises(ValueError, match="predicts only T folds"):
        check_final_config(replace(config, target_folds=("V4", "T1")), locked, manifest)
    with pytest.raises(ValueError, match="must be in time order"):
        check_final_config(replace(config, target_folds=("T2", "T1")), locked, manifest)
    # A T fold as the rehearsal fold: the rehearsal would read T labels.
    with pytest.raises(ValueError, match="M7 predicts and evaluates only V folds"):
        check_final_config(replace(config, rehearsal_target_fold="T1"), locked, manifest)
    # The locked Ranker keeps the Top-12, so a metric at 20 cannot be computed.
    with pytest.raises(ValueError, match="needs the Top-20"):
        check_final_config(replace(config, extra_metrics=("recall@20",)), locked, manifest)
