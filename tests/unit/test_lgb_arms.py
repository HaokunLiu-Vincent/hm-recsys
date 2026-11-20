"""Tests for hpo.lgb_arms on the tiny M8 workspace (final_example): real, tiny LightGBM models.

The workspace has the tiny Ranker lock (4 leaves, 10 trees, checkpoints every 10,
feature_fraction and bagging_fraction 0.8) and what M7 left for V4: its binary
Dataset, its model trained to 20 trees with the locked seed 42, and its rows in
results/v_phase*.csv. The arms train on that same Dataset.
"""

import ast
import dataclasses
from pathlib import Path

import lightgbm as lgb
import polars as pl
import pytest
from final_example import copy_workspace
from ranker_example import METRICS, REPO

from hmrec.hpo import lgb_arms
from hmrec.hpo.lgb_arm_records import ARM_BINARY, ARM_LAMBDARANK, TRIAL_COLUMNS, metric_columns
from hmrec.hpo.lgb_arms import (
    ArmContext,
    ArmSpec,
    arm_config,
    arm_ranker_config,
    arm_train_params,
    check_arm_spec,
    differs_from_m7_model,
    load_arm_context,
    locked_config_id_of_arm,
    metric_rows,
    random_parameters,
    run_arm_trial,
    trial_row,
)
from hmrec.locks import load_ranker_lock
from hmrec.ranker.identity import ranker_config_id
from hmrec.ranker.runtime import read_lightgbm_runtime

FOLD = "V4"  # the fold M7's tiny reference was written for
LOCKED_TREES = 10
REAL_LOCK = load_ranker_lock(REPO / "configs" / "locks" / "ranker.lock.yaml")
MODULES = ("lgb_arms.py", "lgb_arm_records.py", "lgb_arms_run.py")
SCRIPTS = ("m9_lgb_trial.py", "m9_lgb_comparison.py")


def _context(root: Path) -> ArmContext:
    paths = copy_workspace(root).paths
    return load_arm_context(paths, read_lightgbm_runtime(), METRICS)


def _spec(arm: str, seed: int, num_trees: int = LOCKED_TREES) -> ArmSpec:
    return ArmSpec(arm, seed, FOLD, num_trees, (num_trees,))


# ---------------------------------------------------------------- parameters


def test_what_is_random_in_the_locked_parameters() -> None:
    # The real lock: the features and the rows of every tree are drawn at random.
    assert random_parameters(dict(REAL_LOCK.model.train_params)) == {
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
    }
    fixed = {"feature_fraction": 1.0, "bagging_fraction": 1.0, "bagging_freq": 1}
    assert random_parameters(fixed) == {}
    # A bagging fraction below 1 draws nothing unless bagging is switched on.
    assert random_parameters({**fixed, "bagging_fraction": 0.5, "bagging_freq": 0}) == {}
    assert random_parameters({**fixed, "feature_fraction_bynode": 0.5}) == {
        "feature_fraction_bynode": 0.5
    }
    # The locked training passes the one parameter `seed`; no sub-seed is in the lock.
    assert not [name for name in REAL_LOCK.model.train_params if name.endswith("seed")]


def test_the_binary_arm_changes_the_objective_and_nothing_else() -> None:
    locked = dict(REAL_LOCK.model.train_params)
    assert arm_train_params(locked, ARM_LAMBDARANK) == locked
    binary = arm_train_params(locked, ARM_BINARY)
    changed = {name for name in locked if binary[name] != locked[name]}
    assert changed == {"objective", "metric"} and set(binary) == set(locked)
    assert (binary["objective"], binary["metric"]) == ("binary", "binary_logloss")
    assert (binary["learning_rate"], binary["feature_fraction"]) == (0.05, 0.8)
    assert locked["objective"] == "lambdarank"  # the input is not changed
    with pytest.raises(ValueError, match="arm must be one of"):
        arm_train_params(locked, "xendcg")


def test_configuration_ids_of_the_arms(tmp_path: Path) -> None:
    context = _context(tmp_path)
    locked = context.locked
    # lambdarank with the locked seed IS the locked configuration (the M7 record).
    assert locked_config_id_of_arm(context, ARM_LAMBDARANK, 42) == locked.config_id
    ids = {
        locked_config_id_of_arm(context, arm, seed)
        for arm in (ARM_LAMBDARANK, ARM_BINARY)
        for seed in (42, 43, 44)
    }
    assert len(ids) == 6
    # The number of trees is not part of the id (the first k trees do not depend on it).
    longer = arm_config(context, _spec(ARM_BINARY, 43, 20))
    assert ranker_config_id(longer) == locked_config_id_of_arm(context, ARM_BINARY, 43)
    # Only `seed` carries the arm's seed; the sampling seed stays the locked one.
    ranker = arm_ranker_config(locked, ARM_BINARY, 43, 20)
    assert (ranker.seed, ranker.sampling.seed, ranker.num_trees) == (43, 42, 20)
    assert ranker.dataset_params == locked.config.dataset_params
    assert longer["seed"] == 43 and longer["sampling_seed"] == 42


# ---------------------------------------------------------------- one training


def test_lambdarank_with_another_seed_trains_other_trees(tmp_path: Path) -> None:
    context = _context(tmp_path)
    paths = context.paths
    before = {
        path: path.read_bytes() for path in (paths.v_phase_results_file, paths.v_phase_metrics_file)
    }
    spec = _spec(ARM_LAMBDARANK, 43)
    outcome = run_arm_trial(context, spec, "commit")
    assert outcome.config_id == locked_config_id_of_arm(context, ARM_LAMBDARANK, 43)
    assert outcome.train_folds == ("F4", "V1", "V2", "V3")
    assert [row.num_trees for row in outcome.metric_rows] == [LOCKED_TREES]
    row = outcome.metric_rows[0]
    assert set(row.scores) == set(METRICS) and 0 < row.scores["map@12"] <= 1
    # Its files are under m9_lgb/, named by arm and seed.
    per_query = paths.m9_lgb_per_query_file(ARM_LAMBDARANK, 43, FOLD)
    assert per_query.parts[-3:] == ("m9_lgb", "lambdarank_s43", "V4_per_query.parquet")
    assert pl.read_parquet(per_query)["num_trees"].unique().to_list() == [LOCKED_TREES]
    assert paths.m9_lgb_predictions_file(ARM_LAMBDARANK, 43, FOLD).is_file()
    model_file = paths.m9_lgb_model_file(ARM_LAMBDARANK, 43, FOLD)
    assert lgb.Booster(model_file=model_file).num_trees() == LOCKED_TREES
    # Another seed draws other features and rows: the trees are not M7's trees.
    assert differs_from_m7_model(context, spec, model_file) is True
    # Nothing was written to the M7 tables.
    assert all(path.read_bytes() == content for path, content in before.items())
    # A finished training is never replaced.
    with pytest.raises(FileExistsError, match="already exist"):
        run_arm_trial(context, spec, "commit")


def test_the_arm_pipeline_with_the_locked_seed_repeats_the_m7_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The locked seed is refused in a real run (it is the M7 record). With the refusal
    # switched off, the SAME code path must give exactly M7's first trees: this shows
    # that an arm differs from the locked training in its seed (and objective) only.
    context = _context(tmp_path)
    spec = _spec(ARM_LAMBDARANK, context.locked.config.seed)
    with pytest.raises(ValueError, match="is the M7 record"):
        run_arm_trial(context, spec, "commit")
    monkeypatch.setattr(lgb_arms, "check_arm_spec", lambda context, spec: None)
    run_arm_trial(context, spec, "commit")
    model_file = context.paths.m9_lgb_model_file(ARM_LAMBDARANK, spec.seed, FOLD)
    assert differs_from_m7_model(context, spec, model_file) is False


def test_binary_arm_trains_with_the_binary_objective_and_repeats_for_a_seed(tmp_path: Path) -> None:
    context = _context(tmp_path / "a")
    spec = ArmSpec(ARM_BINARY, 42, FOLD, 20, (10, 20))
    outcome = run_arm_trial(context, spec, "commit")
    assert [row.num_trees for row in outcome.metric_rows] == [10, 20]
    assert outcome.config["train_params"]["objective"] == "binary"
    model_file = context.paths.m9_lgb_model_file(ARM_BINARY, 42, FOLD)
    assert "objective=binary" in model_file.read_text(encoding="utf-8")
    per_query = pl.read_parquet(context.paths.m9_lgb_per_query_file(ARM_BINARY, 42, FOLD))
    assert sorted(per_query["num_trees"].unique().to_list()) == [10, 20]
    # The same seed in another workspace gives the same model; another seed does not.
    other = _context(tmp_path / "b")
    again = run_arm_trial(other, spec, "commit")
    assert again.model_sha256 == outcome.model_sha256
    assert [row.scores for row in again.metric_rows] == [row.scores for row in outcome.metric_rows]
    seed_43 = run_arm_trial(other, dataclasses.replace(spec, seed=43), "commit")
    assert seed_43.model_sha256 != outcome.model_sha256
    # The rows the script writes.
    rows = metric_rows(spec, outcome, METRICS, "commit", "now")
    assert [(row["arm"], row["seed"], row["num_trees"]) for row in rows] == [
        ("binary", "42", "10"),
        ("binary", "42", "20"),
    ]
    assert set(rows[0]) == set(metric_columns(METRICS))
    assert float(rows[1]["map@12"]) == outcome.metric_rows[1].scores["map@12"]
    measured = {"total_seconds": 1.5, "process_max_rss_gb": 0.25}
    stamp = {"git_commit": "commit", "git_dirty": "False", "finished_at": "now"}
    row = trial_row(context, spec, outcome, read_lightgbm_runtime(), measured, stamp)
    assert set(row) == set(TRIAL_COLUMNS)
    assert (row["objective"], row["evaluated_trees"], row["num_trees"]) == ("binary", "10 20", "20")
    assert (row["process_max_rss_gb"], row["train_folds"]) == ("0.25", "F4 V1 V2 V3")


# ---------------------------------------------------------------- what a training refuses


@pytest.mark.parametrize("fold", ["T1", "T2", "T3", "T4"])
def test_t_folds_are_refused_before_anything_is_read(tmp_path: Path, fold: str) -> None:
    context = _context(tmp_path)
    spec = dataclasses.replace(_spec(ARM_BINARY, 43), target_fold=fold)
    with pytest.raises(ValueError, match="T folds are evaluated in M8"):
        run_arm_trial(context, spec, "commit")
    assert not context.paths.m9_lgb_model_file(ARM_BINARY, 43, fold).parent.exists()
    assert not context.paths.m9_lgb_predictions_file(ARM_BINARY, 43, fold).parent.exists()


def test_bad_specs_are_refused(tmp_path: Path) -> None:
    context = _context(tmp_path)
    with pytest.raises(ValueError, match="M7 predicts and evaluates only V folds"):
        check_arm_spec(context, dataclasses.replace(_spec(ARM_BINARY, 43), target_fold="F4"))
    with pytest.raises(ValueError, match="arm must be one of"):
        check_arm_spec(context, _spec("xendcg", 43))
    with pytest.raises(ValueError, match="trains the locked 10 trees"):
        check_arm_spec(context, _spec(ARM_LAMBDARANK, 43, 20))
    with pytest.raises(ValueError, match="evaluated_trees must be ascending multiples of 10"):
        check_arm_spec(context, ArmSpec(ARM_BINARY, 43, FOLD, 20, (5, 20)))
    with pytest.raises(ValueError, match="evaluated_trees"):
        check_arm_spec(context, ArmSpec(ARM_BINARY, 43, FOLD, 20, (30,)))
    with pytest.raises(ValueError, match="evaluated_trees"):
        check_arm_spec(context, ArmSpec(ARM_BINARY, 43, FOLD, 20, ()))
    with pytest.raises(ValueError, match="seed and num_trees must be > 0"):
        check_arm_spec(context, _spec(ARM_BINARY, 0))
    check_arm_spec(context, _spec(ARM_BINARY, 42, 20))  # binary may use the locked seed


# ---------------------------------------------------------------- code rules


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def test_the_arm_code_is_on_the_lightgbm_side_and_keeps_to_its_own_tables() -> None:
    files = [REPO / "src" / "hmrec" / "hpo" / name for name in MODULES]
    files += [REPO / "scripts" / name for name in SCRIPTS]
    torch_side = ("torch", "hmrec.recall", "hmrec.deepfm", "hmrec.hpo.runner", "hmrec.hpo.d_phase")
    for path in files:
        for name in _imports(path):
            assert not any(name == bad or name.startswith(bad + ".") for bad in torch_side), (
                f"{path.name} imports {name}"
            )
            # No selection code of M7 and nothing of M8 or the diagnostics.
            assert not name.startswith(("hmrec.final", "hmrec.diagnostics")), path.name
        text = path.read_text(encoding="utf-8")
        # The M7 tables are never written from here, and T labels have no door here.
        assert "v_phase_metrics_file" not in text and "v_phase_results_file" not in text, path.name
        assert "FormalRunPermit" not in text and "DiagnosticAccess" not in text, path.name
        assert "label_week_transactions" not in text, path.name
    # The one place that names a label file hands it to the loader that refuses non-V folds.
    arms = (REPO / "src" / "hmrec" / "hpo" / "lgb_arms.py").read_text(encoding="utf-8")
    assert arms.count("labels_file(") == 1 and "load_m7_evaluator(" in arms
    assert "load_locked_ranker(" in arms and "load_ranker_config" not in arms
    # The two record / driver modules load neither LightGBM nor torch.
    for name in ("lgb_arm_records.py", "lgb_arms_run.py"):
        assert "lightgbm" not in _imports(REPO / "src" / "hmrec" / "hpo" / name), name
