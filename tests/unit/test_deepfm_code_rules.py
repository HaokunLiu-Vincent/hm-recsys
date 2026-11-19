"""Rules of the DeepFM study, checked on the configs and by reading the source (proposal §16).

- The package is on the torch side: it imports no LightGBM and no module that
  loads it (two OpenMP runtimes in one process abort or hang, proposal §9.7).
- It reads labels only through the M7 evaluator and never names a T-label door.
- The default model and the rules of the study are the ones that were recorded.
"""

import ast
import dataclasses
from pathlib import Path

import pytest

from hmrec.config import (
    load_deepfm_config,
    load_deepfm_study_config,
    load_selection_rule,
    load_yaml,
    parse_deepfm_config,
    parse_deepfm_study_config,
)
from hmrec.paths import get_paths

REPO = Path(__file__).resolve().parents[2]
PACKAGE = REPO / "src" / "hmrec" / "deepfm"
SCRIPTS = (
    "scripts/m9_deepfm_benchmark.py",
    "scripts/m9_deepfm_compare.py",
    "scripts/m9_deepfm_study.py",
    "scripts/m9_deepfm_trial.py",
)
MODULES = (
    "__init__.py",
    "benchmark.py",
    "cache.py",
    "compare.py",
    "data.py",
    "inputs.py",
    "loss.py",
    "model.py",
    "process.py",
    "records.py",
    "score.py",
    "study.py",
    "study_run.py",
    "train.py",
    "trial.py",
)
# hmrec modules the package may import although they belong to another package: none
# of them imports LightGBM (checked below by walking their imports too).
ALLOWED_OUTSIDE = (
    "hmrec.ranker.window",
    "hmrec.ranker.sampling",
    "hmrec.ranker.ablation",
    "hmrec.eval.ranker_eval",
    # The result tables of the two LightGBM arms: read by the comparison script.
    "hmrec.hpo.lgb_arm_records",
)
# Modules that import LightGBM themselves or through what they import.
LOADS_LIGHTGBM = (
    "lightgbm",
    "hmrec.ranker.dataset",
    "hmrec.ranker.train",
    "hmrec.ranker.predict",
    "hmrec.ranker.locked",
    "hmrec.ranker.runtime",
    "hmrec.ranker.identity",
    "hmrec.ranker.final_window",
    "hmrec.hpo.v_records",
    "hmrec.hpo.v_trial",
    "hmrec.hpo.v_phase",
    "hmrec.hpo.v_lock",
    "hmrec.hpo.speed_test",
    "hmrec.hpo.lgb_arms",
    "hmrec.final",
    "hmrec.diagnostics",
    "hmrec.eval.final_eval",
    "hmrec.eval.diagnostic_labels",
)


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def _files() -> list[Path]:
    return [PACKAGE / name for name in MODULES] + [REPO / script for script in SCRIPTS]


def test_every_deepfm_file_is_listed_here() -> None:
    found = sorted(path.name for path in PACKAGE.glob("*.py"))
    assert found == sorted(MODULES)
    assert sorted(path.name for path in (REPO / "scripts").glob("m9_deepfm_*.py")) == sorted(
        Path(script).name for script in SCRIPTS
    )
    # The other M9 scripts are the LightGBM arms; they belong to the LightGBM side.
    assert sorted(path.name for path in (REPO / "scripts").glob("m9_lgb_*.py")) == [
        "m9_lgb_comparison.py",
        "m9_lgb_trial.py",
    ]


def test_no_deepfm_file_imports_lightgbm_or_a_module_that_loads_it() -> None:
    for path in _files():
        loading = [
            name
            for name in _imports(path)
            if any(name == bad or name.startswith(bad + ".") for bad in LOADS_LIGHTGBM)
        ]
        assert loading == [], f"{path.name} imports {loading}"


def test_the_modules_borrowed_from_m7_do_not_load_lightgbm_either() -> None:
    for module in ALLOWED_OUTSIDE:
        path = REPO / "src" / Path(*module.split(".")).with_suffix(".py")
        assert "lightgbm" not in _imports(path), module
        for name in _imports(path):
            assert not any(name == bad or name.startswith(bad + ".") for bad in LOADS_LIGHTGBM), (
                f"{module} imports {name}"
            )


def test_labels_are_read_only_through_the_m7_evaluator() -> None:
    for path in _files():
        text = path.read_text(encoding="utf-8")
        assert "label_week_transactions" not in text, path.name
        assert "FormalRunPermit" not in text and "DiagnosticAccess" not in text, path.name
        if path.name != "trial.py":
            assert "labels_file" not in text, f"{path.name} names a label file"
    trial = (PACKAGE / "trial.py").read_text(encoding="utf-8")
    # The one place that opens a label file does it through the loader that refuses
    # every fold that is not a V fold, and the fold window is checked by the M7 guard.
    assert trial.count("labels_file") == 1 and "load_m7_evaluator(" in trial
    assert "m7_training_window(" in trial


def test_results_and_files_stay_in_their_own_places() -> None:
    paths = get_paths()
    assert paths.deepfm_dir == paths.artifacts_root / "deepfm"
    assert paths.deepfm_dir in paths.deepfm_trial_dir("v_phase", "abc", 42, "V1").parents
    assert paths.deepfm_trial_dir("v_phase", "abc", 42, "V1").parts[-3:] == ("abc", "seed42", "V1")
    tables = [paths.deepfm_benchmark_file, paths.deepfm_trials_file, paths.deepfm_metrics_file]
    assert all(table.parent == paths.results_root for table in tables)
    assert all(table.name.startswith("deepfm_") for table in tables)
    cache = paths.deepfm_cache_dir("V1", "binned", "abc")
    assert cache == paths.deepfm_dir / "cache" / "V1_binned_abc"
    comparison = [paths.m9_arms_file, paths.m9_contrasts_file, paths.m9_groups_file]
    assert [table.name for table in comparison] == [
        "m9_arms.csv",
        "m9_contrasts.csv",
        "m9_groups.csv",
    ]


# ---------------------------------------------------------------- the recorded configs

CONFIG_FILE = REPO / "configs" / "deepfm.yaml"
STUDY_FILE = REPO / "configs" / "hpo" / "deepfm.yaml"


def test_default_model_is_the_recorded_one() -> None:
    config = load_deepfm_config(CONFIG_FILE)
    model, training = config.model, config.training
    assert (model.embedding_dim, model.hidden_units, model.dropout) == (16, (256, 128, 64), 0.0)
    assert (model.components, model.numeric_input) == ("full", "binned")
    assert (config.n_bins, config.window_weeks, config.negatives_per_positive) == (32, 4, 40)
    assert (training.loss, training.learning_rate, training.batch_rows) == (
        "pointwise_bce",
        0.001,
        4096,
    )
    assert (training.weight_decay, training.seed) == (0.0, 42)
    # From 2026-11-20 (proposal §16.8): a cosine schedule to 1% of the rate over the run.
    assert (training.lr_schedule, training.min_lr_ratio, training.epochs) == ("cosine", 0.01, 3)
    assert config.cached_numeric_inputs == ("binned",)
    assert config.metrics == ("map@12", "recall@12", "ndcg@12") and config.top_k == 12
    assert config.min_mps_speedup == 1.5
    speed_test = config.speed_test
    assert (speed_test.target_fold, speed_test.benchmark_steps) == ("V1", 300)
    assert (speed_test.benchmark_discard_steps, speed_test.benchmark_scoring_rows) == (
        20,
        1_000_000,
    )
    assert speed_test.epochs == 1


def test_rules_of_the_study_are_the_recorded_ones() -> None:
    study = load_deepfm_study_config(STUDY_FILE)
    assert (study.phase, study.target_folds) == ("V", ("V1", "V2", "V3", "V4"))
    assert (study.metric, study.epoch_tolerance) == ("map@12", 0.01)
    # The schedule length is a hyperparameter; the largest one is the config's default.
    assert study.schedule_epochs == (1, 2, 3)
    assert load_deepfm_config(CONFIG_FILE).training.epochs == max(study.schedule_epochs)
    assert (study.noise_seeds, study.max_seed_range, study.selection_seed) == (
        (42, 43, 44),
        0.01,
        42,
    )
    # One factor at a time; the current value first, then closer to it, smaller first.
    assert [(step.parameter, step.candidates) for step in study.steps] == [
        ("learning_rate", (0.001, 0.0003, 0.003)),
        ("embedding_dim", (16.0, 8.0, 32.0)),
        ("dropout", (0.0, 0.1, 0.3)),
    ]
    # The listwise arm tunes its own learning rate over the same values.
    assert study.listwise_learning_rates == study.steps[0].candidates
    assert study.ablation_components == ("first_order", "fm", "deep", "full")
    assert study.ablation_numeric_inputs == ("standardised",)
    assert study.final_seeds == (42, 43, 44) and not study.extend_grid_on_edge
    # The LightGBM arms: the same seeds, the locked seed first; binary to 1000 trees.
    assert (study.lightgbm_seeds, study.lightgbm_binary_num_trees) == ((42, 43, 44), 1000)
    # The first candidate of every step is the default of configs/deepfm.yaml.
    config = load_deepfm_config(CONFIG_FILE)
    defaults = {
        "learning_rate": config.training.learning_rate,
        "embedding_dim": float(config.model.embedding_dim),
        "dropout": config.model.dropout,
    }
    assert all(step.candidates[0] == defaults[step.parameter] for step in study.steps)
    # The epoch rule's tolerance is the project's selection tolerance (1%).
    rule = load_selection_rule(REPO / "configs" / "selection_rule.yaml")
    assert study.epoch_tolerance == rule.min_relative_improvement == 0.01
    assert rule.min_folds_better == 3


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("model", "components", "wide", "components"),
        ("model", "numeric_input", "raw", "numeric_input"),
        ("model", "dropout", 1.0, "dropout"),
        ("model", "hidden_units", [], "hidden_units"),
        ("training", "loss", "pairwise", "loss"),
        ("training", "device", "cuda", "device"),
        ("training", "batch_rows", 0, "batch_rows"),
        ("training", "lr_schedule", "linear", "lr_schedule"),
        ("training", "min_lr_ratio", 0.0, "min_lr_ratio"),
        ("training", "min_lr_ratio", 1.5, "min_lr_ratio"),
        ("input_cache", "numeric_inputs", ["raw"], "numeric_inputs"),
        ("binning", "n_bins", 300, "n_bins"),
        ("standardise", "clip_low_percentile", 99.95, "low < high"),
        ("device_rule", "min_mps_speedup", 0.5, "min_mps_speedup"),
        ("speed_test", "benchmark_discard_steps", 300, "benchmark_discard_steps"),
    ],
)
def test_config_values_are_validated(section: str, key: str, value: object, message: str) -> None:
    raw = load_yaml(CONFIG_FILE)
    raw[section] = {**raw[section], key: value}
    with pytest.raises(ValueError, match=message):
        parse_deepfm_config(raw)


def test_study_config_is_validated() -> None:
    raw = load_yaml(STUDY_FILE)
    with pytest.raises(ValueError, match="phase must be 'V'"):
        parse_deepfm_study_config({**raw, "phase": "T"})
    with pytest.raises(ValueError, match="unknown keys"):
        parse_deepfm_study_config({**raw, "test_folds": ["T1"]})
    with pytest.raises(ValueError, match="selection_seed must be one of the noise seeds"):
        parse_deepfm_study_config({**raw, "selection_seed": 7})
    bad_step = [{"name": "x", "parameter": "batch_rows", "candidates": [1, 2]}]
    with pytest.raises(ValueError, match="parameter"):
        parse_deepfm_study_config({**raw, "steps": bad_step})
    repeated = [{"name": "x", "parameter": "dropout", "candidates": [0.1, 0.1]}]
    with pytest.raises(ValueError, match="at least 2 distinct numbers"):
        parse_deepfm_study_config({**raw, "steps": repeated})
    with pytest.raises(ValueError, match="a grid is never extended"):
        parse_deepfm_study_config({**raw, "extend_grid_on_edge": True})
    # The rules withdrawn on 2026-11-20 are no longer accepted as settings.
    with pytest.raises(ValueError, match="unknown keys"):
        parse_deepfm_study_config({**raw, "final_epoch_extension": {"epochs": 10}})
    with pytest.raises(ValueError, match="distinct and ascending"):
        parse_deepfm_study_config({**raw, "schedule_epochs": [3, 1, 2]})
    # The schedule length is tuned with the learning rate, so that step must come first.
    with pytest.raises(ValueError, match="the first step must tune learning_rate"):
        parse_deepfm_study_config({**raw, "steps": list(reversed(raw["steps"]))})
    with pytest.raises(ValueError, match="at least 2 distinct numbers"):
        parse_deepfm_study_config({**raw, "listwise": {"learning_rate": [0.001]}})
    lightgbm = {**raw["lightgbm"], "seeds": [43, 42, 44]}
    with pytest.raises(ValueError, match="must start with the selection seed"):
        parse_deepfm_study_config({**raw, "lightgbm": lightgbm})
    with pytest.raises(ValueError, match="selection_seed must be one of the final seeds"):
        parse_deepfm_study_config({**raw, "final_seeds": [43, 44]})
    study = load_deepfm_study_config(STUDY_FILE)
    assert dataclasses.is_dataclass(study)


def test_the_study_runs_on_cpu_and_the_torch_free_modules_stay_torch_free() -> None:
    # A decision by the user (docs/decisions.md 2026-11-19): CPU, so that a seed repeats exactly.
    assert load_deepfm_config(CONFIG_FILE).training.device == "cpu"
    # The rules, the driver and the comparison need no model: they must not load torch.
    torch_free = ("study.py", "study_run.py", "compare.py", "records.py", "data.py", "inputs.py")
    for name in (*torch_free, "cache.py"):
        imports = _imports(PACKAGE / name)
        assert "torch" not in imports, name
        torch_modules = ("model", "train", "score", "trial", "benchmark", "process", "loss")
        assert not [m for m in imports if m in {f"hmrec.deepfm.{t}" for t in torch_modules}], name
