"""Rules the Kaggle late submission must keep, checked by reading the source.

- torch and LightGBM never share a process: the ranker side (kaggle/ranker.py,
  scripts/kaggle_ranker.py) imports no torch and nothing of recall/; the other
  modules import no LightGBM and nothing that loads it.
- Labels: only kaggle/ranker.py names a label file, and nothing of the Kaggle
  code reads a label week from the transactions or uses the two guarded doors
  of M8 (the formal run's and the diagnostics').
- Nothing is tuned or selected: no Kaggle file imports a module of hmrec.hpo
  that selects, and none reads the tuning configs.
- Every module that this test knows about exists, and every Kaggle file is known.
"""

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PACKAGE = REPO / "src" / "hmrec" / "kaggle"
SCRIPTS = REPO / "scripts"

LIGHTGBM_SIDE = ("src/hmrec/kaggle/ranker.py", "scripts/kaggle_ranker.py")
OTHER_FILES = (
    "src/hmrec/kaggle/__init__.py",
    "src/hmrec/kaggle/context.py",
    "src/hmrec/kaggle/features.py",
    "src/hmrec/kaggle/plan.py",
    "src/hmrec/kaggle/prepare.py",
    "src/hmrec/kaggle/recall.py",
    "src/hmrec/kaggle/rehearsal.py",
    "src/hmrec/kaggle/report.py",
    "src/hmrec/kaggle/run.py",
    "src/hmrec/kaggle/submission.py",
    "scripts/kaggle_features.py",
    "scripts/kaggle_prepare.py",
    "scripts/kaggle_record_submission.py",
    "scripts/kaggle_recall.py",
    "scripts/kaggle_run.py",
    "scripts/kaggle_submission.py",
)
# Modules that import LightGBM (directly or through what they import).
LOADS_LIGHTGBM = (
    "lightgbm",
    "hmrec.ranker.",
    "hmrec.hpo.",
    "hmrec.final",
    "hmrec.eval.ranker_eval",
)
# hmrec.ranker.window is a plain module (folds only); the plan needs its window function.
ALLOWED_FROM_RANKER = ("hmrec.ranker.window",)
LOADS_TORCH = ("torch", "hmrec.recall", "hmrec.hpo.runner", "hmrec.hpo.d_phase")
SELECTING_MODULES = (
    "hmrec.hpo.v_phase",
    "hmrec.hpo.v_lock",
    "hmrec.hpo.speed_test",
    "hmrec.hpo.runner",
    "hmrec.hpo.d_phase",
    "hmrec.hpo.selection",
    "hmrec.hpo.grid",
)


def _imports(relative: str) -> list[str]:
    """Every module a file imports (absolute names)."""
    tree = ast.parse((REPO / relative).read_text(encoding="utf-8"))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def _starts_with(name: str, prefixes: tuple[str, ...]) -> bool:
    return any(name == prefix.rstrip(".") or name.startswith(prefix) for prefix in prefixes)


def test_every_kaggle_file_is_listed_here() -> None:
    found = {path.relative_to(REPO).as_posix() for path in PACKAGE.glob("*.py")}
    found |= {path.relative_to(REPO).as_posix() for path in SCRIPTS.glob("kaggle_*.py")}
    assert found == set(LIGHTGBM_SIDE) | set(OTHER_FILES)


def test_the_ranker_side_never_loads_torch() -> None:
    for relative in LIGHTGBM_SIDE:
        loading = [name for name in _imports(relative) if _starts_with(name, LOADS_TORCH)]
        assert loading == [], f"{relative} imports {loading}"
        # What it imports from the Kaggle package must be torch-free too.
        assert "hmrec.kaggle.recall" not in _imports(relative)
        assert "hmrec.kaggle.features" not in _imports(relative)


def test_the_other_files_never_load_lightgbm() -> None:
    for relative in OTHER_FILES:
        loading = [
            name
            for name in _imports(relative)
            if _starts_with(name, LOADS_LIGHTGBM) and name not in ALLOWED_FROM_RANKER
        ]
        assert loading == [], f"{relative} imports {loading}"
        assert "hmrec.kaggle.ranker" not in _imports(relative)


def test_only_the_ranker_module_names_a_label_file() -> None:
    for relative in (*LIGHTGBM_SIDE, *OTHER_FILES):
        text = (REPO / relative).read_text(encoding="utf-8")
        assert "label_week_transactions" not in text, relative
        if relative != "src/hmrec/kaggle/ranker.py":
            assert "labels_file" not in text, f"{relative} names a label file"
        for door in ("hmrec.eval.final_eval", "hmrec.eval.diagnostic_labels", "hmrec.diagnostics"):
            assert door not in _imports(relative), f"{relative} uses the guarded door {door}"
        assert "hmrec.final.gate" not in _imports(relative)


def test_nothing_is_tuned_or_selected() -> None:
    for relative in (*LIGHTGBM_SIDE, *OTHER_FILES):
        selecting = [name for name in _imports(relative) if name in SELECTING_MODULES]
        assert selecting == [], f"{relative} imports {selecting}"
        text = (REPO / relative).read_text(encoding="utf-8")
        # The tuning grids, the D-phase defaults and the result tables a selection reads.
        for forbidden in (
            "v_phase_grid_file",
            "d_phase_grid_file",
            "recall_config_file",
            "d_phase_results_file",
            "v_phase_results_file",
        ):
            assert forbidden not in text, f"{relative} reads {forbidden}"


def test_the_ranker_configuration_comes_from_the_locked_entry_only() -> None:
    text = (PACKAGE / "ranker.py").read_text(encoding="utf-8")
    assert "load_locked_ranker" in text
    for forbidden in ("load_ranker_config", "ranker_config_file"):
        assert forbidden not in text
    recall = (PACKAGE / "recall.py").read_text(encoding="utf-8")
    assert "load_locked_recall" in recall and "build_locked_channel" in recall
    features = (PACKAGE / "features.py").read_text(encoding="utf-8")
    assert "check_features_match_lock" in features
