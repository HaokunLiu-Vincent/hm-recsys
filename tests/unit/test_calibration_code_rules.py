"""Static rules of the calibration code (stage 3, M10; AGENTS.md §12).

The source files are read as text; nothing is imported, so the test itself
loads neither torch nor LightGBM.

    1. Only the calibration code uses the tolerance-accepted entry
       (hmrec.ranker.platform_entry), and no other code imports hmrec.calibration.
    2. The two sides stay apart: the recall items never import a LightGBM
       module, the Ranker items never import torch or the recall code, and the
       shared modules import neither.
    3. No calibration module reaches a label entry point of the T weeks, and
       none imports a module that tunes or selects.
    4. The strict entry does not know the tolerance-accepted one.
"""

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src" / "hmrec"
CALIBRATION = SRC / "calibration"
CALIBRATION_SCRIPT = REPO / "scripts" / "m10_calibrate.py"

SHARED_MODULES = ("records.py", "compare.py", "references.py", "context.py", "run.py", "repeat.py")
TORCH_SIDE_MODULES = ("recall_items.py",)
LIGHTGBM_SIDE_MODULES = ("ranker_items.py",)

# Modules that load LightGBM when imported (AGENTS.md §5).
LIGHTGBM_MODULES = (
    "lightgbm",
    "hmrec.ranker.dataset",
    "hmrec.ranker.train",
    "hmrec.ranker.predict",
    "hmrec.ranker.runtime",
    "hmrec.ranker.locked",
    "hmrec.ranker.platform_entry",
    "hmrec.hpo.v_trial",
    "hmrec.hpo.v_records",
    "hmrec.final",
)
# Modules that load torch when imported.
TORCH_MODULES = ("torch", "hmrec.recall", "hmrec.hpo.runner", "hmrec.hpo.d_phase", "hmrec.deepfm")
# Label entry points of the T weeks, and modules that tune or select.
FORBIDDEN_EVERYWHERE = (
    "hmrec.eval.final_eval",
    "hmrec.eval.diagnostic_labels",
    "hmrec.eval.t_eval_labels",
    "hmrec.final",
    "hmrec.diagnostics",
    "hmrec.t_eval",
    "hmrec.hpo.v_phase",
    "hmrec.hpo.v_lock",
    "hmrec.hpo.d_phase",
    "hmrec.hpo.runner",
    "hmrec.hpo.selection",
)


def imported_modules(path: Path, top_level_only: bool = False) -> set[str]:
    """Full names of the modules a Python file imports ("from a.b import c" gives a.b and a.b.c)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = tree.body if top_level_only else list(ast.walk(tree))
    names: set[str] = set()
    for node in nodes:
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def uses(imports: set[str], module: str) -> bool:
    """True if `module` or one of its submodules is among the imports."""
    return any(name == module or name.startswith(module + ".") for name in imports)


def test_the_module_lists_are_complete() -> None:
    # A new calibration module must be added to one of the three lists above.
    files = sorted(path.name for path in CALIBRATION.glob("*.py") if path.name != "__init__.py")
    assert files == sorted(SHARED_MODULES + TORCH_SIDE_MODULES + LIGHTGBM_SIDE_MODULES)


def test_only_the_calibration_code_uses_the_tolerance_accepted_entry() -> None:
    allowed = {CALIBRATION / name for name in LIGHTGBM_SIDE_MODULES}
    files = [*SRC.rglob("*.py"), *(REPO / "scripts").glob("*.py")]
    for path in files:
        if path in allowed:
            continue
        assert not uses(imported_modules(path), "hmrec.ranker.platform_entry"), path
    for path in allowed:
        assert uses(imported_modules(path), "hmrec.ranker.platform_entry"), path


def test_no_other_code_imports_the_calibration() -> None:
    files = [*SRC.rglob("*.py"), *(REPO / "scripts").glob("*.py")]
    for path in files:
        if CALIBRATION in path.parents or path == CALIBRATION_SCRIPT:
            continue
        assert not uses(imported_modules(path), "hmrec.calibration"), path


def test_the_two_sides_stay_apart() -> None:
    for name in TORCH_SIDE_MODULES:
        imports = imported_modules(CALIBRATION / name)
        for module in LIGHTGBM_MODULES:
            assert not uses(imports, module), f"{name} imports {module}"
    for name in LIGHTGBM_SIDE_MODULES:
        imports = imported_modules(CALIBRATION / name)
        for module in TORCH_MODULES:
            assert not uses(imports, module), f"{name} imports {module}"
    for name in SHARED_MODULES:
        # At the top of a shared module neither side may be imported. Inside a function
        # run.py imports one side's module and repeat.py imports torch, on purpose.
        imports = imported_modules(CALIBRATION / name, top_level_only=True)
        for module in (
            *LIGHTGBM_MODULES,
            *TORCH_MODULES,
            "hmrec.calibration.recall_items",
            "hmrec.calibration.ranker_items",
        ):
            assert not uses(imports, module), f"{name} imports {module} at the top"


def test_no_calibration_code_reaches_t_labels_or_a_module_that_selects() -> None:
    for path in [*CALIBRATION.glob("*.py"), CALIBRATION_SCRIPT]:
        imports = imported_modules(path)
        for module in FORBIDDEN_EVERYWHERE:
            assert not uses(imports, module), f"{path.name} imports {module}"


def test_labels_are_reached_only_through_the_guarded_loaders_of_m6_and_m7() -> None:
    # The two loaders check the fold's phase before a label file is opened: a T fold raises.
    for path in CALIBRATION.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "FoldEvaluator.from_file" not in text, path.name
        assert "label_week_transactions" not in text, path.name
    recall_text = (CALIBRATION / "recall_items.py").read_text(encoding="utf-8")
    ranker_text = (CALIBRATION / "ranker_items.py").read_text(encoding="utf-8")
    assert "load_label_evaluator(" in recall_text
    assert "load_m7_evaluator(" in ranker_text and "m7_training_window(" in ranker_text


def test_the_strict_entry_does_not_know_the_tolerance_accepted_one() -> None:
    strict = imported_modules(SRC / "ranker" / "locked.py")
    assert not uses(strict, "hmrec.ranker.platform_entry")
    assert not uses(strict, "hmrec.calibration")
    assert not uses(strict, "hmrec.platform_fields")
