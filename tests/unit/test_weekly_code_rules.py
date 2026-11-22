"""Static rules of the weekly pipeline's code (stage 3, M12; AGENTS.md §12).

The source files are read as text; nothing is imported, so the test itself
loads neither torch nor LightGBM.

    1. The two sides stay apart: the torch-side modules never import a LightGBM
       module, the LightGBM-side module never imports torch or the recall code,
       and every other module imports neither (run_step imports one side inside
       the function that runs it).
    2. No module of the pipeline reaches a label entry point of M6, M7, M8, the
       diagnostics or the M9 evaluation, and none imports a module that tunes or
       selects. Labels come only through hmrec.eval.pipeline_labels.
    3. Only two places read the SOURCE transactions (data/processed/): C1 ingest
       and the `eval` query list, both in hmrec.weekly.data_steps. Every other
       module reads the tick's ingested table.
    4. No other code imports hmrec.weekly, except its two scripts.
    5. The pipeline writes no stage-1 result table.
"""

import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src" / "hmrec"
WEEKLY = SRC / "weekly"
SCRIPTS = (REPO / "scripts" / "weekly.py", REPO / "scripts" / "m12_step.py")

PLAIN_MODULES = (
    "calendar.py",
    "context.py",
    "data_steps.py",
    "driver.py",
    "feature_steps.py",
    "history.py",
    "monitoring.py",
    "outputs.py",
    "run_step.py",
    "shard_features.py",
    "shards.py",
    "state.py",
    "steps.py",
    "storage.py",
)
TORCH_SIDE_MODULES = ("recall_steps.py", "shard_recall.py")
LIGHTGBM_SIDE_MODULES = ("ranker_steps.py",)

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
    "hmrec.weekly.ranker_steps",
)
TORCH_MODULES = (
    "torch",
    "hmrec.recall",
    "hmrec.hpo.runner",
    "hmrec.hpo.d_phase",
    "hmrec.deepfm",
    "hmrec.weekly.recall_steps",
    "hmrec.weekly.shard_recall",
)
FORBIDDEN_EVERYWHERE = (
    "hmrec.eval.final_eval",
    "hmrec.eval.diagnostic_labels",
    "hmrec.eval.t_eval_labels",
    "hmrec.final",
    "hmrec.diagnostics",
    "hmrec.t_eval",
    "hmrec.calibration",
    "hmrec.hpo.v_phase",
    "hmrec.hpo.v_lock",
    "hmrec.hpo.d_phase",
    "hmrec.hpo.runner",
    "hmrec.hpo.selection",
    "hmrec.tracking",
)
SOURCE_READER = "scan_source_transactions"
STAGE1_TABLE = re.compile(r"paths\.(m6|m7|m8|v_phase|t_phase|d_phase)\w*_file")


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
    # A new module of the pipeline must be added to one of the three lists above.
    files = sorted(path.name for path in WEEKLY.glob("*.py") if path.name != "__init__.py")
    assert files == sorted(PLAIN_MODULES + TORCH_SIDE_MODULES + LIGHTGBM_SIDE_MODULES)


def test_the_two_sides_stay_apart() -> None:
    for name in TORCH_SIDE_MODULES:
        imports = imported_modules(WEEKLY / name)
        for module in LIGHTGBM_MODULES:
            assert not uses(imports, module), f"{name} imports {module}"
    for name in LIGHTGBM_SIDE_MODULES:
        imports = imported_modules(WEEKLY / name)
        for module in TORCH_MODULES:
            assert not uses(imports, module), f"{name} imports {module}"
    for name in PLAIN_MODULES:
        # run_step.py imports one side's module inside the function that runs it, on
        # purpose; at the top of a plain module neither side may appear.
        everywhere = name != "run_step.py"
        imports = imported_modules(WEEKLY / name, top_level_only=not everywhere)
        for module in (*LIGHTGBM_MODULES, *TORCH_MODULES):
            assert not uses(imports, module), f"{name} imports {module}"
    for script in SCRIPTS:
        imports = imported_modules(script)
        for module in (*LIGHTGBM_MODULES, *TORCH_MODULES):
            assert not uses(imports, module), f"{script.name} imports {module}"


def test_no_label_entry_point_of_another_milestone_and_nothing_that_selects() -> None:
    for path in [*sorted(WEEKLY.glob("*.py")), *SCRIPTS, SRC / "eval" / "pipeline_labels.py"]:
        imports = imported_modules(path)
        for module in FORBIDDEN_EVERYWHERE:
            assert not uses(imports, module), f"{path.name} imports {module}"
    # The guarded loaders of M6 and M7 refuse T folds; the pipeline has its own entry.
    for path in sorted(WEEKLY.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for loader in ("load_m7_evaluator", "load_label_evaluator", "FoldEvaluator.from_file"):
            assert loader not in text, f"{path.name} uses {loader}"


def test_only_ingest_and_the_eval_query_list_read_the_source_transactions() -> None:
    for path in sorted(WEEKLY.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        if path.name == "data_steps.py":
            # Exactly two callers: run_ingest (C1) and eval_queries (the `eval` query list).
            assert text.count(f"{SOURCE_READER}(context)") == 2
            continue
        assert SOURCE_READER not in text, f"{path.name} reads the source transactions"
        # The source table's path is context.paths.transactions_file; every other module
        # goes through context.view (the ingested table).
        assert "context.paths.transactions_file" not in text, path.name
        rest = text.replace("view.transactions_file", "")
        if path.name == "driver.py":
            # The driver records the source file's size and sha256 as an input of the
            # tick (tick_inputs); it never opens its rows.
            rest = rest.replace('"processed/transactions": paths.transactions_file', "")
        assert "paths.transactions_file" not in rest, path.name


def test_no_other_code_imports_the_weekly_pipeline() -> None:
    files = [*SRC.rglob("*.py"), *(REPO / "scripts").glob("*.py")]
    for path in files:
        if WEEKLY in path.parents or path in SCRIPTS:
            continue
        assert not uses(imported_modules(path), "hmrec.weekly"), path


def test_the_pipeline_writes_no_stage_1_result_table() -> None:
    for path in sorted(WEEKLY.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        used = set(STAGE1_TABLE.findall(text))
        # m6_config_file is the stop limit of the R6 fine-tune (configs/, read only).
        assert STAGE1_TABLE.search(text.replace("paths.m6_config_file", "")) is None, (path, used)
