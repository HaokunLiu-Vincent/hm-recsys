"""Rules the diagnostics after M8 must keep, checked by reading the source (AGENTS.md §3.2).

- T labels are read again through ONE second door, and only the diagnostics use it.
- A DiagnosticAccess is created in one place, after the read was recorded.
- The diagnostics train nothing, sample nothing and do not use the formal run's code.
- They write only results/t_diag_*.csv.
- No code of M0-M8 uses the diagnostics.
"""

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src" / "hmrec"
DIAG_FILES = (
    REPO / "scripts" / "m8_diagnostics.py",
    *sorted((SRC / "diagnostics").glob("*.py")),
    SRC / "eval" / "diagnostic_labels.py",
)
# What a diagnostics file must not import: training, sampling, scoring with a model, the
# formal run's trial / gate / label door, tuning and selection, recall, torch.
FORBIDDEN_IMPORTS = (
    "hmrec.ranker.train",
    "hmrec.ranker.sampling",
    "hmrec.ranker.dataset",
    "hmrec.ranker.predict",
    "hmrec.ranker.window",
    "hmrec.ranker.final_window",
    "hmrec.hpo.v_trial",
    "hmrec.hpo.v_phase",
    "hmrec.hpo.v_lock",
    "hmrec.hpo.speed_test",
    "hmrec.hpo.runner",
    "hmrec.hpo.d_phase",
    "hmrec.final.trial",
    "hmrec.final.run",
    "hmrec.final.process",
    "hmrec.final.gate",
    "hmrec.final.rehearsal",
    "hmrec.final.monitor",
    "hmrec.final.report",
    "hmrec.final.context",
    "hmrec.eval.final_eval",
    "hmrec.eval.ranker_eval",
    "hmrec.recall",
    "hmrec.features",
    "lightgbm",
    "torch",
)


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _relative(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def _imports(path: Path, module: str) -> bool:
    """True if the file has an import statement of `module` or of something inside it."""
    statements = (f"from {module}", f"import {module}")
    return any(line.strip().startswith(statements) for line in _text(path).splitlines())


def _python_files(*folders: str) -> list[Path]:
    files: list[Path] = []
    for folder in folders:
        files.extend(sorted((REPO / folder).rglob("*.py")))
    return files


def test_the_expected_diagnostics_files_exist() -> None:
    assert [_relative(path) for path in DIAG_FILES] == [
        "scripts/m8_diagnostics.py",
        "src/hmrec/diagnostics/__init__.py",
        "src/hmrec/diagnostics/access.py",
        "src/hmrec/diagnostics/bootstrap.py",
        "src/hmrec/diagnostics/concentration.py",
        "src/hmrec/diagnostics/context.py",
        "src/hmrec/diagnostics/decomposition.py",
        "src/hmrec/diagnostics/inputs.py",
        "src/hmrec/diagnostics/query_groups.py",
        "src/hmrec/diagnostics/recorded.py",
        "src/hmrec/diagnostics/records.py",
        "src/hmrec/diagnostics/run.py",
        "src/hmrec/eval/diagnostic_labels.py",
    ]


def test_an_access_is_created_only_by_the_gate_after_the_record() -> None:
    creators = [
        _relative(path)
        for path in _python_files("src", "scripts")
        if "DiagnosticAccess(" in _text(path)
    ]
    assert creators == ["src/hmrec/diagnostics/access.py"]
    gate = _text(SRC / "diagnostics" / "access.py")
    assert gate.count("DiagnosticAccess(") == 1
    # The row is appended BEFORE the access object is built.
    assert gate.index("append_row(context.paths.t_diag_label_access_file") < gate.index(
        "DiagnosticAccess("
    )


def test_only_the_diagnostics_use_the_second_label_door() -> None:
    files = _python_files("src", "scripts")
    users = [_relative(p) for p in files if _imports(p, "hmrec.eval.diagnostic_labels")]
    assert users == [
        "src/hmrec/diagnostics/access.py",  # the DiagnosticAccess class
        "src/hmrec/diagnostics/concentration.py",  # a column name
        "src/hmrec/diagnostics/run.py",  # the loader and the label counts
    ]
    callers = [_relative(p) for p in files if "load_diagnostic_evaluator(" in _text(p)]
    assert callers == ["src/hmrec/diagnostics/run.py", "src/hmrec/eval/diagnostic_labels.py"]
    # No diagnostics file opens a label file itself, reads the label week or uses the
    # M8 permit: labels come from the one door in hmrec.eval.
    for path in DIAG_FILES:
        text = _text(path)
        if path.name != "diagnostic_labels.py":
            assert "FoldEvaluator.from_file" not in text, path.name
        assert "label_week_transactions" not in text, path.name
        assert "FormalRunPermit" not in text, path.name
        assert "load_final_evaluator" not in text, path.name
    naming = [path.name for path in DIAG_FILES if "labels_file(" in _text(path)]
    assert naming == ["run.py"]  # handed to load_diagnostic_evaluator, twice
    # The M8 label door is untouched: it does not know the diagnostics.
    assert "diagnostic" not in _text(SRC / "eval" / "final_eval.py").lower()


def test_the_diagnostics_import_no_training_sampling_or_m8_run_code() -> None:
    offenders = [
        f"{_relative(path)}: {module}"
        for path in DIAG_FILES
        for module in FORBIDDEN_IMPORTS
        if _imports(path, module)
    ]
    assert offenders == []
    # From the final evaluation they use one thing: the column names of the verdict table.
    final_users = [_relative(path) for path in DIAG_FILES if _imports(path, "hmrec.final")]
    assert final_users == ["src/hmrec/diagnostics/access.py"]
    assert "from hmrec.final.goals import GOAL_COLUMNS" in _text(SRC / "diagnostics" / "access.py")


def test_the_diagnostics_write_only_their_own_tables() -> None:
    for path in DIAG_FILES:
        text = _text(path)
        for forbidden in ("write_parquet", "write_csv", ".write_text(", ".unlink(", "shutil"):
            assert forbidden not in text, f"{path.name}: {forbidden}"
    # Rows are appended in two places only, and only to results/t_diag_*.csv.
    writers = [path.name for path in DIAG_FILES if "append_row(" in _text(path)]
    assert writers == ["access.py", "records.py"]
    assert "append_row(context.paths.t_diag_label_access_file" in _text(
        SRC / "diagnostics" / "access.py"
    )
    run = _text(SRC / "diagnostics" / "run.py")
    written = [
        line.strip() for line in run.splitlines() if "paths.t_" in line or "paths.m8_" in line
    ]
    assert len(written) == 4 and all("paths.t_diag_" in line for line in written), written


def test_no_other_code_uses_the_diagnostics() -> None:
    names = ("hmrec.diagnostics", "hmrec.eval.diagnostic_labels")
    others = [path for path in _python_files("src", "scripts") if path not in DIAG_FILES]
    assert len(others) > 100  # the list really holds the project's code
    offenders = [
        f"{_relative(path)}: {name}" for path in others for name in names if _imports(path, name)
    ]
    assert offenders == []
    for path in others:
        assert "DiagnosticAccess" not in _text(path), _relative(path)


def test_a_diagnostics_process_never_imports_torch() -> None:
    code = (
        "import sys, hmrec.diagnostics.run, hmrec.diagnostics.context; "
        "loaded = [name for name in ('torch', 'implicit') if name in sys.modules]; "
        "loaded += [name for name in sys.modules if name.startswith('hmrec.recall')]; "
        "print(loaded)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=REPO
    )
    assert result.stdout.strip() == "[]"
