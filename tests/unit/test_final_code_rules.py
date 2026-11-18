"""Rules the code of the final evaluation (M8) must keep, checked by reading the source.

- The Ranker configuration comes only from the lock (hmrec.ranker.locked).
- A FormalRunPermit is created in one place; T labels are read in one place.
- The M8 code names no T fold: the folds come from configs/m8_final.yaml.
- The code of M6 and M7 does not use the M8 entry (no T fold in the tuning code).
- An M8 process loads no torch (LightGBM and torch cannot share a process).
"""

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src" / "hmrec"
# The code of the final evaluation. ADD every new M8 module here when it is written.
M8_FILES = (
    *sorted((REPO / "scripts").glob("m8_*.py")),
    *sorted((SRC / "final").glob("*.py")),
    SRC / "eval" / "final_eval.py",
    SRC / "ranker" / "final_window.py",
)
# Ways to get a Ranker configuration that are closed after the lock: configs/ranker.yaml
# (tuned defaults and fixed parameters), the V-phase grid, the V-phase steps table and the
# modules that tune, replay a selection or run the speed test.
PRE_LOCK_RANKER_NAMES = (
    "load_ranker_config",
    "ranker_config_file",
    "tuned_defaults",
    "load_v_phase_grid",
    "v_phase_grid_file",
    "v_phase_steps_file",
    "v_phase_results_file",
    "v_phase_metrics_file",
    "hmrec.hpo.v_phase",
    "hmrec.hpo.v_lock",
    "hmrec.hpo.speed_test",
    "hmrec.hpo.runner",
    "hmrec.hpo.d_phase",
    "hmrec.recall",
)


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _relative(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def _imports(path: Path, module: str) -> bool:
    """True if the file has an import statement of `module` or of something inside it."""
    statements = (f"from {module}", f"import {module}")
    return any(line.strip().startswith(statements) for line in _text(path).splitlines())


def _uses(path: Path, name: str) -> bool:
    """A module name counts when it is imported; any other name when it appears at all."""
    return _imports(path, name) if name.startswith("hmrec.") else name in _text(path)


def _python_files(*folders: str) -> list[Path]:
    files: list[Path] = []
    for folder in folders:
        files.extend(sorted((REPO / folder).rglob("*.py")))
    return files


def test_the_expected_m8_files_exist() -> None:
    assert [_relative(path) for path in M8_FILES] == [
        "scripts/m8_final_eval.py",
        "scripts/m8_step.py",
        "src/hmrec/final/__init__.py",
        "src/hmrec/final/context.py",
        "src/hmrec/final/gate.py",
        "src/hmrec/final/goals.py",
        "src/hmrec/final/monitor.py",
        "src/hmrec/final/process.py",
        "src/hmrec/final/rehearsal.py",
        "src/hmrec/final/report.py",
        "src/hmrec/final/run.py",
        "src/hmrec/final/trial.py",
        "src/hmrec/eval/final_eval.py",
        "src/hmrec/ranker/final_window.py",
    ]


def test_m8_code_gets_the_ranker_configuration_only_from_the_lock() -> None:
    offenders = [
        f"{_relative(path)}: {name}"
        for path in M8_FILES
        for name in PRE_LOCK_RANKER_NAMES
        if _uses(path, name)
    ]
    assert offenders == []
    # The one entry is used, in one place.
    users = [_relative(path) for path in M8_FILES if "load_locked_ranker(" in _text(path)]
    assert users == ["src/hmrec/final/context.py"]


def test_only_the_rehearsal_comparison_reads_the_v_phase_tables() -> None:
    # results/v_phase*.csv are read through results_files(paths, AREA_V_PHASE). In M8 only
    # the rehearsal comparison does that: it compares recorded numbers of the locked
    # configuration and takes no configuration from them.
    users = [_relative(path) for path in M8_FILES if "AREA_V_PHASE" in _text(path)]
    assert users == ["src/hmrec/final/rehearsal.py"]


def test_a_permit_is_created_only_by_the_gate() -> None:
    creators = [
        _relative(path)
        for path in _python_files("src", "scripts")
        if "FormalRunPermit(" in _text(path)
    ]
    assert creators == ["src/hmrec/final/gate.py"]
    # And inside the gate in one function, after the rehearsal and the start were checked.
    assert _text(SRC / "final" / "gate.py").count("FormalRunPermit(") == 1


def test_t_labels_are_read_in_one_place_of_the_m8_code() -> None:
    files = _python_files("src", "scripts")
    evaluators = [_relative(p) for p in files if "load_final_evaluator(" in _text(p)]
    assert evaluators == [
        "src/hmrec/eval/final_eval.py",  # the definition
        "src/hmrec/final/trial.py",  # target_evaluator
        "scripts/checks/check_ranker_eval.py",  # the independent check, through the gate
    ]
    phases = [_relative(p) for p in files if "final_training_label_phases(" in _text(p)]
    assert phases == ["src/hmrec/eval/final_eval.py", "src/hmrec/final/trial.py"]
    # No M8 module opens a label file itself or reads the label week.
    for path in M8_FILES:
        text = _text(path)
        if path.name != "final_eval.py":
            assert "FoldEvaluator.from_file" not in text, path.name
        assert "label_week_transactions" not in text, path.name
        assert "load_training_labels" not in text, path.name
    naming = [path.name for path in M8_FILES if "labels_file(" in _text(path)]
    assert naming == ["trial.py"]  # sample_job and target_evaluator hand it to guarded loaders


def test_m8_code_names_no_t_fold() -> None:
    # The target folds come from configs/m8_final.yaml; no fold name is written in the code.
    t_fold = re.compile(r"""["']T[1-4]["']""")
    assert [_relative(path) for path in M8_FILES if t_fold.search(_text(path))] == []


def test_the_tuning_code_does_not_use_the_m8_entry() -> None:
    # AGENTS.md §3.2: no T fold in the HPO code path. The modules of M6 and M7 neither
    # import the final evaluation nor its label entry.
    names = ("hmrec.final", "hmrec.eval.final_eval", "hmrec.ranker.final_window", "FormalRunPermit")
    tuning = [
        *sorted((SRC / "hpo").glob("*.py")),
        *(p for p in sorted((SRC / "ranker").glob("*.py")) if p.name != "final_window.py"),
        *sorted((SRC / "recall").rglob("*.py")),
        *sorted((SRC / "features").glob("*.py")),
        *sorted((SRC / "fusion").glob("*.py")),
        SRC / "eval" / "ranker_eval.py",
        SRC / "eval" / "monitor.py",
        SRC / "pipeline.py",
        *sorted((REPO / "scripts").glob("m[0-7]_*.py")),
    ]
    assert len(tuning) > 50  # the lists above really found the files
    offenders = [
        f"{_relative(path)}: {name}" for path in tuning for name in names if _uses(path, name)
    ]
    assert offenders == []


def test_m8_processes_never_import_torch() -> None:
    modules = [
        "hmrec.final.process",
        "hmrec.final.run",
        "hmrec.final.rehearsal",
        "hmrec.final.report",
        "hmrec.final.monitor",
        "hmrec.final.trial",
        "hmrec.final.gate",
        "hmrec.pipeline",
    ]
    code = (
        f"import sys, {', '.join(modules)}; "
        "loaded = [name for name in ('torch', 'implicit') if name in sys.modules]; "
        "loaded += [name for name in sys.modules if name.startswith('hmrec.recall')]; "
        "print(loaded)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=REPO
    )
    assert result.stdout.strip() == "[]"
    # Every hmrec import of an M8 script is one of those modules (or below them).
    allowed = ("hmrec.final.", "hmrec.pipeline", "hmrec.threads", "hmrec.logging_utils",
               "hmrec.ranker.ablation")  # fmt: skip
    for path in sorted((REPO / "scripts").glob("m8_*.py")):
        for line in _text(path).splitlines():
            if line.startswith("from hmrec"):
                assert line.split()[1].startswith(allowed), f"{path.name}: {line}"
