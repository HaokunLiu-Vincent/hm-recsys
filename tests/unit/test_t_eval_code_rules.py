"""Rules the M9 evaluation on T1-T4 must keep, checked by reading the source (§16.12).

- T labels are read through ONE third door, and only this evaluation's modules use it.
- A TEvalPermit is created in one place, after the read was recorded.
- The evaluation's code uses neither the M8 door nor the diagnostics' door, and no
  other code uses the evaluation.
- No LightGBM is loaded: the M8 record is only read.
- The recorded configuration is the one the config files give.
"""

import subprocess
import sys
from pathlib import Path

from hmrec.config import load_t_eval_config
from hmrec.deepfm.process import load_trial_inputs
from hmrec.deepfm.records import deepfm_config_id
from hmrec.paths import get_paths
from hmrec.t_eval.trial import chosen_trial_inputs

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src" / "hmrec"
PACKAGE = SRC / "t_eval"
LABEL_DOOR = SRC / "eval" / "t_eval_labels.py"
MODULES = ("__init__.py", "compare.py", "gate.py", "plan.py", "rehearsal.py", "run.py", "trial.py")
SCRIPTS = ("scripts/m9_t_compare.py", "scripts/m9_t_eval.py", "scripts/m9_t_trial.py")
T_EVAL_FILES = (
    *(PACKAGE / name for name in MODULES),
    *(REPO / script for script in SCRIPTS),
    LABEL_DOOR,
)
# Modules that load neither torch nor LightGBM (the driver and the comparison use only these).
NEUTRAL = (
    "hmrec.eval.t_eval_labels",
    "hmrec.t_eval.plan",
    "hmrec.t_eval.gate",
    "hmrec.t_eval.rehearsal",
    "hmrec.t_eval.run",
    "hmrec.t_eval.compare",
)
OTHER_DOORS = (
    "FormalRunPermit",
    "DiagnosticAccess",
    "load_final_evaluator",
    "load_diagnostic_evaluator",
    "label_week_transactions",
    "hmrec.final",
    "hmrec.diagnostics",
    "hmrec.eval.final_eval",
    "hmrec.eval.diagnostic_labels",
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


def test_every_file_of_the_evaluation_is_listed_here() -> None:
    assert sorted(path.name for path in PACKAGE.glob("*.py")) == sorted(MODULES)
    assert sorted(_relative(path) for path in (REPO / "scripts").glob("m9_t_*.py")) == sorted(
        SCRIPTS
    )


def test_a_permit_is_created_only_by_the_gate_after_the_record() -> None:
    creators = [
        _relative(path) for path in _python_files("src", "scripts") if "TEvalPermit(" in _text(path)
    ]
    assert creators == ["src/hmrec/t_eval/gate.py"]
    gate = _text(PACKAGE / "gate.py")
    assert gate.count("TEvalPermit(") == 1
    # In the driver's function the row is appended BEFORE the permit is built.
    driver = gate[gate.index("def begin_formal_run(") : gate.index("def worker_permit(")]
    assert driver.index("append_row(context.paths.m9_t_label_access_file") < driver.index(
        "_permit(context, rehearsal, access)"
    )
    # A worker appends nothing; the access table is written in that one place.
    assert gate.count("append_row(") == 1


def test_only_the_evaluation_uses_the_third_label_door() -> None:
    files = _python_files("src", "scripts")
    users = [_relative(p) for p in files if _imports(p, "hmrec.eval.t_eval_labels")]
    assert users == [
        "src/hmrec/t_eval/gate.py",  # the TEvalPermit class
        "src/hmrec/t_eval/trial.py",  # the two loaders
    ]
    callers = [_relative(p) for p in files if "load_t_eval_evaluator(" in _text(p)]
    assert callers == ["src/hmrec/eval/t_eval_labels.py", "src/hmrec/t_eval/trial.py"]
    # The label file is named in one module of the evaluation, which hands it to the door.
    naming = [_relative(p) for p in T_EVAL_FILES if "labels_file(" in _text(p)]
    assert naming == ["src/hmrec/t_eval/trial.py"]
    for path in T_EVAL_FILES:
        if path != LABEL_DOOR:
            assert "FoldEvaluator.from_file" not in _text(path), path.name


def test_no_other_code_uses_the_evaluation() -> None:
    outside = [p for p in _python_files("src", "scripts") if p not in T_EVAL_FILES]
    users = [_relative(p) for p in outside if _imports(p, "hmrec.t_eval")]
    assert users == []


def test_the_evaluation_uses_no_other_label_door_and_none_of_m8_or_the_diagnostics() -> None:
    for path in T_EVAL_FILES:
        text = _text(path)
        for name in OTHER_DOORS:
            used = _imports(path, name) if name.startswith("hmrec.") else name in text
            assert not used, f"{_relative(path)} uses {name}"
        # The result tables of M7, M8 and the V phase of M9 are never written from here.
        assert "v_phase_metrics_file" not in text and "t_phase_results_file" not in text
    # The V tables are read in one place (the rehearsal comparison) and written nowhere.
    readers = [_relative(p) for p in T_EVAL_FILES if "deepfm_metrics_file" in _text(p)]
    assert readers == ["src/hmrec/t_eval/rehearsal.py"]
    assert all("deepfm_trials_file" not in _text(path) for path in T_EVAL_FILES)


def test_the_processes_of_the_evaluation_load_no_lightgbm_and_the_driver_no_torch() -> None:
    neutral = f"import sys, {', '.join(NEUTRAL)}; " + (
        "bad = [name for name in ('torch', 'lightgbm') if name in sys.modules]; "
        "sys.exit(1 if bad else 0)"
    )
    assert subprocess.run([sys.executable, "-c", neutral], check=False).returncode == 0
    worker = (
        "import sys, hmrec.t_eval.trial, hmrec.deepfm.process; "
        "sys.exit(1 if 'lightgbm' in sys.modules else 0)"
    )
    assert subprocess.run([sys.executable, "-c", worker], check=False).returncode == 0


def test_the_config_files_give_the_configuration_recorded_on_v() -> None:
    paths = get_paths()
    config = load_t_eval_config(paths.t_eval_config_file)
    chosen = chosen_trial_inputs(load_trial_inputs(paths), config, 44)
    assert deepfm_config_id(chosen.config) == config.config_id == "f8811fb59b79"
    assert chosen.config.training.seed == 44
    assert (chosen.config.model.numeric_input, chosen.config.training.loss) == (
        "standardised",
        "listwise_softmax",
    )
    assert chosen.config.training.device == "cpu"
