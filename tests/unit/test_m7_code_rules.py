"""Rules of AGENTS.md §3.1 / §3.2 checked on the text of the M7 code.

- Labels are read in exactly two places: hmrec.eval (FoldEvaluator) and
  hmrec.ranker.sampling (load_training_labels). Everything else in M7 only hands
  a labels path to one of the two guarded loaders.
- M7 code never names a T fold: T folds can only arrive from the command line or
  a config, where the fold guards refuse them (tested in test_ranker_window.py,
  test_v_trial.py and test_ranker_eval.py).
- Feature tables are read only after verify_feature_files.
"""

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
M7_SCRIPTS = sorted((REPO / "scripts").glob("m7_*.py"))
M7_MODULES = [
    *sorted((REPO / "src" / "hmrec" / "ranker").glob("*.py")),
    REPO / "src" / "hmrec" / "hpo" / "v_records.py",
    REPO / "src" / "hmrec" / "hpo" / "v_trial.py",
    REPO / "src" / "hmrec" / "hpo" / "v_phase.py",
    REPO / "src" / "hmrec" / "hpo" / "speed_test.py",
    REPO / "src" / "hmrec" / "hpo" / "v_lock.py",
]


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_the_expected_m7_scripts_exist() -> None:
    assert [path.name for path in M7_SCRIPTS] == [
        "m7_build_dataset.py",
        "m7_ranker_hpo.py",
        "m7_rrf_baseline.py",
        "m7_sample.py",
        "m7_speed_test.py",
        "m7_train_eval.py",
    ]


def test_m7_scripts_never_touch_labels_themselves() -> None:
    names = ("labels_file", "FoldEvaluator", "label_week_transactions", "load_training_labels")
    offenders = [
        f"{path.name}: {name}" for path in M7_SCRIPTS for name in names if name in _text(path)
    ]
    assert offenders == []


def test_m7_modules_read_labels_only_through_the_two_guarded_loaders() -> None:
    # No M7 module builds a FoldEvaluator from a file or reads the label week itself.
    for path in M7_MODULES:
        text = _text(path)
        assert "FoldEvaluator.from_file" not in text, path.name
        assert "label_week_transactions" not in text, path.name
    # The label path is named in two modules only, and each hands it to a guarded loader.
    naming = [path.name for path in M7_MODULES if "labels_file(" in _text(path)]
    assert naming == ["v_trial.py", "v_phase.py"]
    assert _text(REPO / "src/hmrec/hpo/v_trial.py").count("labels_file(") == 2
    assert _text(REPO / "src/hmrec/hpo/v_trial.py").count("load_m7_evaluator(") == 2
    # v_phase only fills SampleJob.labels_file; sampling.sample_fold loads it with the guard.
    assert "labels_file=paths.labels_file(fold_name)" in _text(REPO / "src/hmrec/hpo/v_phase.py")
    sampling = _text(REPO / "src/hmrec/ranker/sampling.py")
    assert "load_training_labels(job.fold, job.labels_file, job.allowed_phases)" in sampling


def test_the_independent_check_gets_labels_through_the_guard() -> None:
    text = _text(REPO / "scripts" / "checks" / "check_ranker_eval.py")
    assert "load_m7_evaluator(" in text and "FoldEvaluator" not in text
    # It recomputes the metrics itself: nothing of the metric code is imported.
    assert "hmrec.eval.metrics" not in text.split('"""', 2)[2]
    assert "hmrec.ranker" not in text.split('"""', 2)[2]


def test_m7_code_names_no_t_fold() -> None:
    t_fold = re.compile(r"""["']T[1-4]["']""")
    offenders = [path.name for path in (*M7_SCRIPTS, *M7_MODULES) if t_fold.search(_text(path))]
    assert offenders == []


def test_feature_tables_are_verified_before_they_are_read() -> None:
    # Every M7 module that reads artifacts/features/ calls verify_feature_files.
    for name in ("ranker/sampling.py", "hpo/v_trial.py"):
        text = _text(REPO / "src" / "hmrec" / name)
        assert "verify_feature_files(" in text, name
    readers = [path.name for path in M7_MODULES if "features_dir" in _text(path)]
    assert set(readers) <= {"sampling.py", "v_trial.py", "v_phase.py"}


def test_m7_processes_never_import_torch() -> None:
    # torch and LightGBM each load their own OpenMP runtime; both in one process abort,
    # hang or crash (tests/conftest.py). So nothing an M7 script imports may import torch
    # (or the recall package, which imports it).
    modules = [
        "hmrec.hpo.speed_test",
        "hmrec.hpo.v_lock",
        "hmrec.hpo.v_phase",
        "hmrec.hpo.v_trial",
        "hmrec.hpo.v_records",
        "hmrec.ranker.locked",
        "hmrec.eval.ranker_eval",
        "hmrec.pipeline",
        "hmrec.run_records",
        "hmrec.memory",
        "hmrec.threads",
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
    # And every import of an M7 script is one of those modules (or below them).
    allowed = ("hmrec.hpo.v_", "hmrec.hpo.speed_test", "hmrec.pipeline", "hmrec.run_records",
               "hmrec.memory", "hmrec.threads", "hmrec.logging_utils", "hmrec.config",
               "hmrec.ranker.")  # fmt: skip
    for path in M7_SCRIPTS:
        for line in _text(path).splitlines():
            if line.startswith("from hmrec"):
                module = line.split()[1]
                assert module.startswith(allowed), f"{path.name}: {line}"
