"""Rules the DeepFM Kaggle submission must keep, checked by reading the source.

- Torch side only: nothing of it loads LightGBM, and the modules that only read
  and compare load no torch either.
- Labels: no label file is named; the read of the sample files' label column is
  recorded BEFORE a sample file is opened.
- It uses none of the guarded T-label doors and nothing of the T evaluation's code.
- It writes its own tables and folders only: nothing of the LightGBM submission.
- The config files give the configuration chosen on V, with seed 42.
"""

import subprocess
import sys
from pathlib import Path

from hmrec.config import load_kaggle_deepfm_config
from hmrec.deepfm.process import load_trial_inputs
from hmrec.deepfm.records import deepfm_config_id
from hmrec.git_state import git_state
from hmrec.kaggle.plan import RUN_SUBMISSION
from hmrec.kaggle_deepfm.context import load_deepfm_context
from hmrec.kaggle_deepfm.train import chosen_inputs
from hmrec.paths import get_paths
from hmrec.run_records import RunStamp
from hmrec.t_eval.trial import chosen_trial_inputs

REPO = Path(__file__).resolve().parents[2]
PACKAGE = REPO / "src" / "hmrec" / "kaggle_deepfm"
MODULES = ("__init__.py", "context.py", "report.py", "score.py", "train.py")
SCRIPTS = (
    "scripts/deepfm_kaggle_record.py",
    "scripts/deepfm_kaggle_run.py",
    "scripts/deepfm_kaggle_step.py",
)
FILES = (*(PACKAGE / name for name in MODULES), *(REPO / script for script in SCRIPTS))
NEUTRAL = ("hmrec.kaggle_deepfm.context", "hmrec.kaggle_deepfm.report")
FORBIDDEN_IMPORTS = (
    "lightgbm",
    "hmrec.kaggle.ranker",
    "hmrec.t_eval",
    "hmrec.eval.t_eval_labels",
    "hmrec.eval.final_eval",
    "hmrec.eval.diagnostic_labels",
    "hmrec.final",
    "hmrec.diagnostics",
)
# Tables and files of the LightGBM submission: read at most, never written from here.
LIGHTGBM_OUTPUTS = (
    "kaggle_runs_file",
    "kaggle_label_access_file",
    "kaggle_rehearsal_file",
    "kaggle_files_file",
    "kaggle_submission_results_file",
    "kaggle_top_file",
    "kaggle_model_file",
    "kaggle_submission_file",
    "kaggle_submission_gzip_file",
)


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _imports(path: Path, module: str) -> bool:
    """True if the file has an import statement of `module` or of something inside it."""
    statements = (f"from {module}", f"import {module}")
    return any(line.strip().startswith(statements) for line in _text(path).splitlines())


def test_every_file_of_the_submission_is_listed_here() -> None:
    assert sorted(path.name for path in PACKAGE.glob("*.py")) == sorted(MODULES)
    found = sorted(
        path.relative_to(REPO).as_posix() for path in (REPO / "scripts").glob("deepfm_kaggle_*.py")
    )
    assert found == sorted(SCRIPTS)


def test_nothing_loads_lightgbm_or_uses_a_guarded_label_door() -> None:
    for path in FILES:
        for module in FORBIDDEN_IMPORTS:
            assert not _imports(path, module), f"{path.name} imports {module}"
        text = _text(path)
        for name in ("labels_file", "label_week_transactions", "FoldEvaluator", "TEvalPermit"):
            assert name not in text, f"{path.name} names {name}"
    worker = (
        "import sys, hmrec.kaggle_deepfm.train, hmrec.kaggle_deepfm.score, "
        "hmrec.kaggle_deepfm.report; sys.exit(1 if 'lightgbm' in sys.modules else 0)"
    )
    assert subprocess.run([sys.executable, "-c", worker], check=False).returncode == 0
    neutral = f"import sys, {', '.join(NEUTRAL)}; " + (
        "bad = [name for name in ('torch', 'lightgbm') if name in sys.modules]; "
        "sys.exit(1 if bad else 0)"
    )
    assert subprocess.run([sys.executable, "-c", neutral], check=False).returncode == 0


def test_the_label_read_is_recorded_before_a_sample_file_is_opened() -> None:
    train = _text(PACKAGE / "train.py")
    prepare = train[train.index("def prepare_run_training(") : train.index("def train_run_model(")]
    recorded = prepare.index("record_label_access(")
    assert recorded < prepare.index("check_sample_files(") < prepare.index("load_training_rows(")
    # The label access table is written in one place, by the context module.
    writers = [path.name for path in FILES if "kaggle_deepfm_label_access_file" in _text(path)]
    assert writers == ["context.py"]


def test_nothing_of_the_lightgbm_submission_is_written() -> None:
    for path in FILES:
        text = _text(path)
        for name in LIGHTGBM_OUTPUTS:
            # The DeepFM names contain "kaggle_deepfm_", so they do not match these.
            assert f"paths.{name}" not in text and f".{name}(" not in text, (
                f"{path.name} uses {name}"
            )
    # No other code uses the new package.
    outside = [
        path
        for folder in ("src", "scripts")
        for path in sorted((REPO / folder).rglob("*.py"))
        if path not in FILES
    ]
    assert [path.name for path in outside if _imports(path, "hmrec.kaggle_deepfm")] == []


def test_the_config_files_give_the_chosen_configuration_with_seed_42() -> None:
    paths = get_paths()
    commit, dirty = git_state(paths.project_root, paths.results_root)
    context = load_deepfm_context(RUN_SUBMISSION, paths, RunStamp(commit, dirty))
    settings = load_kaggle_deepfm_config(paths.kaggle_deepfm_config_file)
    assert context.settings == settings
    inputs = chosen_inputs(context, load_trial_inputs(paths))
    assert deepfm_config_id(inputs.config) == settings.config_id == "f8811fb59b79"
    assert (inputs.config.training.seed, inputs.config.training.device) == (42, "cpu")
    # The same configuration the stage-2 evaluation built for this seed.
    stage2 = chosen_trial_inputs(load_trial_inputs(paths), context.chosen, 42)
    assert inputs.config == stage2.config
    assert context.kaggle.target.name == "S"
