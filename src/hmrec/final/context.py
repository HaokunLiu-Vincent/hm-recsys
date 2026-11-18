"""Everything a process of the final evaluation needs: configs, the locked Ranker, paths.

The Ranker configuration comes ONLY from hmrec.ranker.locked.load_locked_ranker
(configs/locks/ranker.lock.yaml). Nothing here reads the tuned defaults or the
fixed parameters of configs/ranker.yaml, configs/hpo/v_phase.yaml, or replays a
selection (AGENTS.md §3.3).

Two modes run the same code path:

    rehearsal   predicts the configured V fold (configs/m8_final.yaml) and must
                reproduce the recorded V-phase numbers exactly. Reads no T label.
                Its rows go to the area "m8_rehearsal".
    formal      predicts the T folds. Runs ONCE. Its rows go to the area "t_phase".
"""

from collections.abc import Callable
from dataclasses import dataclass

from hmrec.config import (
    BaseConfig,
    CandidateRunConfig,
    FinalEvalConfig,
    load_base_config,
    load_candidate_run_config,
    load_final_eval_config,
)
from hmrec.data.folds import FoldManifest, load_fold_manifest
from hmrec.eval.report import parse_metric
from hmrec.git_state import code_changed_since
from hmrec.hpo.v_records import AREA_M8_REHEARSAL, AREA_T_PHASE
from hmrec.hpo.v_trial import TrialInputs
from hmrec.locks import load_features_lock
from hmrec.paths import ProjectPaths
from hmrec.ranker.final_window import (
    final_training_window,
    rehearsal_training_window,
)
from hmrec.ranker.locked import LockedRanker, load_locked_ranker
from hmrec.ranker.runtime import LightGBMRuntime
from hmrec.run_records import RunStamp

MODE_REHEARSAL = "rehearsal"
MODE_FORMAL = "formal"
MODES = (MODE_REHEARSAL, MODE_FORMAL)

# Reads a recorded commit and says whether the code at HEAD differs from it.
CodeChanged = Callable[[str], bool]


def area_of(mode: str) -> str:
    """The results area of a mode: "m8_rehearsal" or "t_phase"."""
    if mode == MODE_REHEARSAL:
        return AREA_M8_REHEARSAL
    if mode == MODE_FORMAL:
        return AREA_T_PHASE
    raise ValueError(f"mode must be one of {MODES}, got {mode!r}")


@dataclass(frozen=True)
class FinalContext:
    """What every step of the final evaluation works with.

    Attributes:
        paths: Project paths.
        base: configs/base.yaml (threads, memory limit, chunk sizes).
        config: configs/m8_final.yaml (target folds, rehearsal fold, criteria).
        locked: The locked Ranker configuration (already checked).
        monitor: configs/m6_candidates.yaml: the comparison settings of the recall
            monitor (the same ones M6 used for the F and V folds).
        inputs: The same configuration in the shape the shared trial code takes:
            inputs.ranker IS locked.config, inputs.metrics are config.metrics.
        stamp: HEAD and whether the working tree is dirty.
        runtime: LightGBM version and the loaded OpenMP runtime of this process.
        code_changed: Says whether the code at HEAD differs from a recorded commit
            (hmrec.git_state.code_changed_since; tests pass their own).
    """

    paths: ProjectPaths
    base: BaseConfig
    config: FinalEvalConfig
    locked: LockedRanker
    monitor: CandidateRunConfig
    inputs: TrialInputs
    stamp: RunStamp
    runtime: LightGBMRuntime
    code_changed: CodeChanged

    @property
    def manifest(self) -> FoldManifest:
        """configs/fold_manifest.yaml."""
        return self.inputs.manifest


def check_final_config(
    config: FinalEvalConfig, locked: LockedRanker, manifest: FoldManifest
) -> None:
    """Fail unless configs/m8_final.yaml fits the fold manifest and the locked Ranker.

    Checked: every target fold is a T fold with a full training window of V / T
    folds; the rehearsal fold is a V fold with a full M7 window; every metric is
    cut at a K the locked Top-K can serve.

    Raises:
        ValueError: On any violation.
    """
    weeks = locked.config.window_weeks
    for target_fold in config.target_folds:
        final_training_window(manifest, target_fold, weeks)
    names = list(manifest.names)
    positions = [names.index(fold) for fold in config.target_folds]
    if positions != sorted(positions):
        raise ValueError(f"m8_final.target_folds must be in time order, got {config.target_folds}")
    rehearsal_fold = config.rehearsal_target_fold
    rehearsal_training_window(manifest, rehearsal_fold, rehearsal_fold, weeks)
    for metric in config.metrics:
        _, k = parse_metric(metric)
        if k > locked.config.top_k:
            raise ValueError(
                f"metric {metric} needs the Top-{k}, but the locked Ranker keeps the "
                f"Top-{locked.config.top_k} of every query"
            )


def load_final_context(
    paths: ProjectPaths, stamp: RunStamp, runtime: LightGBMRuntime
) -> FinalContext:
    """Read and check every config of the final evaluation.

    Args:
        paths: Project paths.
        stamp: The code state of this process.
        runtime: hmrec.ranker.runtime.read_lightgbm_runtime() of this process;
            load_locked_ranker refuses a runtime the lock does not record.

    Raises:
        ValueError: If a config is invalid, the Ranker lock fails a check or the
            configs do not fit together.
    """
    base = load_base_config(paths.base_config_file)
    config = load_final_eval_config(paths.m8_config_file)
    locked = load_locked_ranker(paths, base, runtime)
    manifest = load_fold_manifest(paths.fold_manifest_file)
    check_final_config(config, locked, manifest)
    inputs = TrialInputs(
        paths=paths,
        base=base,
        ranker=locked.config,
        manifest=manifest,
        lock=load_features_lock(paths.features_lock_file),
        metrics=config.metrics,
    )
    return FinalContext(
        paths=paths,
        base=base,
        config=config,
        locked=locked,
        monitor=load_candidate_run_config(paths.m6_config_file),
        inputs=inputs,
        stamp=stamp,
        runtime=runtime,
        code_changed=lambda commit: code_changed_since(paths.project_root, commit),
    )
