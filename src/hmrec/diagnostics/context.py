"""What the descriptive diagnostics work with: configs, folds, the identity of the locked Ranker.

The diagnostics train nothing and score nothing. From the locked Ranker they
need only the values that say WHERE its saved results are: the configuration
id, the number of trees and the Top-K. These come from
hmrec.ranker.locked.load_locked_ranker, the one entry for the Ranker
configuration (AGENTS.md §3.3).
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from hmrec.config import (
    BaseConfig,
    DiagnosticsConfig,
    PreprocessConfig,
    load_base_config,
    load_diagnostics_config,
    load_final_eval_config,
    load_preprocess_config,
)
from hmrec.data.folds import Fold, FoldManifest, load_fold_manifest
from hmrec.eval.report import parse_metric
from hmrec.git_state import changed_existing_files
from hmrec.locks import load_features_lock
from hmrec.paths import ProjectPaths
from hmrec.ranker.ablation import wide_table_columns
from hmrec.ranker.locked import load_locked_ranker
from hmrec.ranker.runtime import LightGBMRuntime
from hmrec.run_records import RunStamp

# (recorded commit, paths) -> the files under the paths that existed at that commit
# and are changed or deleted at HEAD (hmrec.git_state.changed_existing_files).
ChangedExisting = Callable[[str, Sequence[str]], list[str]]


@dataclass(frozen=True)
class RankerIdentity:
    """Where the saved results of the locked Ranker are.

    Attributes:
        config_id: Id of the locked configuration (rows of the metric tables,
            folder of the saved predictions).
        num_trees: The locked number of trees (the rows and per-query values used).
        top_k: Rows the Ranker keeps per query (12); the RRF baseline keeps the same.
        wide_columns: The columns of the frozen wide table, to verify the files.
    """

    config_id: str
    num_trees: int
    top_k: int
    wide_columns: tuple[str, ...]


@dataclass(frozen=True)
class DiagnosticsContext:
    """Everything a run of the diagnostics needs.

    Attributes:
        paths: Project paths.
        base: configs/base.yaml (memory limit).
        config: configs/m8_diagnostics.yaml.
        preprocess: configs/preprocess.yaml (day_zero; checked against the processed data).
        manifest: configs/fold_manifest.yaml.
        ranker: The identity of the locked Ranker.
        metrics: The metrics of the recorded metric tables, the main metric first
            (configs/m8_final.yaml): "map@12", "recall@12", "ndcg@12".
        stamp: HEAD and whether the working tree is dirty.
        changed_existing: See ChangedExisting (tests pass their own).
    """

    paths: ProjectPaths
    base: BaseConfig
    config: DiagnosticsConfig
    preprocess: PreprocessConfig
    manifest: FoldManifest
    ranker: RankerIdentity
    metrics: tuple[str, ...]
    stamp: RunStamp
    changed_existing: ChangedExisting

    @property
    def map_metric(self) -> str:
        """The main metric, "map@12"."""
        return self.metrics[0]

    @property
    def recall_metric(self) -> str:
        """Recall at the main metric's cut-off, "recall@12"."""
        _, k = parse_metric(self.map_metric)
        return f"recall@{k}"

    def folds_of(self, phases: Sequence[str]) -> tuple[Fold, ...]:
        """The folds of the given phases, in the manifest's time order."""
        return tuple(fold for fold in self.manifest.folds if fold.phase in phases)


def check_metrics(metrics: Sequence[str]) -> None:
    """Fail unless the main metric is a MAP and the recall at the same cut-off is recorded.

    Raises:
        ValueError: Otherwise (table A needs both).
    """
    kind, k = parse_metric(metrics[0])
    if kind != "map" or f"recall@{k}" not in metrics:
        raise ValueError(
            f"the diagnostics need map@K as the main metric and recall@K next to it, "
            f"got {list(metrics)}"
        )


def load_diagnostics_context(
    paths: ProjectPaths, stamp: RunStamp, runtime: LightGBMRuntime
) -> DiagnosticsContext:
    """Read and check every config the diagnostics need.

    Args:
        paths: Project paths.
        stamp: The code state of this process.
        runtime: hmrec.ranker.runtime.read_lightgbm_runtime() of this process
            (load_locked_ranker checks it against the lock; nothing is trained).

    Raises:
        ValueError: If a config is invalid or the Ranker lock fails a check.
    """
    base = load_base_config(paths.base_config_file)
    locked = load_locked_ranker(paths, base, runtime)
    metrics = load_final_eval_config(paths.m8_config_file).metrics
    check_metrics(metrics)
    features_lock = load_features_lock(paths.features_lock_file)
    return DiagnosticsContext(
        paths=paths,
        base=base,
        config=load_diagnostics_config(paths.m8_diagnostics_config_file),
        preprocess=load_preprocess_config(paths.preprocess_config_file),
        manifest=load_fold_manifest(paths.fold_manifest_file),
        ranker=RankerIdentity(
            config_id=locked.config_id,
            num_trees=locked.num_trees,
            top_k=locked.config.top_k,
            wide_columns=wide_table_columns(features_lock),
        ),
        metrics=tuple(metrics),
        stamp=stamp,
        changed_existing=lambda commit, protected: changed_existing_files(
            paths.project_root, commit, protected
        ),
    )
