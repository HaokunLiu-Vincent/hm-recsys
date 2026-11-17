"""Every file path in the project comes from here (AGENTS.md §5).

All paths are derived from three roots in configs/base.yaml:
`paths.data_root`, `paths.artifacts_root`, `paths.results_root`.
Relative roots are resolved against the repository root, so the result does not
depend on the current working directory.

Usage:
    paths = get_paths()                      # reads configs/base.yaml
    paths.transactions_csv                   # data/raw/transactions_train.csv
    paths.candidates_file("D1", "pop")       # artifacts/candidates/D1/pop.parquet

The layout follows docs/proposal.md §14 and §14.2.
"""

from dataclasses import dataclass
from pathlib import Path

from hmrec.config import BaseConfig, load_base_config

# src/hmrec/paths.py -> parents[0] = src/hmrec, parents[1] = src, parents[2] = repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"
BASE_CONFIG_FILE = CONFIGS_DIR / "base.yaml"


def resolve_root(root: str, project_root: Path) -> Path:
    """Turn a root folder from base.yaml into an absolute path.

    Args:
        root: Folder name from the config; absolute or relative.
        project_root: Folder that relative roots are resolved against.

    Returns:
        The absolute path. A relative `root` is joined to `project_root`.
    """
    root_path = Path(root).expanduser()
    if root_path.is_absolute():
        return root_path
    return project_root / root_path


@dataclass(frozen=True)
class ProjectPaths:
    """All project paths, derived from the three roots.

    Attributes:
        project_root: Repository root (holds configs/, src/, docs/).
        data_root: Folder with raw/ and processed/.
        artifacts_root: Folder for regenerable outputs.
        results_root: Folder for committed summary tables.
    """

    project_root: Path
    data_root: Path
    artifacts_root: Path
    results_root: Path

    # ------------------------------------------------------------ configs

    @property
    def configs_dir(self) -> Path:
        """configs/."""
        return self.project_root / "configs"

    @property
    def base_config_file(self) -> Path:
        """configs/base.yaml."""
        return self.configs_dir / "base.yaml"

    @property
    def fold_manifest_file(self) -> Path:
        """configs/fold_manifest.yaml."""
        return self.configs_dir / "fold_manifest.yaml"

    @property
    def selection_rule_file(self) -> Path:
        """configs/selection_rule.yaml."""
        return self.configs_dir / "selection_rule.yaml"

    @property
    def d_phase_grid_file(self) -> Path:
        """configs/hpo/d_phase.yaml: the D-phase experiments (proposal §5)."""
        return self.configs_dir / "hpo" / "d_phase.yaml"

    @property
    def preprocess_config_file(self) -> Path:
        """configs/preprocess.yaml."""
        return self.configs_dir / "preprocess.yaml"

    @property
    def catalog_config_file(self) -> Path:
        """configs/catalog.yaml."""
        return self.configs_dir / "catalog.yaml"

    @property
    def queries_config_file(self) -> Path:
        """configs/queries.yaml."""
        return self.configs_dir / "queries.yaml"

    @property
    def union_check_config_file(self) -> Path:
        """configs/m4_union_check.yaml."""
        return self.configs_dir / "m4_union_check.yaml"

    @property
    def m6_config_file(self) -> Path:
        """configs/m6_candidates.yaml: monitoring and stop settings of M6 candidate generation."""
        return self.configs_dir / "m6_candidates.yaml"

    def recall_config_file(self, channel_file_stem: str) -> Path:
        """configs/recall/{channel_file_stem}.yaml, e.g. "r1_popularity"."""
        return self.configs_dir / "recall" / f"{channel_file_stem}.yaml"

    @property
    def locks_dir(self) -> Path:
        """configs/locks/ (append-only)."""
        return self.configs_dir / "locks"

    @property
    def channels_lock_file(self) -> Path:
        """configs/locks/channels.lock.yaml: the locked configs of R1-R5 (M5)."""
        return self.locks_dir / "channels.lock.yaml"

    @property
    def twotower_lock_file(self) -> Path:
        """configs/locks/twotower.lock.yaml: the locked config and model files of R6 (M5)."""
        return self.locks_dir / "twotower.lock.yaml"

    # ------------------------------------------------------------ raw data

    @property
    def raw_dir(self) -> Path:
        """data/raw/: the original Kaggle CSV files."""
        return self.data_root / "raw"

    @property
    def transactions_csv(self) -> Path:
        """data/raw/transactions_train.csv."""
        return self.raw_dir / "transactions_train.csv"

    @property
    def customers_csv(self) -> Path:
        """data/raw/customers.csv."""
        return self.raw_dir / "customers.csv"

    @property
    def articles_csv(self) -> Path:
        """data/raw/articles.csv."""
        return self.raw_dir / "articles.csv"

    @property
    def sample_submission_csv(self) -> Path:
        """data/raw/sample_submission.csv."""
        return self.raw_dir / "sample_submission.csv"

    # ------------------------------------------------------------ processed data

    @property
    def processed_dir(self) -> Path:
        """data/processed/: M0 Parquet outputs and the precomputed embeddings."""
        return self.data_root / "processed"

    @property
    def image_features_dir(self) -> Path:
        """data/processed/image_features/ (read-only, CLIP ViT-B/32)."""
        return self.processed_dir / "image_features"

    @property
    def image_embeddings_file(self) -> Path:
        """float32 array [n_images, 512]; row i matches embedding_row = i."""
        return self.image_features_dir / "image_embeddings.npy"

    @property
    def image_article_ids_file(self) -> Path:
        """Parquet: embedding_row (int64) -> article_id (10-char string)."""
        return self.image_features_dir / "image_article_ids.parquet"

    @property
    def text_features_dir(self) -> Path:
        """data/processed/text_features/ (read-only, all-MiniLM-L6-v2)."""
        return self.processed_dir / "text_features"

    @property
    def text_embeddings_file(self) -> Path:
        """float32 array [n_articles, 384]; row i matches embedding_row = i."""
        return self.text_features_dir / "text_embeddings.npy"

    @property
    def text_article_ids_file(self) -> Path:
        """Parquet: embedding_row (int64) -> article_id (10-char string)."""
        return self.text_features_dir / "text_article_ids.parquet"

    # ------------------------------------------------------------ M0 outputs (data/processed/)

    @property
    def customer_id_map_file(self) -> Path:
        """Parquet: customer_idx (int32) <-> customer_id (64-char string)."""
        return self.processed_dir / "customer_id_map.parquet"

    @property
    def article_id_map_file(self) -> Path:
        """Parquet: article_idx (int32) <-> article_id (10-char string)."""
        return self.processed_dir / "article_id_map.parquet"

    @property
    def transactions_file(self) -> Path:
        """Parquet: one row per purchase line, sorted by (t_dat_day, customer_idx)."""
        return self.processed_dir / "transactions.parquet"

    @property
    def customers_file(self) -> Path:
        """Parquet: one row per customer, keyed by customer_idx."""
        return self.processed_dir / "customers.parquet"

    @property
    def articles_file(self) -> Path:
        """Parquet: one row per article, keyed by article_idx."""
        return self.processed_dir / "articles.parquet"

    @property
    def processed_embeddings_dir(self) -> Path:
        """data/processed/embeddings/: embeddings re-ordered by article_idx."""
        return self.processed_dir / "embeddings"

    @property
    def image_embeddings_by_idx_file(self) -> Path:
        """float32 array [n_articles, 512]; row i is article_idx i (zeros when missing)."""
        return self.processed_embeddings_dir / "image_by_article_idx.npy"

    @property
    def image_embedding_mask_file(self) -> Path:
        """Bool array [n_articles]; True where article_idx i has an image vector."""
        return self.processed_embeddings_dir / "image_mask.npy"

    @property
    def text_embeddings_by_idx_file(self) -> Path:
        """float32 array [n_articles, 384]; row i is article_idx i (zeros when missing)."""
        return self.processed_embeddings_dir / "text_by_article_idx.npy"

    @property
    def text_embedding_mask_file(self) -> Path:
        """Bool array [n_articles]; True where article_idx i has a text vector."""
        return self.processed_embeddings_dir / "text_mask.npy"

    @property
    def processed_metadata_file(self) -> Path:
        """JSON: day_zero, age buckets and config hash used to build data/processed/."""
        return self.processed_dir / "metadata.json"

    # ------------------------------------------------------------ artifacts (proposal §14.2)

    def queries_file(self, fold: str) -> Path:
        """artifacts/queries/{fold}.parquet."""
        return self.artifacts_root / "queries" / f"{fold}.parquet"

    def labels_file(self, fold: str) -> Path:
        """artifacts/labels/{fold}.parquet. Read only by eval/ and ranker/sampling.py."""
        return self.artifacts_root / "labels" / f"{fold}.parquet"

    def candidates_dir(self, fold: str) -> Path:
        """artifacts/candidates/{fold}/."""
        return self.artifacts_root / "candidates" / fold

    def candidates_file(self, fold: str, channel: str) -> Path:
        """artifacts/candidates/{fold}/{channel}.parquet."""
        return self.candidates_dir(fold) / f"{channel}.parquet"

    def r1_bucket_scores_file(self, fold: str) -> Path:
        """artifacts/candidates/{fold}/r1_bucket_scores.parquet: R1's full S(g, i) table.

        One row per (age_bucket, article_idx), not cut to Top-100 and not limited to
        A(a); read by features/ for pop_bucket_score (proposal §8.3). It is NOT a
        channel's candidate file.
        """
        return self.candidates_dir(fold) / "r1_bucket_scores.parquet"

    def wals_vectors_dir(self, fold: str) -> Path:
        """artifacts/models/wals/{fold}/: R5 vectors of the fold (hmrec.vectors files)."""
        return self.model_dir("wals", fold)

    def twotower_vectors_dir(self, fold: str) -> Path:
        """artifacts/models/twotower/vectors/{fold}/: R6 vectors of the fold (hmrec.vectors files).

        Computed with the model that PREDICTED the fold, i.e. before the fold's fine-tune.
        """
        return self.artifacts_root / "models" / "twotower" / "vectors" / fold

    def twotower_locked_chain_checkpoint_file(self, fold: str) -> Path:
        """artifacts/models/twotower/chain_locked/{fold}.pt: chain model after the fold's fine-tune.

        The F / V / T continuation of the training chain (proposal §6.6). The model
        of fold k predicts fold k + 1; its sha256 is in results/m6_twotower_chain.csv.
        """
        return self.artifacts_root / "models" / "twotower" / "chain_locked" / f"{fold}.pt"

    def features_dir(self, fold: str) -> Path:
        """artifacts/features/{fold}/ (holds part-*.parquet)."""
        return self.artifacts_root / "features" / fold

    def lgb_dataset_dir(self, run_id: str) -> Path:
        """artifacts/lgb_datasets/{run_id}/."""
        return self.artifacts_root / "lgb_datasets" / run_id

    def model_dir(self, model_name: str, fold: str) -> Path:
        """artifacts/models/{model_name}/{fold}/, model_name in {wals, twotower, ranker}."""
        return self.artifacts_root / "models" / model_name / fold

    def predictions_file(self, run_id: str, fold: str) -> Path:
        """artifacts/predictions/{run_id}/{fold}.parquet."""
        return self.artifacts_root / "predictions" / run_id / f"{fold}.parquet"

    def run_dir(self, run_id: str) -> Path:
        """artifacts/runs/{run_id}/: config snapshot, logs, metrics."""
        return self.artifacts_root / "runs" / run_id

    def twotower_data_dir(self, window_name: str) -> Path:
        """artifacts/twotower/data/{window_name}/: Two-Tower training arrays, e.g. "base"."""
        return self.artifacts_root / "twotower" / "data" / window_name

    def m4_candidates_file(self, fold: str, channel: str, trial_id: str) -> Path:
        """artifacts/m4/candidates/{fold}/{channel}_{trial_id}.parquet (M4 union check)."""
        return self.artifacts_root / "m4" / "candidates" / fold / f"{channel}_{trial_id}.parquet"

    def lock_check_candidates_file(self, fold: str, channel: str, run_name: str = "") -> Path:
        """artifacts/m5/lock_check/{fold}/{channel}.parquet: candidates of the M5 lock check.

        Kept apart from artifacts/candidates/ on purpose: these files only prove
        that the lock files reproduce the recorded D-phase scores. No later step
        reads them.

        A REPEATED check (e.g. a regression check after a refactor) passes a
        `run_name` and writes to artifacts/m5/lock_check_{run_name}/{fold}/..., so
        the files of the first check are never touched.
        """
        folder = "lock_check" if run_name == "" else f"lock_check_{run_name}"
        return self.artifacts_root / "m5" / folder / fold / f"{channel}.parquet"

    def twotower_checkpoint_file(self, epoch: int) -> Path:
        """artifacts/models/twotower/base/epoch_{epoch}.pt: base model after `epoch` epochs."""
        return self.model_dir("twotower", "base") / f"epoch_{epoch}.pt"

    def twotower_chain_checkpoint_file(self, trial_id: str, fold: str) -> Path:
        """artifacts/models/twotower/chain/{trial_id}/{fold}.pt: model after fine-tuning on fold."""
        return self.artifacts_root / "models" / "twotower" / "chain" / trial_id / f"{fold}.pt"

    def hpo_trial_dir(self, phase: str, channel: str, trial_id: str) -> Path:
        """artifacts/hpo/{phase}/{channel}/{trial_id}/: per-trial side files (pool flags)."""
        return self.artifacts_root / "hpo" / phase / channel / trial_id

    @property
    def union_precheck_dir(self) -> Path:
        """artifacts/hpo/D/union_precheck/: R1-R5 union pre-check (kept apart from M4)."""
        return self.artifacts_root / "hpo" / "D" / "union_precheck"

    def union_precheck_candidates_file(self, fold: str, channel: str, trial_id: str) -> Path:
        """artifacts/hpo/D/union_precheck/{fold}/{channel}_{trial_id}.parquet."""
        return self.union_precheck_dir / fold / f"{channel}_{trial_id}.parquet"

    # ------------------------------------------------------------ results (committed)

    @property
    def d_phase_results_file(self) -> Path:
        """results/d_phase.csv."""
        return self.results_root / "d_phase.csv"

    def d_phase_steps_file(self, channel: str) -> Path:
        """results/d_phase_steps/{channel}.csv: every step's candidates and decision."""
        return self.results_root / "d_phase_steps" / f"{channel}.csv"

    @property
    def union_precheck_results_file(self) -> Path:
        """results/d_phase_union_precheck.csv: union recall table of the R1-R5 pre-check."""
        return self.results_root / "d_phase_union_precheck.csv"

    @property
    def m4_union_check_results_file(self) -> Path:
        """results/m4_union_check.csv: the six-channel union check (proposal §5.7)."""
        return self.results_root / "m4_union_check.csv"

    def m4_union_check_variant_file(self, variant: str) -> Path:
        """results/m4_union_check_{variant}.csv: the union check of one variant."""
        return self.results_root / f"m4_union_check_{variant}.csv"

    @property
    def m4_trials_file(self) -> Path:
        """results/m4_trials.csv: trials of M4 variants (rule outcomes; never select)."""
        return self.results_root / "m4_trials.csv"

    @property
    def m6_runs_file(self) -> Path:
        """results/m6_runs.csv: one row per M6 process (fold, step): time and true memory peak."""
        return self.results_root / "m6_runs.csv"

    @property
    def m6_candidates_results_file(self) -> Path:
        """results/m6_candidates.csv: per fold, monitoring rows of each channel and of the union."""
        return self.results_root / "m6_candidates.csv"

    @property
    def m6_twotower_chain_results_file(self) -> Path:
        """results/m6_twotower_chain.csv: one row per saved chain model (sha256, losses)."""
        return self.results_root / "m6_twotower_chain.csv"

    @property
    def m5_lock_check_results_file(self) -> Path:
        """results/m5_lock_check.csv: one row per run of the M5 lock check (score, time, memory)."""
        return self.results_root / "m5_lock_check.csv"

    def d_phase_seeds_file(self, channel: str) -> Path:
        """results/d_phase_steps/{channel}_seeds.csv: extra-seed runs (reported only)."""
        return self.results_root / "d_phase_steps" / f"{channel}_seeds.csv"

    @property
    def v_phase_results_file(self) -> Path:
        """results/v_phase.csv."""
        return self.results_root / "v_phase.csv"

    @property
    def t_phase_results_file(self) -> Path:
        """results/t_phase.csv."""
        return self.results_root / "t_phase.csv"


def build_paths(config: BaseConfig, project_root: Path) -> ProjectPaths:
    """Derive ProjectPaths from a BaseConfig.

    This function does not touch the file system; it only builds paths.

    Args:
        config: Parsed base config.
        project_root: Folder that relative roots are resolved against.

    Returns:
        ProjectPaths with absolute roots.
    """
    return ProjectPaths(
        project_root=project_root,
        data_root=resolve_root(config.paths.data_root, project_root),
        artifacts_root=resolve_root(config.paths.artifacts_root, project_root),
        results_root=resolve_root(config.paths.results_root, project_root),
    )


def get_paths(base_config_file: Path = BASE_CONFIG_FILE) -> ProjectPaths:
    """Read base.yaml and return the project paths.

    Args:
        base_config_file: Path to base.yaml. Defaults to configs/base.yaml.
            Its grandparent folder (the folder that holds configs/) is used as
            the project root for resolving relative roots.

    Returns:
        ProjectPaths with absolute roots.
    """
    config = load_base_config(base_config_file)
    project_root = base_config_file.resolve().parent.parent
    return build_paths(config, project_root)
