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
    def features_config_file(self) -> Path:
        """configs/features.yaml: the 56 wide-table features and their dtypes (proposal §8)."""
        return self.configs_dir / "features.yaml"

    @property
    def ranker_config_file(self) -> Path:
        """configs/ranker.yaml: the Ranker's fixed parameters and tuned defaults (proposal §9)."""
        return self.configs_dir / "ranker.yaml"

    @property
    def v_phase_grid_file(self) -> Path:
        """configs/hpo/v_phase.yaml: the V-phase steps of the Ranker and their priority order."""
        return self.configs_dir / "hpo" / "v_phase.yaml"

    @property
    def m8_config_file(self) -> Path:
        """configs/m8_final.yaml: target folds, rehearsal fold and acceptance criteria of M8."""
        return self.configs_dir / "m8_final.yaml"

    @property
    def m8_diagnostics_config_file(self) -> Path:
        """configs/m8_diagnostics.yaml: what the descriptive diagnostics after M8 describe."""
        return self.configs_dir / "m8_diagnostics.yaml"

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

    @property
    def features_lock_file(self) -> Path:
        """configs/locks/features.lock.yaml: the locked Ranker features (M6, proposal §8.7)."""
        return self.locks_dir / "features.lock.yaml"

    @property
    def ranker_lock_file(self) -> Path:
        """configs/locks/ranker.lock.yaml: the final Ranker configuration (M7, proposal §9.6)."""
        return self.locks_dir / "ranker.lock.yaml"

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

    def features_part_file(self, fold: str, part: int) -> Path:
        """artifacts/features/{fold}/part-{part:05d}.parquet: the wide table of one query chunk."""
        return self.features_dir(fold) / f"part-{part:05d}.parquet"

    def features_metadata_file(self, fold: str) -> Path:
        """artifacts/features/{fold}/features.json: written LAST; it marks a finished fold."""
        return self.features_dir(fold) / "features.json"

    @property
    def category_codes_file(self) -> Path:
        """artifacts/features/category_codes.json: the category code table shared by all folds."""
        return self.artifacts_root / "features" / "category_codes.json"

    def lgb_dataset_dir(self, run_id: str) -> Path:
        """artifacts/lgb_datasets/{run_id}/."""
        return self.artifacts_root / "lgb_datasets" / run_id

    def model_dir(self, model_name: str, fold: str) -> Path:
        """artifacts/models/{model_name}/{fold}/, model_name in {wals, twotower, ranker}."""
        return self.artifacts_root / "models" / model_name / fold

    def predictions_file(self, run_id: str, fold: str) -> Path:
        """artifacts/predictions/{run_id}/{fold}.parquet."""
        return self.artifacts_root / "predictions" / run_id / f"{fold}.parquet"

    # ------------------------------------------------------------ ranker (M7, M8)

    def ranker_sample_file(self, experiment: str, fold: str, ratio: int) -> Path:
        """artifacts/ranker/samples/{experiment}/{fold}/r{ratio}.parquet: sampled training rows.

        experiment is "full" or "minus_r6"; ratio is the negatives per positive.
        """
        return self.artifacts_root / "ranker" / "samples" / experiment / fold / f"r{ratio}.parquet"

    def ranker_sample_record_file(self, experiment: str, fold: str, ratio: int) -> Path:
        """The record next to a sample file (r{ratio}.json): written LAST, marks it finished."""
        return self.ranker_sample_file(experiment, fold, ratio).with_suffix(".json")

    def ranker_dataset_dir(self, experiment: str, target_fold: str, ratio: int) -> Path:
        """artifacts/lgb_datasets/{experiment}/{target_fold}_r{ratio}/: the binary Dataset.

        The Dataset holds the 4 training weeks of the model that predicts target_fold.
        """
        return self.lgb_dataset_dir(experiment) / f"{target_fold}_r{ratio}"

    def final_dataset_dir(self, area: str, experiment: str, target_fold: str, ratio: int) -> Path:
        """artifacts/lgb_datasets/{area}/{experiment}/{target_fold}_r{ratio}/: an M8 Dataset.

        area is "t_phase" or "m8_rehearsal". The area keeps M8's Datasets apart from
        the V phase's (ranker_dataset_dir): the rehearsal builds its own Dataset of
        the rehearsal fold instead of loading the one M7 built.
        """
        return self.lgb_dataset_dir(area) / experiment / f"{target_fold}_r{ratio}"

    def ranker_model_file(self, area: str, experiment: str, config_id: str, fold: str) -> Path:
        """artifacts/models/ranker/{area}/{experiment}/{config_id}/{fold}/model.txt.

        area is "v_phase", "speed_test", "t_phase" or "m8_rehearsal"; fold is the fold
        the model predicts.
        """
        return self.model_dir("ranker", area) / experiment / config_id / fold / "model.txt"

    def ranker_predictions_file(
        self, area: str, experiment: str, config_id: str, fold: str
    ) -> Path:
        """artifacts/predictions/{area}/{experiment}/{config_id}/{fold}.parquet: Top-K rows."""
        return self.predictions_file(f"{area}/{experiment}/{config_id}", fold)

    def ranker_per_query_file(self, area: str, experiment: str, config_id: str, fold: str) -> Path:
        """artifacts/predictions/.../{fold}_per_query.parquet: every query's metric values."""
        predictions = self.ranker_predictions_file(area, experiment, config_id, fold)
        return predictions.with_name(f"{fold}_per_query.parquet")

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
    def m6_features_results_file(self) -> Path:
        """results/m6_features.csv: per fold and feature, missing rate, min, max and mean."""
        return self.results_root / "m6_features.csv"

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
        """results/v_phase.csv: one row per Ranker trial (a configuration on one fold)."""
        return self.results_root / "v_phase.csv"

    @property
    def v_phase_metrics_file(self) -> Path:
        """results/v_phase_metrics.csv: one row per (configuration, fold, number of trees)."""
        return self.results_root / "v_phase_metrics.csv"

    @property
    def v_phase_steps_file(self) -> Path:
        """results/v_phase_steps.csv: every step's configurations, their k and the decision."""
        return self.results_root / "v_phase_steps.csv"

    @property
    def v_phase_feature_importance_file(self) -> Path:
        """results/v_phase_feature_importance.csv: gain of the selected configuration's models."""
        return self.results_root / "v_phase_feature_importance.csv"

    @property
    def m7_runs_file(self) -> Path:
        """results/m7_runs.csv: one row per M7 helper process (sampling, Dataset, baseline)."""
        return self.results_root / "m7_runs.csv"

    @property
    def m7_speed_test_file(self) -> Path:
        """results/m7_speed_test.csv: the trials of the speed test (same columns as v_phase.csv)."""
        return self.results_root / "m7_speed_test.csv"

    @property
    def m7_speed_test_metrics_file(self) -> Path:
        """results/m7_speed_test_metrics.csv: the speed test's metric rows."""
        return self.results_root / "m7_speed_test_metrics.csv"

    @property
    def t_phase_results_file(self) -> Path:
        """results/t_phase.csv: one row per trial of the final evaluation (one T fold)."""
        return self.results_root / "t_phase.csv"

    @property
    def t_phase_metrics_file(self) -> Path:
        """results/t_phase_metrics.csv: one row per (experiment, T fold): the metric values."""
        return self.results_root / "t_phase_metrics.csv"

    @property
    def t_phase_recall_file(self) -> Path:
        """results/t_phase_recall.csv: recall monitoring of the T folds (channels and union)."""
        return self.results_root / "t_phase_recall.csv"

    @property
    def t_phase_goals_file(self) -> Path:
        """results/t_phase_goals.csv: the verdict on G1-G3, one row per goal."""
        return self.results_root / "t_phase_goals.csv"

    @property
    def t_phase_label_access_file(self) -> Path:
        """results/t_phase_label_access.csv: when the labels of a T fold were first read."""
        return self.results_root / "t_phase_label_access.csv"

    @property
    def m8_rehearsal_results_file(self) -> Path:
        """results/m8_rehearsal.csv: the rehearsal's trial rows (same columns as v_phase.csv)."""
        return self.results_root / "m8_rehearsal.csv"

    @property
    def m8_rehearsal_metrics_file(self) -> Path:
        """results/m8_rehearsal_metrics.csv: the rehearsal's metric rows."""
        return self.results_root / "m8_rehearsal_metrics.csv"

    @property
    def m8_rehearsal_recall_file(self) -> Path:
        """results/m8_rehearsal_recall.csv: the rehearsal's recall monitoring rows."""
        return self.results_root / "m8_rehearsal_recall.csv"

    @property
    def m8_rehearsal_check_file(self) -> Path:
        """results/m8_rehearsal_check.csv: one row per comparison of a rehearsal with M7."""
        return self.results_root / "m8_rehearsal_check.csv"

    @property
    def m8_runs_file(self) -> Path:
        """results/m8_runs.csv: one row per M8 helper process (sampling, Dataset, monitor, ...)."""
        return self.results_root / "m8_runs.csv"

    @property
    def t_diag_label_access_file(self) -> Path:
        """results/t_diag_label_access.csv: every later read of T labels for diagnostics."""
        return self.results_root / "t_diag_label_access.csv"

    @property
    def t_diag_decomposition_file(self) -> Path:
        """results/t_diag_decomposition.csv: union recall, Recall@12, their ratio and MAP@12."""
        return self.results_root / "t_diag_decomposition.csv"

    @property
    def t_diag_concentration_file(self) -> Path:
        """results/t_diag_concentration.csv: how concentrated each label week's purchases are."""
        return self.results_root / "t_diag_concentration.csv"

    @property
    def t_diag_query_groups_file(self) -> Path:
        """results/t_diag_query_groups.csv: query groups by purchase history and their scores."""
        return self.results_root / "t_diag_query_groups.csv"

    @property
    def t_diag_bootstrap_file(self) -> Path:
        """results/t_diag_bootstrap.csv: bootstrap confidence intervals of the pooled MAP@12."""
        return self.results_root / "t_diag_bootstrap.csv"

    # ------------------------------------------------------------ DeepFM (stage 2, M9)

    @property
    def deepfm_config_file(self) -> Path:
        """configs/deepfm.yaml: the DeepFM model, its training and the speed test."""
        return self.configs_dir / "deepfm.yaml"

    @property
    def deepfm_study_config_file(self) -> Path:
        """configs/hpo/deepfm.yaml: the rules of the DeepFM study."""
        return self.configs_dir / "hpo" / "deepfm.yaml"

    @property
    def deepfm_extension_config_file(self) -> Path:
        """configs/hpo/deepfm_extension.yaml: the runs of the input-representation extension."""
        return self.configs_dir / "hpo" / "deepfm_extension.yaml"

    @property
    def deepfm_dir(self) -> Path:
        """artifacts/deepfm/: every file of the DeepFM study."""
        return self.artifacts_root / "deepfm"

    def deepfm_trial_dir(self, area: str, config_id: str, seed: int, fold: str) -> Path:
        """The files of one training: artifacts/deepfm/{area}/{config_id}/seed{seed}/{fold}/."""
        return self.deepfm_dir / area / config_id / f"seed{seed}" / fold

    def deepfm_cache_dir(self, target_fold: str, numeric_input: str, key: str) -> Path:
        """artifacts/deepfm/cache/{target_fold}_{numeric_input}_{key}/: encoded inputs of a week."""
        return self.deepfm_dir / "cache" / f"{target_fold}_{numeric_input}_{key}"

    def deepfm_log_file(self, label: str) -> Path:
        """The log of one DeepFM process: artifacts/deepfm/logs/{label}.log."""
        return self.deepfm_dir / "logs" / f"{label}.log"

    @property
    def deepfm_benchmark_file(self) -> Path:
        """results/deepfm_benchmark.csv: the device benchmark, one row per device run."""
        return self.results_root / "deepfm_benchmark.csv"

    @property
    def deepfm_trials_file(self) -> Path:
        """results/deepfm_trials.csv: one row per training (time, true memory peak, device)."""
        return self.results_root / "deepfm_trials.csv"

    @property
    def deepfm_study_file(self) -> Path:
        """results/deepfm_study.csv: what the study's rules decided, one row per configuration."""
        return self.results_root / "deepfm_study.csv"

    @property
    def m9_arms_file(self) -> Path:
        """results/m9_arms.csv: the four arms of the M9 comparison, per seed and summarised."""
        return self.results_root / "m9_arms.csv"

    @property
    def m9_contrasts_file(self) -> Path:
        """results/m9_contrasts.csv: bootstrap intervals of the differences between arms."""
        return self.results_root / "m9_contrasts.csv"

    @property
    def m9_groups_file(self) -> Path:
        """results/m9_groups.csv: MAP@12 of the four arms in the three query groups."""
        return self.results_root / "m9_groups.csv"

    @property
    def m9_ext_arms_file(self) -> Path:
        """results/m9_ext_arms.csv: the eight arms of the input-representation extension."""
        return self.results_root / "m9_ext_arms.csv"

    @property
    def m9_ext_contrasts_file(self) -> Path:
        """results/m9_ext_contrasts.csv: bootstrap intervals of the extension's contrasts."""
        return self.results_root / "m9_ext_contrasts.csv"

    @property
    def m9_ext_groups_file(self) -> Path:
        """results/m9_ext_groups.csv: MAP@12 of the eight arms in the three query groups."""
        return self.results_root / "m9_ext_groups.csv"

    @property
    def m9_lgb_trials_file(self) -> Path:
        """results/m9_lgb_trials.csv: one row per training of the two LightGBM arms of M9."""
        return self.results_root / "m9_lgb_trials.csv"

    @property
    def m9_lgb_metrics_file(self) -> Path:
        """results/m9_lgb_metrics.csv: one row per (arm, seed, fold, number of trees)."""
        return self.results_root / "m9_lgb_metrics.csv"

    def m9_lgb_predictions_file(self, arm: str, seed: int, fold: str) -> Path:
        """artifacts/predictions/m9_lgb/{arm}_s{seed}/{fold}.parquet: Top-K rows of an arm."""
        return self.predictions_file(f"m9_lgb/{arm}_s{seed}", fold)

    def m9_lgb_per_query_file(self, arm: str, seed: int, fold: str) -> Path:
        """artifacts/predictions/m9_lgb/{arm}_s{seed}/{fold}_per_query.parquet."""
        predictions = self.m9_lgb_predictions_file(arm, seed, fold)
        return predictions.with_name(f"{fold}_per_query.parquet")

    def m9_lgb_model_file(self, arm: str, seed: int, fold: str) -> Path:
        """artifacts/models/ranker/m9_lgb/{arm}_s{seed}/{fold}/model.txt."""
        return self.model_dir("ranker", "m9_lgb") / f"{arm}_s{seed}" / fold / "model.txt"

    @property
    def deepfm_metrics_file(self) -> Path:
        """results/deepfm_metrics.csv: one row per (configuration, seed, fold, epoch)."""
        return self.results_root / "deepfm_metrics.csv"


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
