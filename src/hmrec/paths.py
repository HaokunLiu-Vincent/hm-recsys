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

from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

from hmrec.config import BaseConfig, load_base_config

# src/hmrec/paths.py -> parents[0] = src/hmrec, parents[1] = src, parents[2] = repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"
BASE_CONFIG_FILE = CONFIGS_DIR / "base.yaml"
# The channel whose candidates calibration item E reads from another tree than R1-R5, and
# the tree of a calibration run that holds its channels (see EndToEndInputPaths).
R6_CHANNEL_NAME = "r6_twotower"
CALIBRATION_MAIN_TREE = "main"

# The step folders of a week in the state of the weekly pipeline (StateLayout, M12).
INGEST_DIR = "ingest"
QUERIES_DIR = "queries"
RECALL_DIR_PREFIX = "recall_"
FEATURES_DIR = "features"
FEATURES_SHARED_DIR = "features_shared"
SHARDS_DIR = "shards"
SHARD_TOP_DIR = "top"
DATASET_DIR = "ranker_dataset"
MODEL_DIR = "ranker_model"
SCORES_DIR = "scores"
EVALUATION_DIR = "evaluation"
SAMPLE_DIR = "sample"
TWOTOWER_DIR = "twotower"
STAGING_DIR = ".staging"
FAILED_DIR = ".failed"
# The files of a prediction set (predictions/{anchor}/, written by step O5) and the folder
# of its serving form inside it (hmrec.serving_set, M14).
PREDICTION_FILE = "top.parquet"
PREDICTION_MANIFEST_FILE = "manifest.json"
SERVING_DIR = "serving"
R1_CHANNEL_NAME = "r1_popularity"
R5_CHANNEL_NAME = "r5_wals"

# Where a tick of the weekly pipeline reads its inputs and keeps its state when they do
# not lie inside the repository (M13: two folders of a mounted bucket). Code names no
# bucket: the two folders come from the environment of the process (TickRootPaths).
PIPELINE_INPUTS_ROOT_VARIABLE = "HMREC_PIPELINE_INPUTS_ROOT"
PIPELINE_STATE_ROOT_VARIABLE = "HMREC_PIPELINE_STATE_ROOT"
# The state of a calibration tick lies NEXT TO the state, in a folder with this suffix.
CALIBRATION_STATE_SUFFIX = "-calibration"
# The folder of the pipeline's result tables inside a state that is not in the repository.
STATE_TABLES_DIR = "results"


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

    # ------------------------------------------------------------ M9: the evaluation on T1-T4

    @property
    def t_eval_config_file(self) -> Path:
        """configs/m9_t_eval.yaml: the one configuration and the rules of the evaluation on T."""
        return self.configs_dir / "m9_t_eval.yaml"

    @property
    def m9_t_trials_file(self) -> Path:
        """results/m9_t_trials.csv: one row per training of the evaluation on T1-T4."""
        return self.results_root / "m9_t_trials.csv"

    @property
    def m9_t_metrics_file(self) -> Path:
        """results/m9_t_metrics.csv: the metric values of every training of that evaluation."""
        return self.results_root / "m9_t_metrics.csv"

    @property
    def m9_t_rehearsal_check_file(self) -> Path:
        """results/m9_t_rehearsal_check.csv: one row per comparison of a rehearsal with V4."""
        return self.results_root / "m9_t_rehearsal_check.csv"

    @property
    def m9_t_label_access_file(self) -> Path:
        """results/m9_t_label_access.csv: every read of T labels by the M9 evaluation."""
        return self.results_root / "m9_t_label_access.csv"

    @property
    def m9_t_arms_file(self) -> Path:
        """results/m9_t_arms.csv: DeepFM (per seed, summarised) and the M8 record on T1-T4."""
        return self.results_root / "m9_t_arms.csv"

    @property
    def m9_t_contrasts_file(self) -> Path:
        """results/m9_t_contrasts.csv: the bootstrap interval of DeepFM minus the M8 record."""
        return self.results_root / "m9_t_contrasts.csv"

    @property
    def m9_t_v_and_t_file(self) -> Path:
        """results/m9_t_v_and_t.csv: pooled MAP@12 of the two models on V1-V4 and on T1-T4."""
        return self.results_root / "m9_t_v_and_t.csv"

    # ------------------------------------------------------------ stage 3: platform records

    @property
    def platforms_dir(self) -> Path:
        """configs/platforms/: one record per accepted platform (append-only, like locks)."""
        return self.configs_dir / "platforms"

    def platform_record_file(self, platform: str) -> Path:
        """configs/platforms/{platform}.yaml, e.g. "linux-arm64" (hmrec.platform_records)."""
        return self.platforms_dir / f"{platform}.yaml"

    # ------------------------------------------------------------ stage 3: M10 calibration
    # docs/mlops.md §8.3, docs/m10_calibration_plan.md. Everything a calibration writes
    # lies under these two folders; no stage-1 artifact or table is written.

    @property
    def m10_calibration_config_file(self) -> Path:
        """configs/m10_calibration.yaml: the week, input hashes and the provisional tolerance."""
        return self.configs_dir / "m10_calibration.yaml"

    @property
    def r6_repeatability_config_file(self) -> Path:
        """configs/m10_r6_repeatability.yaml: the synthetic R6 fine-tuning reproducer."""
        return self.configs_dir / "m10_r6_repeatability.yaml"

    @property
    def r6_repeatability_file(self) -> Path:
        """results/m10_calibration/r6_repeatability.csv: one row per pair of reproducer runs."""
        return self.calibration_results_dir / "r6_repeatability.csv"

    @property
    def calibration_artifacts_dir(self) -> Path:
        """artifacts/m10_calibration/: every file a calibration generates (not in git)."""
        return self.artifacts_root / "m10_calibration"

    @property
    def calibration_results_dir(self) -> Path:
        """results/m10_calibration/: the calibration's own result tables (committed)."""
        return self.results_root / "m10_calibration"

    def calibration_tree_dir(self, platform: str, run: str, tree: str) -> Path:
        """artifacts/m10_calibration/{platform}/{run}/{tree}/: one artifacts tree of a run.

        Below it the layout is the one of artifacts/ (candidates/, models/, ...), so the
        stage-1 functions can write into it through a second ProjectPaths
        (calibration_output_paths).

        Args:
            platform: e.g. "linux-arm64" (hmrec.platform_fields.PlatformFields.name).
            run: "run1" or "run2" (the second run checks repeatability).
            tree: "main" (R1-R5, the R6 chain, the union, the Ranker) or
                "recorded_model" (R6 candidates from the recorded model, item C1).
        """
        return self.calibration_artifacts_dir / platform / run / tree

    @property
    def calibration_comparisons_file(self) -> Path:
        """results/m10_calibration/comparisons.csv: one row per measured quantity."""
        return self.calibration_results_dir / "comparisons.csv"

    @property
    def calibration_runs_file(self) -> Path:
        """results/m10_calibration/runs.csv: one row per calibration process."""
        return self.calibration_results_dir / "runs.csv"

    @property
    def calibration_repeatability_file(self) -> Path:
        """results/m10_calibration/repeatability.csv: two container runs compared bit by bit."""
        return self.calibration_results_dir / "repeatability.csv"

    def calibration_chain_file(self, platform: str, run: str) -> Path:
        """results/m10_calibration/twotower_chain_{platform}_{run}.csv: the R6 fine-tune of a run.

        The same columns as results/m6_twotower_chain.csv, which is never written
        by a calibration.
        """
        return self.calibration_results_dir / f"twotower_chain_{platform}_{run}.csv"

    def calibration_log_file(self, platform: str, run: str, label: str) -> Path:
        """artifacts/m10_calibration/{platform}/{run}/logs/{label}.log: the log of one process."""
        return self.calibration_artifacts_dir / platform / run / "logs" / f"{label}.log"

    # ------------------------------------------------------------ stage 3: M11 versioning
    # docs/mlops.md §6.3 and §7. The data themselves are tracked with DVC; git holds the
    # pointer files (*.dvc) and the data manifest.

    @property
    def versioning_config_file(self) -> Path:
        """configs/m11_versioning.yaml: the bootstrap state and the local MLflow view."""
        return self.configs_dir / "m11_versioning.yaml"

    @property
    def data_manifest_file(self) -> Path:
        """configs/data_manifest.json: path, size and sha256 of every DVC-tracked file (in git)."""
        return self.configs_dir / "data_manifest.json"

    @property
    def bootstrap_dir(self) -> Path:
        """data/bootstrap/: the state at the end of F4 that the replay starts from.

        Below it the layout is the one of artifacts/ (models/, twotower/, ranker/),
        plus monitoring/ for the history rows of the result tables.
        """
        return self.data_root / "bootstrap"

    @property
    def bootstrap_manifest_file(self) -> Path:
        """data/bootstrap/manifest.json: every file of the state with its stage-1 source."""
        return self.bootstrap_dir / "manifest.json"

    @property
    def pipeline_runs_dir(self) -> Path:
        """artifacts/pipeline/runs/: one folder per tick (the runs/ prefix of mlops §7.2)."""
        return self.artifacts_root / "pipeline" / "runs"

    def run_manifest_file(self, run_id: str) -> Path:
        """artifacts/pipeline/runs/{run_id}/manifest.json: the record of one tick."""
        return self.pipeline_runs_dir / run_id / "manifest.json"

    def bootstrap_monitoring_file(self, table: str) -> Path:
        """data/bootstrap/monitoring/{table}: the F1-F4 rows of a stage-1 result table."""
        return self.bootstrap_dir / "monitoring" / table

    # ------------------------------------------------------------ stage 3: M12 weekly pipeline
    # docs/mlops.md §6 and §7.2. The state of the pipeline lies under artifacts/pipeline/
    # (StateLayout); its result tables under results/m12_replay/.

    @property
    def pipeline_config_file(self) -> Path:
        """configs/m12_pipeline.yaml: shard size, metrics, output signals, the replay's ticks."""
        return self.configs_dir / "m12_pipeline.yaml"

    @property
    def monitoring_config_file(self) -> Path:
        """configs/m12_monitoring.yaml: the monitoring rules, fixed before the first replay tick."""
        return self.configs_dir / "m12_monitoring.yaml"

    @property
    def pipeline_state_dir(self) -> Path:
        """artifacts/pipeline/: the local state of the weekly pipeline (layout of mlops §7.2)."""
        return self.artifacts_root / "pipeline"

    @property
    def replay_results_dir(self) -> Path:
        """results/m12_replay/: the result tables of the weekly pipeline and its replay."""
        return self.results_root / "m12_replay"

    def replay_table_file(self, table: str) -> Path:
        """results/m12_replay/{table}.csv, e.g. "monitoring" or "comparison"."""
        return self.replay_results_dir / f"{table}.csv"

    @property
    def kaggle_reference_dir(self) -> Path:
        """artifacts/kaggle/submission/: the saved prediction set of the Kaggle late submission.

        Read only by the `serve` tick's diagnostic (docs/mlops.md §6.3); the files
        were written on branch kaggle-late-submission and are never written here.
        """
        return self.artifacts_root / "kaggle" / "submission"

    # ------------------------------------------------------------ stage 3: M13 cloud

    @property
    def cloud_config_file(self) -> Path:
        """configs/m13_cloud.yaml: the shape of the Cloud Run job and the prices of its cost."""
        return self.configs_dir / "m13_cloud.yaml"

    @property
    def cloud_results_dir(self) -> Path:
        """results/m13_cloud/: the records of the replay on Cloud Run.

        Copies of the tables the pipeline wrote into its state in the bucket, the
        comparison with the stage-1 records, and one row per execution of the job.
        """
        return self.results_root / "m13_cloud"

    def cloud_table_file(self, table: str) -> Path:
        """results/m13_cloud/{table}.csv, e.g. "executions"."""
        return self.cloud_results_dir / f"{table}.csv"

    @property
    def serving_results_dir(self) -> Path:
        """results/m14_serving/: the records of M14 (docs/mlops.md §9, §10).

        What was measured of the recommendation service with synthetic requests,
        the rollback drill, the provoked alerts, and one row per execution of a
        Cloud Run job of the milestone.
        """
        return self.results_root / "m14_serving"

    def serving_table_file(self, table: str) -> Path:
        """results/m14_serving/{table}.csv, e.g. "probe_summary" or "executions"."""
        return self.serving_results_dir / f"{table}.csv"

    @property
    def cloud_state_records_dir(self) -> Path:
        """artifacts/m13_cloud/pipeline/: the SMALL records of the state in the bucket.

        A copy made by `bash infra/operate.sh download-records`: run manifests,
        step records, markers, the tables. It has the layout of the state
        (StateLayout) without the large files, which stay in the bucket.
        """
        return self.artifacts_root / "m13_cloud" / "pipeline"

    @property
    def cloud_calibration_records_dir(self) -> Path:
        """artifacts/m13_cloud/pipeline-calibration/: the same copy of the calibration state."""
        return calibration_state_dir(self.cloud_state_records_dir)

    @property
    def mlflow_dir(self) -> Path:
        """artifacts/mlflow/: the local MLflow store (not in git; rebuilt by `make mlflow-sync`)."""
        return self.artifacts_root / "mlflow"

    @property
    def mlflow_store_file(self) -> Path:
        """artifacts/mlflow/mlflow.db: the SQLite backend of the local MLflow view."""
        return self.mlflow_dir / "mlflow.db"

    @property
    def mlflow_artifacts_dir(self) -> Path:
        """artifacts/mlflow/artifacts/: where the local MLflow store would keep run files."""
        return self.mlflow_dir / "artifacts"


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


def calibration_output_paths(
    paths: ProjectPaths, platform: str, run: str, tree: str
) -> ProjectPaths:
    """A second ProjectPaths whose artifacts and results lie in the calibration folders.

    The stage-1 functions write where a ProjectPaths tells them. Given this one,
    they write a candidates file to
    artifacts/m10_calibration/{platform}/{run}/{tree}/candidates/{fold}/... instead
    of artifacts/candidates/{fold}/..., and a result table under
    results/m10_calibration/. Inputs (queries, models, features) are still read
    through the first ProjectPaths.

    Args:
        paths: The project paths (the stage-1 locations).
        platform: e.g. "linux-arm64".
        run: "run1" or "run2".
        tree: "main" or "recorded_model" (ProjectPaths.calibration_tree_dir).

    Returns:
        The same project root and data root; artifacts_root and results_root replaced.
    """
    return ProjectPaths(
        project_root=paths.project_root,
        data_root=paths.data_root,
        artifacts_root=paths.calibration_tree_dir(platform, run, tree),
        results_root=paths.calibration_results_dir,
    )


@dataclass(frozen=True)
class EndToEndInputPaths(ProjectPaths):
    """Where calibration item E reads the six channels of a week from (docs/mlops.md §8.3).

    Item E builds the features of a week from the container's OWN candidates. They
    come from two runs: R1-R5 (with R1's bucket score table and R5's vectors) from
    the run that generated them, and R6 (with its vectors) from the run whose R6
    chain is used. This class is a ProjectPaths that answers each of those paths
    from the right tree, so that the locked feature code
    (hmrec.features.build.load_candidates, load_dense_sources) reads them as it
    reads a stage-1 tree. Processed tables and embeddings come from data_root as always.

    Attributes:
        artifacts_root: The tree that holds R6 (its candidates and vectors).
        statistical_artifacts_root: The tree that holds R1-R5.
    """

    statistical_artifacts_root: Path

    @property
    def statistical_paths(self) -> ProjectPaths:
        """A ProjectPaths whose artifacts root is the tree of R1-R5."""
        return ProjectPaths(
            project_root=self.project_root,
            data_root=self.data_root,
            artifacts_root=self.statistical_artifacts_root,
            results_root=self.results_root,
        )

    def candidates_file(self, fold: str, channel: str) -> Path:
        """R6 from artifacts_root; every other channel from the tree of R1-R5."""
        if channel == R6_CHANNEL_NAME:
            return super().candidates_file(fold, channel)
        return self.statistical_paths.candidates_file(fold, channel)

    def r1_bucket_scores_file(self, fold: str) -> Path:
        """R1's bucket score table, from the tree of R1-R5."""
        return self.statistical_paths.r1_bucket_scores_file(fold)

    def wals_vectors_dir(self, fold: str) -> Path:
        """R5's vectors, from the tree of R1-R5."""
        return self.statistical_paths.wals_vectors_dir(fold)


def end_to_end_input_paths(
    paths: ProjectPaths, platform: str, r6_run: str, statistical_run: str
) -> EndToEndInputPaths:
    """The input paths of calibration item E.

    Args:
        paths: The project paths (the stage-1 locations; data_root is kept).
        platform: e.g. "linux-arm64".
        r6_run: The calibration run whose R6 candidates and vectors are used.
        statistical_run: The calibration run whose R1-R5 outputs are used.
    """
    return EndToEndInputPaths(
        project_root=paths.project_root,
        data_root=paths.data_root,
        artifacts_root=paths.calibration_tree_dir(platform, r6_run, CALIBRATION_MAIN_TREE),
        results_root=paths.calibration_results_dir,
        statistical_artifacts_root=paths.calibration_tree_dir(
            platform, statistical_run, CALIBRATION_MAIN_TREE
        ),
    )


# ---------------------------------------------------------------- stage 3: M12 weekly pipeline


@dataclass(frozen=True)
class StateLayout:
    """Where everything of the weekly pipeline's state lies (docs/mlops.md §7.2).

        {root}/state/current.json           pointer to the current prediction set
        {root}/weeks/{anchor}/{step}/       the outputs of one step for one week
        {root}/weeks/{anchor}/.staging/     a step's outputs until its checks passed
        {root}/weeks/{anchor}/.failed/      staging folders of failed attempts (kept)
        {root}/predictions/{anchor}/        one prediction set
        {root}/runs/{run_id}/               manifest.json, logs/, failures/

    A week is named by its anchor (ISO date). On the laptop the root is
    artifacts/pipeline/; in the cloud the same keys lie in a bucket (M13).

    Attributes:
        root: The folder that holds the state.
    """

    root: Path

    @property
    def current_file(self) -> Path:
        """state/current.json: replaced only by step O5."""
        return self.root / "state" / "current.json"

    def week_dir(self, anchor: date) -> Path:
        """weeks/{anchor}/: everything of the week that starts at `anchor`."""
        return self.root / "weeks" / anchor.isoformat()

    def step_dir(self, anchor: date, step_dir: str) -> Path:
        """weeks/{anchor}/{step_dir}/: the finished outputs of one step."""
        return self.week_dir(anchor) / step_dir

    def staging_dir(self, anchor: date, step_dir: str) -> Path:
        """weeks/{anchor}/.staging/{step_dir}/: where a running step writes."""
        return self.week_dir(anchor) / STAGING_DIR / step_dir

    def failed_dir(self, anchor: date) -> Path:
        """weeks/{anchor}/.failed/: staging folders of failed attempts, moved here."""
        return self.week_dir(anchor) / FAILED_DIR

    def published_file(self, anchor: date) -> Path:
        """weeks/{anchor}/published.json: written by O5; the week's predictions are out."""
        return self.week_dir(anchor) / "published.json"

    def closed_file(self, anchor: date) -> Path:
        """weeks/{anchor}/closed.json: written when C2-C4 of the week are done."""
        return self.week_dir(anchor) / "closed.json"

    def monitoring_file(self, anchor: date) -> Path:
        """weeks/{anchor}/monitoring.json: the findings of the tick at `anchor`."""
        return self.week_dir(anchor) / "monitoring.json"

    def prediction_set_dir(self, anchor: date) -> Path:
        """predictions/{anchor}/: the prediction set of the week (Top-K, manifest)."""
        return self.root / "predictions" / anchor.isoformat()

    def prediction_staging_dir(self, anchor: date) -> Path:
        """predictions/.staging/{anchor}/: the prediction set until it is complete."""
        return self.root / "predictions" / STAGING_DIR / anchor.isoformat()

    def prediction_file(self, anchor: date) -> Path:
        """predictions/{anchor}/top.parquet: customer_idx, rank, article_idx."""
        return self.prediction_set_dir(anchor) / PREDICTION_FILE

    def prediction_manifest_file(self, anchor: date) -> Path:
        """predictions/{anchor}/manifest.json: the record of the prediction set."""
        return self.prediction_set_dir(anchor) / PREDICTION_MANIFEST_FILE

    def serving_dir(self, anchor: date) -> Path:
        """predictions/{anchor}/serving/: the form the service loads (hmrec.serving_set)."""
        return self.prediction_set_dir(anchor) / SERVING_DIR

    @property
    def predictions_dir(self) -> Path:
        """predictions/: one folder per prediction set, named by its anchor."""
        return self.root / "predictions"

    @property
    def runs_dir(self) -> Path:
        """runs/: one folder per run of a tick."""
        return self.root / "runs"

    def run_manifest_file(self, run_id: str) -> Path:
        """runs/{run_id}/manifest.json: the record of one run of a tick."""
        return self.runs_dir / run_id / "manifest.json"

    def step_log_file(self, run_id: str, step: str) -> Path:
        """runs/{run_id}/logs/{step}.log: the log of one step's process."""
        return self.runs_dir / run_id / "logs" / f"{step}.log"

    def step_failure_file(self, run_id: str, step: str) -> Path:
        """runs/{run_id}/failures/{step}.txt: why a step's process stopped."""
        return self.runs_dir / run_id / "failures" / f"{step}.txt"


def shard_name(shard: int) -> str:
    """The folder name of a shard, e.g. 7 -> "007"."""
    if shard < 0:
        raise ValueError(f"a shard number must be >= 0, got {shard}")
    return f"{shard:03d}"


def recall_step_dir(channel: str) -> str:
    """The step folder of one recall channel, e.g. "recall_r4_itemcf"."""
    return f"{RECALL_DIR_PREFIX}{channel}"


def shard_step_dir(shard: int, part: str) -> str:
    """The step folder of one part of a shard, e.g. (7, "features") -> "shards/007/features"."""
    return f"{SHARDS_DIR}/{shard_name(shard)}/{part}"


@dataclass(frozen=True)
class PipelinePaths(ProjectPaths):
    """The ProjectPaths one step of a tick of the weekly pipeline works with (M12).

    The stage-1 functions read and write where a ProjectPaths tells them. Given
    this one they work inside the pipeline's state, unchanged:

        artifacts_root     the READ-ONLY bootstrap state (data/bootstrap/). Its
                           layout is the one of artifacts/, so the locked base
                           model, the chain model after F4, the alarm users and the
                           samples of F1-F4 are found where the stage-1 code looks.
        week-keyed paths   a week the pipeline opened (week_anchors) is answered
                           from the state: weeks/{anchor}/{step folder}/...
                           A fold before the first tick keeps the bootstrap path.
        transactions_file  the table step C1 of THIS tick wrote: every purchase
                           line before the tick's anchor and nothing later. This
                           is how the simulated clock is enforced by construction
                           (docs/mlops.md §6.3): a step cannot read a later row,
                           because its input does not hold one.
        staging            the outputs of the step that is running go to
                           weeks/{anchor}/.staging/{step folder}/; the driver
                           renames the folder when the step's checks passed.
        shard              in `serve` mode a week's candidates, vectors and
                           features are cut into shards of customers; with a shard
                           number the same methods answer that shard's files.

    Attributes:
        layout: The state's layout.
        tick_anchor: The anchor of the tick this process belongs to.
        week_anchors: (week name, anchor) of every week the pipeline has opened
            up to this tick, e.g. (("V1", 2020-07-29), ...).
        category_codes: The category code table shared by all weeks
            (artifacts/features/category_codes.json, DVC-tracked; only compared).
        staging: (anchor of the week, step folder) of the running step, or None.
        shard: The shard the paths are answered for, or None (a whole week).
    """

    layout: StateLayout
    tick_anchor: date
    week_anchors: tuple[tuple[str, date], ...]
    category_codes: Path
    staging: tuple[date, str] | None
    shard: int | None

    def for_shard(self, shard: int) -> "PipelinePaths":
        """The same paths, answered for one shard of customers."""
        return replace(self, shard=shard)

    def anchor_of(self, week: str) -> date | None:
        """The anchor of a week the pipeline opened; None for a fold of the bootstrap state."""
        for name, anchor in self.week_anchors:
            if name == week:
                return anchor
        return None

    def require_anchor(self, week: str) -> date:
        """The anchor of a pipeline week; fails for a fold the pipeline never opened."""
        anchor = self.anchor_of(week)
        if anchor is None:
            known = [name for name, _ in self.week_anchors]
            raise ValueError(f"{week!r} is not a week of the pipeline; its weeks are {known}")
        return anchor

    def step_dir(self, anchor: date, step_dir: str) -> Path:
        """The folder of a step: the staging folder for the running step, else the finished one."""
        if self.staging == (anchor, step_dir):
            return self.layout.staging_dir(anchor, step_dir)
        return self.layout.step_dir(anchor, step_dir)

    def week_step_dir(self, week: str, step_dir: str) -> Path:
        """step_dir for a week given by name."""
        return self.step_dir(self.require_anchor(week), step_dir)

    def _recall_dir(self, week: str, channel: str) -> Path:
        """The folder of a channel's candidates and vectors (of the shard, if one is set)."""
        folder = self.week_step_dir(week, recall_step_dir(channel))
        if self.shard is None:
            return folder
        return folder / SHARDS_DIR / shard_name(self.shard)

    # ------------------------------------------------------------ inputs of a tick

    @property
    def transactions_file(self) -> Path:
        """The table C1 of this tick wrote: purchase lines with t_dat before the tick's anchor."""
        return self.step_dir(self.tick_anchor, INGEST_DIR) / "transactions.parquet"

    @property
    def ingest_record_file(self) -> Path:
        """weeks/{tick}/ingest/ingest.json: row counts per week and the checked date range."""
        return self.step_dir(self.tick_anchor, INGEST_DIR) / "ingest.json"

    @property
    def category_codes_file(self) -> Path:
        """The category code table shared by all weeks (compared, never written here)."""
        return self.category_codes

    # ------------------------------------------------------------ week-keyed stage-1 paths

    def queries_file(self, fold: str) -> Path:
        """weeks/{anchor}/queries/queries.parquet."""
        return self.week_step_dir(fold, QUERIES_DIR) / "queries.parquet"

    def labels_file(self, fold: str) -> Path:
        """weeks/{anchor}/evaluation/labels.parquet: written by C2 when the week is closed."""
        return self.week_step_dir(fold, EVALUATION_DIR) / "labels.parquet"

    def candidates_file(self, fold: str, channel: str) -> Path:
        """weeks/{anchor}/recall_{channel}/[shards/{k}/]candidates.parquet."""
        return self._recall_dir(fold, channel) / "candidates.parquet"

    def r1_bucket_scores_file(self, fold: str) -> Path:
        """R1's S(g, i) table: one per week, also in `serve` mode (the same for every shard)."""
        folder = self.week_step_dir(fold, recall_step_dir(R1_CHANNEL_NAME))
        return folder / "r1_bucket_scores.parquet"

    def wals_vectors_dir(self, fold: str) -> Path:
        """weeks/{anchor}/recall_r5_wals/[shards/{k}/]vectors/."""
        return self._recall_dir(fold, R5_CHANNEL_NAME) / "vectors"

    def twotower_vectors_dir(self, fold: str) -> Path:
        """weeks/{anchor}/recall_r6_twotower/[shards/{k}/]vectors/."""
        return self._recall_dir(fold, R6_CHANNEL_NAME) / "vectors"

    def features_dir(self, fold: str) -> Path:
        """weeks/{anchor}/features/, or weeks/{anchor}/shards/{k}/features/ for a shard."""
        if self.shard is None:
            return self.week_step_dir(fold, FEATURES_DIR)
        return self.week_step_dir(fold, shard_step_dir(self.shard, FEATURES_DIR))

    def twotower_locked_chain_checkpoint_file(self, fold: str) -> Path:
        """The chain model after the fine-tune on a week.

        A pipeline week: weeks/{anchor}/twotower/{fold}.pt (written by step C4 of the
        next tick). A fold before the first tick: the bootstrap state's file.
        """
        if self.anchor_of(fold) is None:
            return super().twotower_locked_chain_checkpoint_file(fold)
        return self.week_step_dir(fold, TWOTOWER_DIR) / f"{fold}.pt"

    def chain_record_file(self, fold: str) -> Path:
        """weeks/{anchor}/twotower/chain_record.csv: the row of the week's fine-tune."""
        return self.week_step_dir(fold, TWOTOWER_DIR) / "chain_record.csv"

    def ranker_sample_file(self, experiment: str, fold: str, ratio: int) -> Path:
        """The training sample of a week: the state's for a pipeline week, else the bootstrap's."""
        if self.anchor_of(fold) is None:
            return super().ranker_sample_file(experiment, fold, ratio)
        return self.week_step_dir(fold, SAMPLE_DIR) / f"r{ratio}.parquet"

    def ranker_dataset_dir(self, experiment: str, target_fold: str, ratio: int) -> Path:
        """weeks/{anchor}/ranker_dataset/: the Dataset of the model that predicts the week."""
        return self.week_step_dir(target_fold, DATASET_DIR)

    def ranker_model_file(self, area: str, experiment: str, config_id: str, fold: str) -> Path:
        """weeks/{anchor}/ranker_model/model.txt: the Ranker that predicts the week."""
        return self.week_step_dir(fold, MODEL_DIR) / "model.txt"

    # ------------------------------------------------------------ files only the pipeline has

    def feature_stats_file(self, fold: str) -> Path:
        """The per-feature statistics next to a week's (or a shard's) features."""
        return self.features_dir(fold) / "feature_stats.csv"

    def item_table_file(self, fold: str) -> Path:
        """`serve` mode: the article table of the week, computed once for all shards."""
        return self.week_step_dir(fold, FEATURES_SHARED_DIR) / "item_table.parquet"

    def type_share_file(self, fold: str) -> Path:
        """`serve` mode: the global product-type shares of the week, computed once."""
        return self.week_step_dir(fold, FEATURES_SHARED_DIR) / "global_type_share.parquet"

    def top_file(self, fold: str) -> Path:
        """The Top-K of a week (scores/top.parquet) or of one shard (shards/{k}/top/top.parquet)."""
        if self.shard is None:
            return self.week_step_dir(fold, SCORES_DIR) / "top.parquet"
        return self.week_step_dir(fold, shard_step_dir(self.shard, SHARD_TOP_DIR)) / "top.parquet"

    def signals_file(self, fold: str) -> Path:
        """weeks/{anchor}/scores/signals.json: the output signals of the week (mlops §10)."""
        return self.week_step_dir(fold, SCORES_DIR) / "signals.json"

    def evaluation_file(self, fold: str) -> Path:
        """weeks/{anchor}/evaluation/metrics.json: what C2 measured for the closed week."""
        return self.week_step_dir(fold, EVALUATION_DIR) / "metrics.json"

    def model_record_file(self, fold: str) -> Path:
        """weeks/{anchor}/ranker_model/model.json: sha256 and training facts of the Ranker."""
        return self.week_step_dir(fold, MODEL_DIR) / "model.json"


def pipeline_paths(
    paths: ProjectPaths,
    layout: StateLayout,
    tick_anchor: date,
    week_anchors: tuple[tuple[str, date], ...],
    staging: tuple[date, str] | None,
) -> PipelinePaths:
    """The paths of one step of a tick (see PipelinePaths).

    Args:
        paths: The project paths (data/processed/, data/bootstrap/, configs/).
        layout: The state's layout.
        tick_anchor: The anchor of the tick.
        week_anchors: (name, anchor) of every week the pipeline opened up to this tick.
        staging: (anchor of the week, step folder) of the running step, or None
            for a process that only reads the state.

    Returns:
        A PipelinePaths whose artifacts root is the bootstrap state and whose
        result tables go to results/m12_replay/.
    """
    return PipelinePaths(
        project_root=paths.project_root,
        data_root=paths.data_root,
        artifacts_root=paths.bootstrap_dir,
        results_root=paths.replay_results_dir,
        layout=layout,
        tick_anchor=tick_anchor,
        week_anchors=week_anchors,
        category_codes=paths.category_codes_file,
        staging=staging,
        shard=None,
    )


# ---------------------------------------------------------------- stage 3: M13 roots of a tick


@dataclass(frozen=True)
class TickRootPaths(ProjectPaths):
    """The ProjectPaths of a tick whose state is not artifacts/pipeline/ of the repository.

    Two cases (docs/mlops.md §6.4, §8.3):

        a state root from the environment   the state of the pipeline lies where
                                            HMREC_PIPELINE_STATE_ROOT says, e.g. a
                                            folder of a mounted bucket;
        a calibration tick                  everything is written under a SEPARATE
                                            state, next to the state and named
                                            "{state folder}-calibration".

    In both cases the pipeline's result tables lie INSIDE the state
    ({state_root}/results/): the repository's results/m12_replay/ holds the
    replay of M12 and is not written by another run.

    Attributes:
        state_root: The folder that holds this tick's state (already the
            calibration folder for a calibration tick).
        calibration: True for a calibration tick.
    """

    state_root: Path
    calibration: bool

    @property
    def pipeline_state_dir(self) -> Path:
        """The state of this tick: `state_root`."""
        return self.state_root

    @property
    def replay_results_dir(self) -> Path:
        """{state_root}/results/: the pipeline's result tables of this state."""
        return self.state_root / STATE_TABLES_DIR


def calibration_state_dir(state_dir: Path) -> Path:
    """The state folder of calibration ticks: next to `state_dir`, e.g. pipeline-calibration/."""
    return state_dir.with_name(state_dir.name + CALIBRATION_STATE_SUFFIX)


def absolute_root(variable: str, value: str) -> Path:
    """A root folder given in the environment.

    Raises:
        ValueError: If it is not an absolute path. A relative path would depend
            on the working directory of the process.
    """
    root = Path(value)
    if not root.is_absolute():
        raise ValueError(f"{variable} must be an absolute path, got {value!r}")
    return root


def tick_root_paths(
    paths: ProjectPaths, environment: Mapping[str, str], calibration: bool
) -> ProjectPaths:
    """The project paths of a tick, with the roots the environment gives (M13).

        HMREC_PIPELINE_INPUTS_ROOT   a folder that holds the pipeline's inputs in
                                     their plain layout: data/processed/,
                                     data/bootstrap/ and
                                     artifacts/features/category_codes.json.
        HMREC_PIPELINE_STATE_ROOT    the folder that holds the state (the layout
                                     of StateLayout) and the result tables.

    Neither is set on the laptop: the paths of configs/base.yaml are used, the
    state is artifacts/pipeline/ and the tables are results/m12_replay/, as in M12.

    Args:
        paths: The project paths of configs/base.yaml.
        environment: The environment of the process (os.environ).
        calibration: True for a calibration tick: the state is the folder next
            to the state, see calibration_state_dir.

    Returns:
        `paths` itself if no root is given and the tick is not a calibration
        tick; else a TickRootPaths.

    Raises:
        ValueError: If a root is not an absolute path, or an inputs root is
            given without a state root (the state would then be written among
            the inputs).
    """
    inputs_value = environment.get(PIPELINE_INPUTS_ROOT_VARIABLE, "")
    state_value = environment.get(PIPELINE_STATE_ROOT_VARIABLE, "")
    if inputs_value == "" and state_value == "" and not calibration:
        return paths
    if inputs_value != "" and state_value == "":
        raise ValueError(
            f"{PIPELINE_INPUTS_ROOT_VARIABLE} is set but {PIPELINE_STATE_ROOT_VARIABLE} is not: "
            "the state would be written among the inputs"
        )
    data_root, artifacts_root = paths.data_root, paths.artifacts_root
    if inputs_value != "":
        inputs_root = absolute_root(PIPELINE_INPUTS_ROOT_VARIABLE, inputs_value)
        data_root, artifacts_root = inputs_root / "data", inputs_root / "artifacts"
    state_root = paths.pipeline_state_dir
    if state_value != "":
        state_root = absolute_root(PIPELINE_STATE_ROOT_VARIABLE, state_value)
    if calibration:
        state_root = calibration_state_dir(state_root)
    return TickRootPaths(
        project_root=paths.project_root,
        data_root=data_root,
        artifacts_root=artifacts_root,
        results_root=paths.results_root,
        state_root=state_root,
        calibration=calibration,
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
