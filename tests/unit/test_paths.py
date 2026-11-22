from pathlib import Path

import pytest
import yaml

from hmrec.config import BaseConfig, ChunkingConfig, MemoryConfig, PathsConfig
from hmrec.paths import (
    PIPELINE_INPUTS_ROOT_VARIABLE,
    PIPELINE_STATE_ROOT_VARIABLE,
    PROJECT_ROOT,
    TickRootPaths,
    build_paths,
    calibration_state_dir,
    get_paths,
    resolve_root,
    tick_root_paths,
)


def make_config(data_root: str, artifacts_root: str, results_root: str = "results") -> BaseConfig:
    return BaseConfig(
        paths=PathsConfig(
            data_root=data_root, artifacts_root=artifacts_root, results_root=results_root
        ),
        seed=0,
        num_threads=1,
        memory=MemoryConfig(limit_gb=1.0, sample_interval_seconds=0.1),
        chunking=ChunkingConfig(
            query_chunk_size=1,
            item_chunk_size=1,
            dense_score_query_chunk_size=1,
            twotower_score_query_chunk_size=1,
            pair_chunk_size=1,
        ),
    )


def test_project_root_holds_configs_and_src() -> None:
    assert (PROJECT_ROOT / "configs" / "base.yaml").is_file()
    assert (PROJECT_ROOT / "src" / "hmrec" / "paths.py").is_file()


def test_resolve_root_relative_and_absolute(tmp_path: Path) -> None:
    assert resolve_root("data", tmp_path) == tmp_path / "data"
    absolute = tmp_path / "elsewhere"
    assert resolve_root(str(absolute), Path("/ignored")) == absolute


def test_build_paths_derives_everything_from_roots(tmp_path: Path) -> None:
    paths = build_paths(make_config("my_data", "/abs/artifacts"), tmp_path)

    assert paths.data_root == tmp_path / "my_data"
    assert paths.artifacts_root == Path("/abs/artifacts")
    assert paths.transactions_csv == tmp_path / "my_data" / "raw" / "transactions_train.csv"
    assert paths.image_embeddings_file == (
        tmp_path / "my_data" / "processed" / "image_features" / "image_embeddings.npy"
    )
    assert paths.text_article_ids_file == (
        tmp_path / "my_data" / "processed" / "text_features" / "text_article_ids.parquet"
    )
    # Proposal §14.2 artifact layout.
    assert paths.candidates_file("D1", "pop") == Path("/abs/artifacts/candidates/D1/pop.parquet")
    assert paths.features_dir("F2") == Path("/abs/artifacts/features/F2")
    assert paths.labels_file("V3") == Path("/abs/artifacts/labels/V3.parquet")
    assert paths.model_dir("wals", "T1") == Path("/abs/artifacts/models/wals/T1")
    assert paths.predictions_file("run7", "T4") == Path(
        "/abs/artifacts/predictions/run7/T4.parquet"
    )
    assert paths.d_phase_results_file == tmp_path / "results" / "d_phase.csv"
    # Ranker (M7).
    assert paths.ranker_config_file == tmp_path / "configs" / "ranker.yaml"
    assert paths.v_phase_grid_file == tmp_path / "configs" / "hpo" / "v_phase.yaml"
    assert paths.ranker_sample_file("full", "F1", 40) == Path(
        "/abs/artifacts/ranker/samples/full/F1/r40.parquet"
    )
    assert paths.ranker_sample_record_file("minus_r6", "V3", 60) == Path(
        "/abs/artifacts/ranker/samples/minus_r6/V3/r60.json"
    )
    assert paths.ranker_dataset_dir("full", "V1", 40) == Path(
        "/abs/artifacts/lgb_datasets/full/V1_r40"
    )
    assert paths.ranker_model_file("v_phase", "full", "abc", "V2") == Path(
        "/abs/artifacts/models/ranker/v_phase/full/abc/V2/model.txt"
    )
    assert paths.ranker_predictions_file("speed_test", "full", "abc", "V1") == Path(
        "/abs/artifacts/predictions/speed_test/full/abc/V1.parquet"
    )
    assert paths.ranker_per_query_file("speed_test", "full", "abc", "V1") == Path(
        "/abs/artifacts/predictions/speed_test/full/abc/V1_per_query.parquet"
    )
    for name in (
        "v_phase",
        "v_phase_metrics",
        "v_phase_steps",
        "v_phase_feature_importance",
        "m7_runs",
        "m7_speed_test",
        "m7_speed_test_metrics",
    ):
        attribute = "v_phase_results_file" if name == "v_phase" else f"{name}_file"
        assert getattr(paths, attribute) == tmp_path / "results" / f"{name}.csv"
    assert paths.union_precheck_candidates_file("D2", "r5_wals", "abc123") == Path(
        "/abs/artifacts/hpo/D/union_precheck/D2/r5_wals_abc123.parquet"
    )
    assert paths.union_precheck_results_file == (
        tmp_path / "results" / "d_phase_union_precheck.csv"
    )
    assert paths.twotower_data_dir("base") == Path("/abs/artifacts/twotower/data/base")
    assert paths.twotower_checkpoint_file(3) == Path(
        "/abs/artifacts/models/twotower/base/epoch_3.pt"
    )
    assert paths.twotower_chain_checkpoint_file("abc", "D4") == Path(
        "/abs/artifacts/models/twotower/chain/abc/D4.pt"
    )
    assert paths.m4_candidates_file("D3", "r6_twotower", "abc") == Path(
        "/abs/artifacts/m4/candidates/D3/r6_twotower_abc.parquet"
    )
    assert paths.m4_union_check_results_file == tmp_path / "results" / "m4_union_check.csv"
    assert paths.m4_union_check_variant_file("r3_active_pool") == (
        tmp_path / "results" / "m4_union_check_r3_active_pool.csv"
    )
    assert paths.m4_trials_file == tmp_path / "results" / "m4_trials.csv"
    # M6 features: one part file per query chunk, the fold marker, the shared code table.
    assert paths.features_config_file == tmp_path / "configs" / "features.yaml"
    assert paths.features_part_file("F1", 3) == Path(
        "/abs/artifacts/features/F1/part-00003.parquet"
    )
    assert paths.features_metadata_file("F1") == Path("/abs/artifacts/features/F1/features.json")
    assert paths.category_codes_file == Path("/abs/artifacts/features/category_codes.json")
    assert paths.m6_features_results_file == tmp_path / "results" / "m6_features.csv"
    # M6: candidates, the files the features read, the locked chain models, results.
    assert paths.m6_config_file == tmp_path / "configs" / "m6_candidates.yaml"
    assert paths.r1_bucket_scores_file("F1") == Path(
        "/abs/artifacts/candidates/F1/r1_bucket_scores.parquet"
    )
    assert paths.wals_vectors_dir("F1") == Path("/abs/artifacts/models/wals/F1")
    assert paths.twotower_vectors_dir("V2") == Path("/abs/artifacts/models/twotower/vectors/V2")
    assert paths.twotower_locked_chain_checkpoint_file("T3") == Path(
        "/abs/artifacts/models/twotower/chain_locked/T3.pt"
    )
    assert paths.m6_runs_file == tmp_path / "results" / "m6_runs.csv"
    assert paths.m6_candidates_results_file == tmp_path / "results" / "m6_candidates.csv"
    assert paths.m6_twotower_chain_results_file == tmp_path / "results" / "m6_twotower_chain.csv"
    assert paths.m5_lock_check_results_file == tmp_path / "results" / "m5_lock_check.csv"
    assert paths.channels_lock_file == tmp_path / "configs" / "locks" / "channels.lock.yaml"
    assert paths.twotower_lock_file == tmp_path / "configs" / "locks" / "twotower.lock.yaml"
    assert paths.features_lock_file == tmp_path / "configs" / "locks" / "features.lock.yaml"
    assert paths.ranker_lock_file == tmp_path / "configs" / "locks" / "ranker.lock.yaml"
    # The M5 lock check writes its candidates away from artifacts/candidates/.
    assert paths.lock_check_candidates_file("D1", "r4_itemcf") == Path(
        "/abs/artifacts/m5/lock_check/D1/r4_itemcf.parquet"
    )
    # A repeated check gets its own folder, so the first check's files stay untouched.
    assert paths.lock_check_candidates_file("D1", "r4_itemcf", "after_m6_refactor") == Path(
        "/abs/artifacts/m5/lock_check_after_m6_refactor/D1/r4_itemcf.parquet"
    )
    assert paths.d_phase_seeds_file("r6_twotower") == (
        tmp_path / "results" / "d_phase_steps" / "r6_twotower_seeds.csv"
    )


def test_m0_outputs_live_in_data_processed(tmp_path: Path) -> None:
    paths = build_paths(make_config("my_data", "artifacts"), tmp_path)
    processed = tmp_path / "my_data" / "processed"

    assert paths.preprocess_config_file == tmp_path / "configs" / "preprocess.yaml"
    assert paths.transactions_file == processed / "transactions.parquet"
    assert paths.customers_file == processed / "customers.parquet"
    assert paths.articles_file == processed / "articles.parquet"
    assert paths.customer_id_map_file == processed / "customer_id_map.parquet"
    assert paths.article_id_map_file == processed / "article_id_map.parquet"
    assert paths.processed_metadata_file == processed / "metadata.json"
    assert (
        paths.image_embeddings_by_idx_file == processed / "embeddings" / "image_by_article_idx.npy"
    )
    assert paths.image_embedding_mask_file == processed / "embeddings" / "image_mask.npy"
    assert paths.text_embeddings_by_idx_file == processed / "embeddings" / "text_by_article_idx.npy"
    assert paths.text_embedding_mask_file == processed / "embeddings" / "text_mask.npy"
    # The aligned copies must not overwrite the read-only source embeddings.
    assert paths.image_embeddings_by_idx_file != paths.image_embeddings_file


def test_get_paths_resolves_relative_to_config_location(tmp_path: Path) -> None:
    # A fake repo: <tmp>/configs/base.yaml with relative roots.
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    base_yaml = configs_dir / "base.yaml"
    raw = yaml.safe_load((PROJECT_ROOT / "configs" / "base.yaml").read_text())
    raw["paths"]["data_root"] = "somewhere/data"
    base_yaml.write_text(yaml.safe_dump(raw))

    paths = get_paths(base_yaml)

    assert paths.project_root == tmp_path.resolve()
    assert paths.data_root == tmp_path.resolve() / "somewhere" / "data"
    assert paths.fold_manifest_file == tmp_path.resolve() / "configs" / "fold_manifest.yaml"


def test_get_paths_default_uses_repo_config() -> None:
    paths = get_paths()
    assert paths.data_root == PROJECT_ROOT / "data"
    assert paths.artifacts_root == PROJECT_ROOT / "artifacts"
    assert paths.results_root == PROJECT_ROOT / "results"


def test_paths_of_the_final_evaluation(tmp_path: Path) -> None:
    paths = build_paths(make_config("data", "artifacts"), tmp_path)
    results = tmp_path / "results"
    assert paths.m8_config_file == tmp_path / "configs" / "m8_final.yaml"
    assert paths.t_phase_results_file == results / "t_phase.csv"
    assert paths.t_phase_metrics_file == results / "t_phase_metrics.csv"
    assert paths.t_phase_recall_file == results / "t_phase_recall.csv"
    assert paths.t_phase_goals_file == results / "t_phase_goals.csv"
    assert paths.t_phase_label_access_file == results / "t_phase_label_access.csv"
    assert paths.m8_rehearsal_results_file == results / "m8_rehearsal.csv"
    assert paths.m8_rehearsal_metrics_file == results / "m8_rehearsal_metrics.csv"
    assert paths.m8_rehearsal_recall_file == results / "m8_rehearsal_recall.csv"
    assert paths.m8_rehearsal_check_file == results / "m8_rehearsal_check.csv"
    assert paths.m8_runs_file == results / "m8_runs.csv"
    # An M8 Dataset lives under its area, apart from the V phase's Dataset of the same fold.
    lgb = tmp_path / "artifacts" / "lgb_datasets"
    assert paths.final_dataset_dir("m8_rehearsal", "full", "V4", 40) == (
        lgb / "m8_rehearsal" / "full" / "V4_r40"
    )
    assert paths.final_dataset_dir("t_phase", "minus_r6", "T1", 40) == (
        lgb / "t_phase" / "minus_r6" / "T1_r40"
    )
    assert paths.ranker_dataset_dir("full", "V4", 40) == lgb / "full" / "V4_r40"


def test_paths_of_the_diagnostics_after_m8(tmp_path: Path) -> None:
    paths = build_paths(make_config("data", "artifacts"), tmp_path)
    results = tmp_path / "results"
    assert paths.m8_diagnostics_config_file == tmp_path / "configs" / "m8_diagnostics.yaml"
    assert paths.t_diag_label_access_file == results / "t_diag_label_access.csv"
    assert paths.t_diag_decomposition_file == results / "t_diag_decomposition.csv"
    assert paths.t_diag_concentration_file == results / "t_diag_concentration.csv"
    assert paths.t_diag_query_groups_file == results / "t_diag_query_groups.csv"
    assert paths.t_diag_bootstrap_file == results / "t_diag_bootstrap.csv"
    # The diagnostics' tables are other files than the records of the formal run.
    diag_files = {
        paths.t_diag_label_access_file,
        paths.t_diag_decomposition_file,
        paths.t_diag_concentration_file,
        paths.t_diag_query_groups_file,
        paths.t_diag_bootstrap_file,
    }
    assert len(diag_files) == 5 and all(path.name.startswith("t_diag_") for path in diag_files)
    assert paths.t_phase_label_access_file not in diag_files


def test_paths_of_m11_versioning(tmp_path: Path) -> None:
    paths = build_paths(make_config("data", "artifacts"), tmp_path)
    assert paths.versioning_config_file == tmp_path / "configs" / "m11_versioning.yaml"
    # The manifest of the DVC-tracked files is in git, next to the other configs.
    assert paths.data_manifest_file == tmp_path / "configs" / "data_manifest.json"
    # The bootstrap state is data (tracked with DVC), not a regenerable artifact.
    assert paths.bootstrap_dir == tmp_path / "data" / "bootstrap"
    assert paths.bootstrap_manifest_file == tmp_path / "data" / "bootstrap" / "manifest.json"
    runs = tmp_path / "artifacts" / "pipeline" / "runs"
    assert paths.pipeline_runs_dir == runs
    assert paths.run_manifest_file("2020-07-29-eval") == runs / "2020-07-29-eval" / "manifest.json"
    # The MLflow store is under artifacts/, which git ignores; a pipeline run is not a
    # stage-1 run folder (artifacts/runs/).
    assert paths.mlflow_store_file == tmp_path / "artifacts" / "mlflow" / "mlflow.db"
    assert paths.mlflow_artifacts_dir == tmp_path / "artifacts" / "mlflow" / "artifacts"
    assert paths.pipeline_runs_dir != paths.run_dir("x").parent


# ---------------------------------------------------------------- M13: the roots of a tick


def test_without_roots_a_tick_uses_the_paths_of_the_repository(tmp_path: Path) -> None:
    paths = build_paths(make_config("data", "artifacts"), tmp_path)
    # The laptop: no variable, no calibration. The very same object comes back, so the
    # state is artifacts/pipeline/ and the tables are results/m12_replay/, as in M12.
    assert tick_root_paths(paths, {}, calibration=False) is paths
    assert tick_root_paths(paths, {PIPELINE_STATE_ROOT_VARIABLE: ""}, False) is paths


def test_roots_from_the_environment_move_inputs_state_and_tables(tmp_path: Path) -> None:
    paths = build_paths(make_config("data", "artifacts"), tmp_path)
    environment = {
        PIPELINE_INPUTS_ROOT_VARIABLE: "/mnt/inputs",
        PIPELINE_STATE_ROOT_VARIABLE: "/mnt/pipeline",
    }
    rooted = tick_root_paths(paths, environment, calibration=False)
    assert isinstance(rooted, TickRootPaths) and not rooted.calibration
    # Inputs keep their plain layout below the inputs root.
    assert rooted.transactions_file == Path("/mnt/inputs/data/processed/transactions.parquet")
    assert rooted.bootstrap_dir == Path("/mnt/inputs/data/bootstrap")
    assert rooted.category_codes_file == Path("/mnt/inputs/artifacts/features/category_codes.json")
    # The state and the pipeline's result tables are under the state root.
    assert rooted.pipeline_state_dir == Path("/mnt/pipeline")
    assert rooted.replay_table_file("steps") == Path("/mnt/pipeline/results/steps.csv")
    # Code and configs stay in the repository (the image).
    assert rooted.project_root == tmp_path
    assert rooted.ranker_lock_file == tmp_path / "configs" / "locks" / "ranker.lock.yaml"
    # Only a state root: the inputs stay where configs/base.yaml puts them.
    only_state = tick_root_paths(paths, {PIPELINE_STATE_ROOT_VARIABLE: "/mnt/pipeline"}, False)
    assert only_state.data_root == paths.data_root
    assert only_state.pipeline_state_dir == Path("/mnt/pipeline")


def test_a_calibration_tick_has_a_state_of_its_own_next_to_the_state(tmp_path: Path) -> None:
    paths = build_paths(make_config("data", "artifacts"), tmp_path)
    local = tick_root_paths(paths, {}, calibration=True)
    assert isinstance(local, TickRootPaths) and local.calibration
    assert local.pipeline_state_dir == tmp_path / "artifacts" / "pipeline-calibration"
    assert local.replay_results_dir == tmp_path / "artifacts" / "pipeline-calibration" / "results"
    mounted = tick_root_paths(paths, {PIPELINE_STATE_ROOT_VARIABLE: "/mnt/pipeline"}, True)
    assert mounted.pipeline_state_dir == Path("/mnt/pipeline-calibration")
    assert calibration_state_dir(Path("/mnt/pipeline")) == Path("/mnt/pipeline-calibration")


def test_roots_must_be_absolute_and_inputs_need_a_state_root(tmp_path: Path) -> None:
    paths = build_paths(make_config("data", "artifacts"), tmp_path)
    with pytest.raises(ValueError, match="must be an absolute path"):
        tick_root_paths(paths, {PIPELINE_STATE_ROOT_VARIABLE: "pipeline"}, False)
    # Inputs elsewhere and no state root: the state would be written among the inputs.
    with pytest.raises(ValueError, match="among the inputs"):
        tick_root_paths(paths, {PIPELINE_INPUTS_ROOT_VARIABLE: "/mnt/inputs"}, False)
