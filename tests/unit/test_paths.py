from pathlib import Path

import yaml

from hmrec.config import BaseConfig, ChunkingConfig, MemoryConfig, PathsConfig
from hmrec.paths import PROJECT_ROOT, build_paths, get_paths, resolve_root


def make_config(data_root: str, artifacts_root: str, results_root: str = "results") -> BaseConfig:
    return BaseConfig(
        paths=PathsConfig(
            data_root=data_root, artifacts_root=artifacts_root, results_root=results_root
        ),
        seed=0,
        num_threads=1,
        memory=MemoryConfig(limit_gb=1.0, sample_interval_seconds=0.1),
        chunking=ChunkingConfig(query_chunk_size=1, item_chunk_size=1),
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
