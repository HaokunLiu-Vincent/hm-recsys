"""Tests for hmrec.calibration.repeat: two container runs compared bit by bit."""

from pathlib import Path

import numpy as np
import polars as pl
import torch

from hmrec.calibration.repeat import (
    VECTOR_FILES,
    repeat_model_weights,
    repeat_table,
    repeat_vectors,
)


def _table(scores: list[float]) -> pl.DataFrame:
    return pl.DataFrame(
        {"query_id": [1, 1, 2], "article_idx": [10, 11, 10], "score": scores},
        schema={"query_id": pl.Int32, "article_idx": pl.Int32, "score": pl.Float32},
    )


def test_two_parquet_files_are_compared_by_their_values(tmp_path: Path) -> None:
    # Two files with the same rows: written with another compression their bytes differ,
    # but the results they hold are the same.
    _table([0.5, 0.25, 1.0]).write_parquet(tmp_path / "first.parquet", compression="zstd")
    _table([0.5, 0.25, 1.0]).write_parquet(tmp_path / "second.parquet", compression="snappy")
    _table([0.5, 0.25, 2.0]).write_parquet(tmp_path / "third.parquet")
    assert (tmp_path / "first.parquet").read_bytes() != (tmp_path / "second.parquet").read_bytes()
    same = repeat_table("B", "candidates", tmp_path / "first.parquet", tmp_path / "second.parquet")
    assert (same.item, same.artifact, same.bit_identical) == ("B", "candidates", True)
    other = repeat_table("B", "candidates", tmp_path / "first.parquet", tmp_path / "third.parquet")
    assert not other.bit_identical and "score" in other.detail


def _save_vectors(directory: Path, item_value: float) -> None:
    directory.mkdir()
    np.save(directory / "user_customer_idx.npy", np.array([1, 2], dtype=np.int32))
    np.save(directory / "user_vectors.npy", np.ones((2, 3), dtype=np.float32))
    np.save(directory / "item_article_idx.npy", np.array([10], dtype=np.int32))
    np.save(directory / "item_vectors.npy", np.full((1, 3), item_value, dtype=np.float32))


def test_vectors_are_compared_array_by_array(tmp_path: Path) -> None:
    _save_vectors(tmp_path / "first", 0.5)
    _save_vectors(tmp_path / "same", 0.5)
    _save_vectors(tmp_path / "other", 0.75)
    same = repeat_vectors("B", tmp_path / "first", tmp_path / "same")
    assert [check.artifact for check in same] == [f"vectors: {name}" for name in VECTOR_FILES]
    assert all(check.bit_identical for check in same)
    other = {
        c.artifact: c.bit_identical
        for c in repeat_vectors("B", tmp_path / "first", tmp_path / "other")
    }
    # Only the item vectors differ; the three other arrays are the same.
    assert other == {
        "vectors: user_customer_idx": True,
        "vectors: user_vectors": True,
        "vectors: item_article_idx": True,
        "vectors: item_vectors": False,
    }


def _checkpoint(path: Path, weight: float, seconds: float) -> None:
    state = {"layer.weight": torch.tensor([weight, 2.0])}
    torch.save({"state_dict": state, "metadata": {"fine_tune_seconds": seconds}}, path)


def test_models_are_compared_by_their_weights_not_by_their_files(tmp_path: Path) -> None:
    # The checkpoint stores how long its fine-tune took, so two runs never give the same
    # file. What must repeat is the weights.
    _checkpoint(tmp_path / "first.pt", 1.0, seconds=55.0)
    _checkpoint(tmp_path / "second.pt", 1.0, seconds=61.0)
    _checkpoint(tmp_path / "third.pt", 1.5, seconds=55.0)
    assert (tmp_path / "first.pt").read_bytes() != (tmp_path / "second.pt").read_bytes()
    same = repeat_model_weights(tmp_path / "first.pt", tmp_path / "second.pt")
    assert (same.item, same.bit_identical, same.detail) == ("C2", True, "1 tensors")
    other = repeat_model_weights(tmp_path / "first.pt", tmp_path / "third.pt")
    assert not other.bit_identical and "layer.weight" in other.detail
