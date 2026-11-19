"""Tests for deepfm.train and deepfm.score: batches, reproducibility, variants, Top-K."""

from dataclasses import replace

import numpy as np
import pytest
import torch
from deepfm_example import CATEGORICAL, NUMERIC, REAL_CONFIG, make_wide

from hmrec.config import DeepFMConfig
from hmrec.deepfm.inputs import RawRows, encode_rows, fit_transform
from hmrec.deepfm.score import score_rows, top_k_per_query
from hmrec.deepfm.train import (
    TrainingSet,
    build_model,
    build_optimizer,
    listwise_batches,
    pointwise_batches,
    query_boundaries,
    train_epoch,
    training_set,
)

CPU = torch.device("cpu")
SMALL = replace(
    REAL_CONFIG,
    model=replace(REAL_CONFIG.model, embedding_dim=4, hidden_units=(8,)),
    n_bins=4,
    training=replace(REAL_CONFIG.training, batch_rows=64, device="cpu"),
)


def _variant(**changes: str | float) -> DeepFMConfig:
    """SMALL with model / training values replaced by name."""
    model = {k: v for k, v in changes.items() if hasattr(SMALL.model, k)}
    training = {k: v for k, v in changes.items() if hasattr(SMALL.training, k)}
    return replace(
        SMALL,
        model=replace(SMALL.model, **model),  # type: ignore[arg-type]  # test helper
        training=replace(SMALL.training, **training),  # type: ignore[arg-type]  # test helper
    )


def _data(config: DeepFMConfig, seed: int = 0):
    """A learnable fold of 30 queries x 12 candidates, encoded for `config`."""
    wide, labels = make_wide("F1", seed)
    raw = RawRows(
        wide.select(NUMERIC).to_numpy().astype(np.float32),
        wide.select(CATEGORICAL).to_numpy().astype(np.int16),
    )
    transform = fit_transform(
        raw.numeric, NUMERIC, CATEGORICAL, (5,) * len(CATEGORICAL),
        config.model.numeric_input, config.n_bins, (0.1, 99.9),
    )  # fmt: skip
    positives = set(labels.select("query_id", "article_idx").iter_rows())
    keys = wide.select("query_id", "article_idx").iter_rows()
    label_array = np.array([key in positives for key in keys], dtype=np.uint8)
    encoded = encode_rows(raw, transform)
    data = training_set(encoded, label_array, wide["query_id"].to_numpy(), CPU)
    return transform, encoded, data, wide


def _train(config: DeepFMConfig, seed: int, epochs: int = 2):
    transform, _, data, _ = _data(config)
    torch.set_num_threads(1)
    model = build_model(transform.layout, config.model, seed, CPU)
    optimizer = build_optimizer(model, config.training)
    results = [
        train_epoch(model, optimizer, data, config.training, seed, epoch)
        for epoch in range(1, epochs + 1)
    ]
    return model, results


# ---------------------------------------------------------------- batches


def test_pointwise_batches_hold_every_row_once_in_a_new_order_each_epoch() -> None:
    batches = list(pointwise_batches(10, 4, seed=7, epoch=1))
    assert [batch.size for batch in batches] == [4, 4, 2]
    assert sorted(np.concatenate(batches).tolist()) == list(range(10))
    again = list(pointwise_batches(10, 4, seed=7, epoch=1))
    assert all(np.array_equal(a, b) for a, b in zip(batches, again, strict=True))
    other_epoch = np.concatenate(list(pointwise_batches(10, 4, seed=7, epoch=2)))
    other_seed = np.concatenate(list(pointwise_batches(10, 4, seed=8, epoch=1)))
    assert not np.array_equal(np.concatenate(batches), other_epoch)
    assert not np.array_equal(np.concatenate(batches), other_seed)


def test_listwise_batches_never_split_a_query() -> None:
    # Queries of 3, 2, 6, 1, 4 rows (row 0..15); at most 5 rows per batch.
    query_ids = np.array([10] * 3 + [11] * 2 + [12] * 6 + [13] * 1 + [14] * 4, dtype=np.int32)
    boundaries = query_boundaries(query_ids)
    assert boundaries.tolist() == [0, 3, 5, 11, 12, 16]
    batches = list(listwise_batches(boundaries, 5, seed=3, epoch=1))
    rows = np.concatenate([batch[0] for batch in batches])
    assert sorted(rows.tolist()) == list(range(16))  # every row exactly once
    for batch_rows, query_index, n_queries in batches:
        # Whole queries only: the rows of a batch are complete runs of one query_id each.
        assert query_index.max() + 1 == n_queries
        for query in range(n_queries):
            ids = set(query_ids[batch_rows[query_index == query]].tolist())
            assert len(ids) == 1
            assert (query_ids == ids.pop()).sum() == (query_index == query).sum()
        # At most 5 rows, except the 6-row query, which is a batch alone.
        assert batch_rows.size <= 5 or n_queries == 1
    assert any(batch[0].size == 6 and batch[2] == 1 for batch in batches)
    with pytest.raises(ValueError, match="sorted by query_id"):
        query_boundaries(np.array([2, 1], dtype=np.int32))


# ---------------------------------------------------------------- training


def test_same_seed_on_cpu_gives_identical_weights_and_losses() -> None:
    first, first_results = _train(SMALL, seed=42)
    second, second_results = _train(SMALL, seed=42)
    assert [r.mean_loss for r in first_results] == [r.mean_loss for r in second_results]
    for name, tensor in first.state_dict().items():
        assert torch.equal(tensor, second.state_dict()[name]), name
    other, other_results = _train(SMALL, seed=43)
    assert other_results[0].mean_loss != first_results[0].mean_loss
    assert not torch.equal(other.embedding.weight, first.embedding.weight)


def test_training_learns_the_signal() -> None:
    # 360 rows in batches of 64 = 6 steps per epoch; the loss falls over 30 epochs.
    config = _variant(learning_rate=0.01)
    _, results = _train(config, seed=42, epochs=30)
    assert results[0].steps == 6
    assert results[-1].mean_loss < 0.6 * results[0].mean_loss


@pytest.mark.parametrize("components", ["first_order", "fm", "deep", "full"])
@pytest.mark.parametrize("numeric_input", ["binned", "standardised"])
@pytest.mark.parametrize("loss", ["pointwise_bce", "listwise_softmax"])
def test_every_variant_trains_and_scores(components: str, numeric_input: str, loss: str) -> None:
    config = _variant(components=components, numeric_input=numeric_input, loss=loss)
    model, results = _train(config, seed=42)
    assert all(np.isfinite(result.mean_loss) for result in results)
    _, encoded, _, _ = _data(config)
    scores = score_rows(model, encoded, 100, CPU)
    assert scores.shape == (360,) and scores.dtype == np.float32 and np.isfinite(scores).all()
    # The parts a variant does not use are never updated.
    untouched_mlp = components in ("first_order", "fm")
    fresh = build_model(_data(config)[0].layout, config.model, 42, CPU)
    first_layer, fresh_layer = model.mlp[0], fresh.mlp[0]
    assert isinstance(first_layer, torch.nn.Linear) and isinstance(fresh_layer, torch.nn.Linear)
    assert torch.equal(first_layer.weight, fresh_layer.weight) == untouched_mlp


def test_listwise_training_counts_whole_query_batches() -> None:
    config = _variant(loss="listwise_softmax")
    _, results = _train(config, seed=42, epochs=1)
    # 30 queries of 12 rows, at most 64 rows per batch -> 5 queries per batch -> 6 batches.
    assert results[0].steps == 6


def test_training_set_needs_rows_sorted_by_query() -> None:
    _, encoded, data, _ = _data(SMALL)
    assert isinstance(data, TrainingSet) and data.n_rows == 360
    assert data.rows.codes.dtype == torch.uint8  # stored compactly; widened per batch only
    with pytest.raises(ValueError, match="sorted by query_id"):
        training_set(encoded, np.zeros(360, dtype=np.uint8), np.arange(360, 0, -1), CPU)


# ---------------------------------------------------------------- scoring


def test_scores_do_not_depend_on_the_chunk_size() -> None:
    model, _ = _train(SMALL, seed=42)
    _, encoded, _, _ = _data(SMALL)
    whole = score_rows(model, encoded, 1000, CPU)
    for rows_per_chunk in (1, 7, 360):
        assert np.allclose(score_rows(model, encoded, rows_per_chunk, CPU), whole, atol=1e-6)
    with pytest.raises(ValueError, match="rows_per_chunk must be > 0"):
        score_rows(model, encoded, 0, CPU)


def test_top_k_orders_by_score_then_article() -> None:
    query_ids = np.array([5, 5, 5, 5, 9, 9], dtype=np.int32)
    articles = np.array([40, 10, 30, 20, 7, 8], dtype=np.int32)
    scores = np.array([0.5, 0.9, 0.5, 0.1, -1.0, 2.0], dtype=np.float32)
    top = top_k_per_query(query_ids, articles, scores, 3)
    # Query 5: 0.9 first; the tie at 0.5 goes to the smaller article_idx (30 before 40).
    assert top.select("query_id", "article_idx", "rank").rows() == [
        (5, 10, 1),
        (5, 30, 2),
        (5, 40, 3),
        (9, 8, 1),
        (9, 7, 2),
    ]
    assert str(top.schema["score"]) == "Float64" and str(top.schema["rank"]) == "Int32"
    with pytest.raises(ValueError, match="NaN"):
        top_k_per_query(query_ids, articles, np.full(6, np.nan, dtype=np.float32), 3)
    with pytest.raises(ValueError, match="top_k must be > 0"):
        top_k_per_query(query_ids, articles, scores, 0)
