"""Tests for recall.twotower.benchmark: the multi-positive loss, a short CPU run, the summary."""

import math

import numpy as np
import polars as pl
import pytest
import torch

from hmrec.config import TwoTowerBenchmarkConfig
from hmrec.recall.twotower.benchmark import (
    BenchmarkTimes,
    in_batch_softmax_loss,
    make_placeholder_model,
    placeholder_step_function,
    run_benchmark,
    summarize_benchmark,
)
from hmrec.recall.twotower.data import (
    TrainingArrays,
    build_item_feature_arrays,
    items_to_device,
    samples_to_device,
)

CONFIG = TwoTowerBenchmarkConfig(
    steps=4,
    warmup_steps=1,
    batch_size=3,
    embedding_dim=8,
    learning_rate=0.01,
    temperature=0.5,
    seed=0,
)


def test_loss_drops_other_positives_of_the_same_group() -> None:
    # Users and items are unit vectors; user i . item j = 1 if i == j, else 0.
    # Samples 0 and 1 share a group, so for user 0 item 1 is removed (and vice
    # versa). With temperature 1:
    #   rows 0, 1: softmax over {self: e^1, item 2: e^0}  -> loss ln(1 + e^-1)
    #   row 2:     softmax over {self: e^1, 2 x e^0}       -> loss ln(1 + 2 e^-1)
    eye = torch.eye(3)
    loss = in_batch_softmax_loss(eye, eye, torch.tensor([7, 7, 8]), temperature=1.0)
    expected = (2 * math.log(1 + math.exp(-1)) + math.log(1 + 2 * math.exp(-1))) / 3
    assert loss.item() == pytest.approx(expected, rel=1e-6)


def _tiny_setup() -> tuple:
    rng = np.random.default_rng(0)
    n_articles = 5
    articles = pl.DataFrame(
        {
            "article_idx": list(range(n_articles)),
            "product_code": [1, 1, 2, 2, 3],
            "product_type_no": [1, 2, 1, 2, 1],
            "product_group_name": ["a", "b", "a", "b", "a"],
            "garment_group_no": [1] * n_articles,
            "index_group_no": [1] * n_articles,
            "section_no": [1] * n_articles,
            "colour_group_code": [1, 2, 3, 4, 5],
        }
    )
    embeddings = {
        "image": rng.standard_normal((n_articles, 6)).astype(np.float32),
        "image_mask": np.array([True, True, False, True, True]),
        "text": rng.standard_normal((n_articles, 4)).astype(np.float32),
        "text_mask": np.ones(n_articles, dtype=np.bool_),
    }
    items = build_item_feature_arrays(embeddings, articles, image_dim=6, text_dim=4)
    history = np.full((6, 2), n_articles, dtype=np.int32)  # all padding ...
    history[:3, 0] = [0, 1, 3]  # ... except one purchase for the first 3 samples
    arrays = TrainingArrays(
        customer_idx=np.array([0, 0, 1, 1, 2, 2], dtype=np.int32),
        t_dat_day=np.zeros(6, dtype=np.int16),
        article_idx=np.array([0, 1, 2, 3, 4, 0], dtype=np.int32),
        group_id=np.array([0, 0, 1, 1, 2, 2], dtype=np.int32),
        history=history,
        history_mask=history < n_articles,
    )
    return items, arrays


def test_short_cpu_run_records_every_step_and_learns() -> None:
    items, arrays = _tiny_setup()
    device = torch.device("cpu")
    device_items = items_to_device(items, device)
    profiles = torch.tensor([[1, 1], [2, 1], [1, 2]], dtype=torch.int32)
    model = make_placeholder_model(device_items, items.codes.cardinalities, (3, 3), CONFIG)
    step = placeholder_step_function(model, CONFIG.learning_rate, CONFIG.temperature)
    times = run_benchmark(step, samples_to_device(arrays, device), profiles, CONFIG, device)
    assert len(times.fetch_seconds) == len(times.compute_seconds) == CONFIG.steps
    assert all(math.isfinite(loss) for loss in times.losses)
    # The padding row's ID embedding stays zero (padding_idx), even after training.
    assert torch.count_nonzero(model.article_embedding.weight[items.padding_idx]) == 0


def test_summarize_benchmark_excludes_warmup() -> None:
    times = BenchmarkTimes(
        fetch_seconds=[5.0, 1.0, 1.0],
        compute_seconds=[5.0, 3.0, 3.0],
        losses=[2.0, 1.5, 1.0],
    )
    summary = summarize_benchmark(times, warmup_steps=1)
    assert (summary.fetch_total, summary.compute_total) == (7.0, 11.0)
    assert (summary.steady_fetch_mean, summary.steady_compute_mean) == (1.0, 3.0)
    assert summary.steady_fetch_share == 0.25
    assert (summary.first_loss, summary.last_loss) == (2.0, 1.0)
