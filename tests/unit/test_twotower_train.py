"""Tests for recall.twotower.train: alarm split, order, checkpoints, exact CPU reproducibility."""

from pathlib import Path

import numpy as np
import pytest
import torch
from test_twotower_model import TINY_MODEL

from conftest import (
    TINY_N_ARTICLES,
    make_tiny_articles,
    make_tiny_customers,
    make_tiny_embeddings,
)
from hmrec.config import TwoTowerTrainingConfig
from hmrec.recall.twotower.data import (
    CategoryCodes,
    ItemFeatureArrays,
    TrainingArrays,
    build_customer_profiles,
    build_item_feature_arrays,
    get_batch,
    items_to_device,
    profiles_to_device,
    samples_to_device,
)
from hmrec.recall.twotower.loss import article_log_q
from hmrec.recall.twotower.model import TwoTowerModel, build_two_tower_model
from hmrec.recall.twotower.train import (
    TrainingContext,
    batch_slices,
    check_model_identity,
    epoch_order,
    load_checkpoint,
    make_optimizer,
    model_identity,
    save_checkpoint,
    split_alarm_users,
    train_step,
)

PAD = TINY_N_ARTICLES
CPU = torch.device("cpu")
TRAINING = TwoTowerTrainingConfig(
    batch_size=3,
    temperature=0.05,
    learning_rate=0.01,
    fine_tune_learning_rate=0.001,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-8,
    weight_decay=0.0,
    epochs=2,
    seed=42,
    alarm_user_share=0.25,
    log_every_steps=100,
    device="cpu",
)


def _arrays() -> TrainingArrays:
    """8 samples of customers 0..3; some histories padded, one empty."""
    history = np.array(
        [[1, 2, PAD], [1, 2, PAD], [0, PAD, PAD], [PAD, PAD, PAD]]
        + [[3, 4, 5], [5, PAD, PAD], [2, 1, 0], [4, PAD, PAD]],
        dtype=np.int32,
    )
    return TrainingArrays(
        customer_idx=np.array([0, 0, 1, 2, 3, 1, 2, 3], dtype=np.int32),
        t_dat_day=np.array([5, 5, 5, 6, 6, 7, 7, 7], dtype=np.int16),
        article_idx=np.array([0, 3, 2, 5, 1, 0, 4, 0], dtype=np.int32),
        group_id=np.array([0, 0, 1, 2, 3, 4, 5, 6], dtype=np.int32),
        history=history,
        history_mask=history != PAD,
    )


def _setup() -> tuple[ItemFeatureArrays, CategoryCodes, TrainingContext]:
    items = build_item_feature_arrays(make_tiny_embeddings(), make_tiny_articles(), 4, 3)
    profiles = build_customer_profiles(make_tiny_customers())
    arrays = _arrays()
    context = TrainingContext(
        samples=samples_to_device(arrays, CPU),
        profiles=profiles_to_device(profiles, CPU),
        features=items_to_device(items, CPU),
        log_q=torch.from_numpy(article_log_q(arrays.article_idx, TINY_N_ARTICLES + 1)),
        temperature=TRAINING.temperature,
        device=CPU,
    )
    return items, profiles, context


def _model(items: ItemFeatureArrays, profiles: CategoryCodes) -> TwoTowerModel:
    return build_two_tower_model(TINY_N_ARTICLES, 4, 3, items.codes, profiles, 3, TINY_MODEL, 42)


# ---------------------------------------------------------------- split and order


def test_alarm_split_is_by_user_and_reproducible() -> None:
    customer_idx = np.repeat(np.arange(100, dtype=np.int32), 3)
    split = split_alarm_users(customer_idx, share=0.1, seed=7)
    assert split.n_alarm_users == 10
    assert split.alarm_indices.size == 30  # 10 users x 3 samples
    assert np.array_equal(
        np.sort(np.concatenate([split.train_indices, split.alarm_indices])), np.arange(300)
    )
    alarm_users = set(customer_idx[split.alarm_indices].tolist())
    assert not alarm_users & set(customer_idx[split.train_indices].tolist())
    again = split_alarm_users(customer_idx, share=0.1, seed=7)
    assert np.array_equal(split.alarm_indices, again.alarm_indices)


def test_epoch_order_is_seeded_per_epoch() -> None:
    indices = np.arange(50, dtype=np.int64)
    first = epoch_order(indices, seed=42, epoch=1)
    assert np.array_equal(first, epoch_order(indices, seed=42, epoch=1))
    assert not np.array_equal(first, epoch_order(indices, seed=42, epoch=2))
    assert np.array_equal(np.sort(first), indices)


def test_batch_slices_keep_the_last_partial_batch() -> None:
    assert [(s.start, s.stop) for s in batch_slices(5, 2)] == [(0, 2), (2, 4), (4, 5)]


# ---------------------------------------------------------------- reproducibility


def _three_steps() -> tuple[list[float], dict[str, torch.Tensor]]:
    items, profiles, context = _setup()
    model = _model(items, profiles)
    optimizer = make_optimizer(model, TRAINING, TRAINING.learning_rate)
    order = epoch_order(np.arange(8, dtype=np.int64), TRAINING.seed, epoch=1)
    losses = []
    for part in list(batch_slices(order.size, TRAINING.batch_size)):
        batch = get_batch(context.samples, context.profiles, torch.from_numpy(order[part]))
        losses.append(float(train_step(model, optimizer, batch, context).item()))
    return losses, model.state_dict()


def test_three_cpu_steps_reproduce_exactly() -> None:
    losses_a, state_a = _three_steps()
    losses_b, state_b = _three_steps()
    assert len(losses_a) == 3  # 8 samples, batch 3 -> 3, 3, 2
    assert losses_a == losses_b
    assert all(torch.equal(state_a[name], state_b[name]) for name in state_a)
    # Training moved the weights away from the zero-initialised article embedding.
    assert torch.count_nonzero(state_a["item_tower.article_embedding.weight"]) > 0
    # ... but never the padding row.
    assert torch.count_nonzero(state_a["item_tower.article_embedding.weight"][PAD]) == 0


# ---------------------------------------------------------------- checkpoints


def test_checkpoint_round_trip_and_no_overwrite(tmp_path: Path) -> None:
    items, profiles, _ = _setup()
    trained_losses, trained_state = _three_steps()
    model = _model(items, profiles)
    model.load_state_dict(trained_state)
    identity = model_identity(TINY_N_ARTICLES, items.codes, profiles)
    path = tmp_path / "epoch_1.pt"
    save_checkpoint(
        model, path, {"model_identity": identity, "epoch": {"loss": trained_losses[-1]}}
    )
    fresh = build_two_tower_model(TINY_N_ARTICLES, 4, 3, items.codes, profiles, 3, TINY_MODEL, 0)
    metadata = load_checkpoint(fresh, path)
    assert all(torch.equal(fresh.state_dict()[k], v) for k, v in trained_state.items())
    check_model_identity(metadata["model_identity"], identity, path)
    with pytest.raises(FileExistsError):
        save_checkpoint(model, path, {})


def test_model_identity_mismatch_is_refused(tmp_path: Path) -> None:
    items, profiles, _ = _setup()
    identity = model_identity(TINY_N_ARTICLES, items.codes, profiles)
    changed = {**identity, "profile_vocabularies": {"age_bucket": ["other"]}}
    with pytest.raises(ValueError, match="profile_vocabularies differs"):
        check_model_identity(changed, identity, tmp_path / "x.pt")
