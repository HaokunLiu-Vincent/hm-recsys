"""Tests for recall.twotower.model: encodings, dimensions, pooling, padding, dedup encoding."""

import math

import numpy as np
import pytest
import torch

from conftest import (
    TINY_N_ARTICLES,
    make_tiny_articles,
    make_tiny_customers,
    make_tiny_embeddings,
)
from hmrec.config import TwoTowerModelConfig
from hmrec.recall.twotower.data import (
    Batch,
    CategoryCodes,
    build_customer_profiles,
    build_item_feature_arrays,
    items_to_device,
)
from hmrec.recall.twotower.model import (
    CategoryEncoder,
    ItemTower,
    UserTower,
    build_two_tower_model,
)

TINY_MODEL = TwoTowerModelConfig(
    output_dim=4,
    image_projection_dim=3,
    text_projection_dim=3,
    article_embedding_dim=2,
    product_code_embedding_dim=2,
    category_embedding_dim=2,
    one_hot_max_cardinality=3,
    hidden_dim=5,
    embedding_init_std=0.02,
)
# The real sizes of configs/recall/r6_twotower.yaml.
REAL_MODEL = TwoTowerModelConfig(128, 128, 128, 64, 32, 16, 20, 256, 0.02)
PAD = TINY_N_ARTICLES
CPU = torch.device("cpu")


def _tiny_parts() -> tuple:
    items = build_item_feature_arrays(make_tiny_embeddings(), make_tiny_articles(), 4, 3)
    profiles = build_customer_profiles(make_tiny_customers())
    model = build_two_tower_model(TINY_N_ARTICLES, 4, 3, items.codes, profiles, 3, TINY_MODEL, 0)
    return model, items_to_device(items, CPU), items, profiles


def _batch(
    articles: list[int], groups: list[int], history: list[list[int]], profile: list[list[int]]
) -> Batch:
    history_tensor = torch.tensor(history, dtype=torch.int64)
    return Batch(
        article_idx=torch.tensor(articles, dtype=torch.int64),
        group_id=torch.tensor(groups, dtype=torch.int64),
        history=history_tensor,
        history_mask=history_tensor != PAD,
        profile=torch.tensor(profile, dtype=torch.int64),
    )


BATCH = _batch(
    articles=[0, 3, 0],
    groups=[0, 0, 1],
    history=[[1, 2, PAD], [1, PAD, PAD], [PAD, PAD, PAD]],
    profile=[[1, 1, 1], [2, 0, 2], [3, 2, 0]],
)


# ---------------------------------------------------------------- encodings and sizes


@pytest.mark.parametrize(
    ("cardinality", "expected_dim", "one_hot"), [(20, 20, True), (21, 16, False), (6, 6, True)]
)
def test_category_encoder_rule(cardinality: int, expected_dim: int, one_hot: bool) -> None:
    encoder = CategoryEncoder(cardinality, 20, 16, 0.02)
    assert encoder.output_dim == expected_dim
    assert (encoder.embedding is None) == one_hot
    codes = torch.tensor([0, cardinality - 1])
    assert encoder(codes).shape == (2, expected_dim)
    if encoder.embedding is not None:
        assert torch.count_nonzero(encoder.embedding.weight[0]) == 0  # padding row stays 0


def _codes(columns: tuple[str, ...], cardinalities: tuple[int, ...]) -> CategoryCodes:
    vocabularies = {
        c: tuple(str(v) for v in range(n - 1)) for c, n in zip(columns, cardinalities, strict=True)
    }
    return CategoryCodes(np.zeros((5, len(columns)), dtype=np.int32), columns, vocabularies)


def test_real_sizes_give_442_and_144() -> None:
    # Code counts of the real data (code 0 included), see the model.py docstring.
    article_codes = _codes(
        (
            "product_code",
            "product_type_no",
            "product_group_name",
            "garment_group_no",
            "index_group_no",
            "section_no",
            "colour_group_code",
        ),
        (47225, 133, 20, 22, 6, 58, 51),
    )
    item_tower = ItemTower(4, 512, 384, article_codes, REAL_MODEL)
    # 128 + 128 + 64 + 32 + 16 (type) + 20 (group, one-hot) + 16 (garment)
    # + 6 (index, one-hot) + 16 (section) + 16 (colour) = 442
    assert item_tower.concat_dim == 442
    profile_codes = _codes(
        ("age_bucket", "club_member_status", "fashion_news_frequency"), (8, 4, 4)
    )
    user_tower = UserTower(20, 128, profile_codes, REAL_MODEL)
    assert user_tower.profile_dim == 16  # all one-hot: 8 + 4 + 4
    assert user_tower.mlp[0].in_features == 144


def test_tiny_item_tower_concat_dim_and_unit_output() -> None:
    model, features, _, _ = _tiny_parts()
    # 3 + 3 + 2 + 2 + 2 (type, emb) + 3 (group, 1-hot) + 2 (garment, 1-hot)
    # + 3 (index, 1-hot) + 2 (section, emb) + 2 (colour, emb) = 24
    assert model.item_tower.concat_dim == 24
    vectors = model.item_tower(torch.arange(TINY_N_ARTICLES), features)
    assert torch.allclose(vectors.norm(dim=1), torch.ones(TINY_N_ARTICLES), atol=1e-6)


def test_id_embeddings_start_at_zero() -> None:
    model, _, _, _ = _tiny_parts()
    assert torch.count_nonzero(model.item_tower.article_embedding.weight) == 0
    assert torch.count_nonzero(model.item_tower.product_code_embedding.weight) == 0


def test_missing_image_contributes_nothing_not_even_bias() -> None:
    # Article 2 has no image: changing its image vector must not change its output.
    model, features, items, _ = _tiny_parts()
    before = model.item_tower(torch.tensor([2]), features)
    items.image[2] = 100.0
    after = model.item_tower(torch.tensor([2]), items_to_device(items, CPU))
    assert torch.equal(before, after)


# ---------------------------------------------------------------- pooling


def test_position_weighted_pooling_hand_computed() -> None:
    model, _, _, _ = _tiny_parts()
    tower = model.user_tower
    with torch.no_grad():
        tower.position_logits.copy_(torch.tensor([0.0, math.log(2.0), 5.0]))
    # Row 0: slots 0, 1 filled -> weights e^0 : e^ln2 = 1 : 2 -> (1/3, 2/3, 0); the large
    # logit of the padded slot 2 is ignored. Row 1: no history -> all 0.
    mask = torch.tensor([[True, True, False], [False, False, False]])
    weights = tower.position_weights(mask)
    assert torch.allclose(weights[0], torch.tensor([1 / 3, 2 / 3, 0.0]))
    assert torch.equal(weights[1], torch.zeros(3))
    history = torch.zeros(2, 3, 4)
    history[0, 0, 0], history[0, 1, 1], history[0, 2, 2] = 1.0, 1.0, 99.0
    pooled = tower.pool(history, mask)
    assert torch.allclose(pooled[0], torch.tensor([1 / 3, 2 / 3, 0.0, 0.0]))
    assert torch.equal(pooled[1], torch.zeros(4))


def test_empty_history_gives_finite_vectors_and_gradients() -> None:
    model, features, _, _ = _tiny_parts()
    history = torch.full((1, 3), PAD)
    users = model.encode_users(history, history != PAD, torch.tensor([[1, 1, 1]]), features)
    users.sum().backward()
    assert torch.isfinite(users).all()
    assert torch.isfinite(model.user_tower.position_logits.grad).all()


# ---------------------------------------------------------------- padding and dedup


def test_padding_is_never_encoded_and_gets_no_gradient() -> None:
    model, features, _, _ = _tiny_parts()
    seen: list[torch.Tensor] = []
    model.item_tower.register_forward_pre_hook(lambda _, args: seen.append(args[0].clone()))
    encoding = model.encode_batch(BATCH, features)
    (encoding.users.sum() + encoding.columns.sum()).backward()
    assert all(PAD not in ids.tolist() for ids in seen)
    tower = model.item_tower
    assert tower.article_embedding.weight.grad is not None
    assert torch.count_nonzero(tower.article_embedding.weight.grad[PAD]) == 0
    assert tower.product_code_embedding.weight.grad is not None
    assert torch.count_nonzero(tower.product_code_embedding.weight.grad[0]) == 0


def test_dedup_encoding_matches_encoding_one_by_one() -> None:
    model, features, _, _ = _tiny_parts()
    encoding = model.encode_batch(BATCH, features)
    # Targets: articles 0, 3, 0 -> 2 distinct columns; samples 0 and 2 share column 0.
    assert encoding.column_article_idx.tolist() == [0, 3]
    assert encoding.target_column.tolist() == [0, 1, 0]
    one_by_one = torch.cat([model.item_tower(torch.tensor([a]), features) for a in [0, 3, 0]])
    assert torch.allclose(encoding.columns[encoding.target_column], one_by_one, atol=1e-6)
    # Users: history slots encoded one at a time, pooled with the same weights.
    history_vectors = torch.zeros(3, 3, 4)
    for row, slot in zip(*torch.nonzero(BATCH.history_mask, as_tuple=True), strict=True):
        article = BATCH.history[row, slot].reshape(1)
        history_vectors[row, slot] = model.item_tower(article, features)[0]
    users = model.user_tower(history_vectors, BATCH.history_mask, BATCH.profile)
    assert torch.allclose(encoding.users, users, atol=1e-6)
