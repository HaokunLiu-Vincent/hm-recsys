"""Tests for recall.twotower.infer: time cut-off ("poison"), output rules, histories.

Tiny fold: anchor day 10 on FIXTURE_DAY_ZERO, articles 0..5, active pool {0, 1, 3, 4}.
Purchase lines (customer, article, day):
    customer 0: (0, 1, 3), (0, 2, 8), (0, 2, 8)      -> history as of day 10: [2, 1]
    customer 1: (1, 4, 9)                            -> history [4]
    customer 2: nothing                              -> empty history (profile only)
Poison (on / after the anchor): (0, 5, 10), (1, 3, 12), (2, 0, 10).
"""

import dataclasses
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch
from conftest import (
    TINY_N_ARTICLES,
    make_queries,
    make_recall_inputs,
    make_tiny_articles,
    make_tiny_customers,
    make_tiny_embeddings,
)
from test_twotower_model import TINY_MODEL

from hmrec.config import load_twotower_config
from hmrec.recall.base import validate_candidates
from hmrec.recall.twotower.data import (
    build_customer_profiles,
    build_item_feature_arrays,
    items_to_device,
    profiles_to_device,
)
from hmrec.recall.twotower.infer import (
    TWOTOWER_SPEC,
    TwoTowerChannel,
    TwoTowerChunks,
    TwoTowerResources,
    query_histories,
)
from hmrec.recall.twotower.model import build_two_tower_model
from hmrec.vectors import dot_pairs

REPO = Path(__file__).resolve().parents[2]
CPU = torch.device("cpu")
PAD = TINY_N_ARTICLES
LINES = [(0, 1, 3), (0, 2, 8), (0, 2, 8), (1, 4, 9)]
POISON = [(0, 5, 10), (1, 3, 12), (2, 0, 10)]
ACTIVE = [0, 1, 3, 4]
CHUNKS = TwoTowerChunks(encode_queries=2, score_queries=2, pairs=3)
QUERIES = make_queries([(11, 0), (12, 1), (13, 2)])
CONFIG = dataclasses.replace(
    load_twotower_config(REPO / "configs" / "recall" / "r6_twotower.yaml"),
    history_length=3,
    quota=2,
)


def _resources() -> TwoTowerResources:
    items = build_item_feature_arrays(make_tiny_embeddings(), make_tiny_articles(), 4, 3)
    profiles = build_customer_profiles(make_tiny_customers())
    model = build_two_tower_model(TINY_N_ARTICLES, 4, 3, items.codes, profiles, 3, TINY_MODEL, 3)
    # Give the zero-initialised ID embeddings some values, as after training.
    with torch.no_grad():
        model.item_tower.article_embedding.weight[:PAD].normal_(0.0, 1.0)
    return TwoTowerResources(
        model=model.eval(),
        features=items_to_device(items, CPU),
        profiles=profiles_to_device(profiles, CPU),
        n_articles=TINY_N_ARTICLES,
        device=CPU,
        identity={},
    )


def _channel() -> TwoTowerChannel:
    return TwoTowerChannel(CONFIG, _resources(), CHUNKS)


def test_histories_as_of_the_anchor() -> None:
    inputs = make_recall_inputs(LINES + POISON, 10, active_pool=ACTIVE)
    history, mask = query_histories(inputs, np.array([0, 1, 2], dtype=np.int32), 3, PAD)
    # Two lines of article 2 on day 8 are one purchase; poison lines are not used.
    assert history.tolist() == [[2, 1, PAD], [4, PAD, PAD], [PAD, PAD, PAD]]
    assert mask.tolist() == [[True, True, False], [True, False, False], [False] * 3]


def test_poison_on_and_after_the_anchor_changes_nothing() -> None:
    channel = _channel()
    clean = channel.generate(QUERIES, make_recall_inputs(LINES, 10, active_pool=ACTIVE))
    poisoned = channel.generate(QUERIES, make_recall_inputs(LINES + POISON, 10, active_pool=ACTIVE))
    assert clean.equals(poisoned)


def test_output_rules_quota_pool_and_profile_only_users() -> None:
    candidates = _channel().generate(QUERIES, make_recall_inputs(LINES, 10, active_pool=ACTIVE))
    validate_candidates(candidates, TWOTOWER_SPEC, quota=2)  # order, ranks, tie-break, dtypes
    assert set(candidates["article_idx"].to_list()) <= set(ACTIVE)
    # Every query gets exactly 2 candidates, including customer 2 with no history.
    assert candidates.group_by("query_id").len().sort("query_id")["len"].to_list() == [2, 2, 2]


def test_score_is_the_dot_pairs_value() -> None:
    channel = _channel()
    inputs = make_recall_inputs(LINES, 10, active_pool=ACTIVE)
    candidates = channel.generate(QUERIES, inputs)
    vectors = channel.vectors(QUERIES, inputs)
    customers = candidates.join(QUERIES, on="query_id")["customer_idx"].to_numpy()
    expected = dot_pairs(vectors, customers, candidates["article_idx"].to_numpy(), 4)
    assert np.array_equal(candidates["score"].to_numpy(), expected)


def test_tie_break_by_article_idx() -> None:
    # Make every active article identical: all scores tie, so the 2 kept are the
    # smallest article_idx (0, 1), in that order.
    resources = _resources()
    with torch.no_grad():
        resources.model.item_tower.article_embedding.weight.zero_()
    same = resources.features
    identical = dataclasses.replace(
        same,
        image=same.image[0].expand_as(same.image).clone(),
        image_mask=torch.ones_like(same.image_mask),
        text=same.text[0].expand_as(same.text).clone(),
        codes=same.codes[0].expand_as(same.codes).clone(),
    )
    channel = TwoTowerChannel(CONFIG, dataclasses.replace(resources, features=identical), CHUNKS)
    candidates = channel.generate(QUERIES, make_recall_inputs(LINES, 10, active_pool=ACTIVE))
    for query in candidates.partition_by("query_id"):
        assert query.sort("rank")["article_idx"].to_list() == [0, 1]


def test_no_queries_gives_an_empty_valid_table() -> None:
    empty = QUERIES.filter(pl.lit(False))
    result = _channel().generate(empty, make_recall_inputs(LINES, 10, active_pool=ACTIVE))
    assert result.height == 0
    assert result.schema == TWOTOWER_SPEC.schema


@pytest.mark.parametrize("score_queries", [1, 2, 3, 100])
def test_score_chunk_size_does_not_change_the_candidates(score_queries: int) -> None:
    # The score chunk only bounds memory: any size gives the same table as one big chunk.
    inputs = make_recall_inputs(LINES, 10, active_pool=ACTIVE)
    reference = _channel().generate(QUERIES, inputs)
    chunks = TwoTowerChunks(encode_queries=2, score_queries=score_queries, pairs=3)
    assert TwoTowerChannel(CONFIG, _resources(), chunks).generate(QUERIES, inputs).equals(reference)
