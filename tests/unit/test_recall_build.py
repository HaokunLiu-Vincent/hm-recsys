"""Tests for recall.build: one place builds R1-R5 for the HPO runner and the locked path."""

import dataclasses
from pathlib import Path

import pytest

from hmrec.config import load_base_config, load_yaml
from hmrec.hpo.runner import build_channel
from hmrec.recall.build import build_statistical_channel, twotower_chunks
from hmrec.recall.r4_itemcf import ItemCFChannel
from hmrec.recall.r5_wals import WalsChannel
from hmrec.recall.twotower.infer import TwoTowerChunks

REPO = Path(__file__).resolve().parents[2]
BASE = load_base_config(REPO / "configs" / "base.yaml")
# Chunk sizes that are all different, so a swapped argument would be noticed.
CHUNKING = dataclasses.replace(
    BASE.chunking,
    query_chunk_size=11,
    item_chunk_size=12,
    dense_score_query_chunk_size=13,
    twotower_score_query_chunk_size=14,
    pair_chunk_size=15,
)
SMALL_BASE = dataclasses.replace(BASE, num_threads=3, chunking=CHUNKING)
STATISTICAL = ("r1_popularity", "r2_repurchase", "r3_sibling", "r4_itemcf", "r5_wals")


def _default(channel: str) -> dict:
    return load_yaml(REPO / "configs" / "recall" / f"{channel}.yaml")


def test_r4_and_r5_get_their_chunk_sizes_and_threads_from_base_yaml() -> None:
    r4 = build_statistical_channel("r4_itemcf", _default("r4_itemcf"), SMALL_BASE)
    assert isinstance(r4, ItemCFChannel)
    assert (r4.item_chunk_size, r4.query_chunk_size) == (12, 11)
    r5 = build_statistical_channel("r5_wals", _default("r5_wals"), SMALL_BASE)
    assert isinstance(r5, WalsChannel)
    assert (r5.num_threads, r5.query_chunk_size, r5.pair_chunk_size) == (3, 13, 15)


@pytest.mark.parametrize("channel", STATISTICAL)
def test_hpo_runner_builds_r1_to_r5_through_the_same_function(channel: str) -> None:
    raw = _default(channel)
    from_runner = build_channel(channel, raw, SMALL_BASE)
    from_build = build_statistical_channel(channel, raw, SMALL_BASE)
    assert type(from_runner) is type(from_build)
    # Same config and the same chunk sizes / threads (every attribute of the channel).
    assert vars(from_runner) == vars(from_build)


def test_r6_and_unknown_channels_are_refused() -> None:
    for channel in ("r6_twotower", "r7_graph"):
        with pytest.raises(ValueError, match="not one of the channels R1-R5"):
            build_statistical_channel(channel, {}, BASE)


def test_invalid_config_is_refused() -> None:
    raw = {**_default("r2_repurchase"), "quota": 0}
    with pytest.raises(ValueError, match="quota"):
        build_statistical_channel("r2_repurchase", raw, BASE)


def test_twotower_chunks_map_the_three_base_yaml_sizes() -> None:
    assert twotower_chunks(SMALL_BASE) == TwoTowerChunks(
        encode_queries=13, score_queries=14, pairs=15
    )
