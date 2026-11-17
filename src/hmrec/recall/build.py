"""Build R1-R5 channels from a raw config; one place for the chunk-size wiring.

Two callers build the same channels and must never drift apart:
    hmrec.hpo.runner     the D-phase trials (configs from configs/recall/ + the HPO grid)
    hmrec.recall.locked  everything after M5 (configs from configs/locks/)

Chunk sizes and threads come from configs/base.yaml. They change memory and
speed, never the candidates.
"""

from collections.abc import Mapping
from typing import Any

from hmrec.config import (
    BaseConfig,
    parse_itemcf_config,
    parse_popularity_config,
    parse_repurchase_config,
    parse_sibling_config,
    parse_wals_config,
)
from hmrec.recall.base import RecallChannel
from hmrec.recall.r1_popularity import PopularityChannel
from hmrec.recall.r2_repurchase import RepurchaseChannel
from hmrec.recall.r3_sibling import SiblingChannel
from hmrec.recall.r4_itemcf import ItemCFChannel
from hmrec.recall.r5_wals import WalsChannel
from hmrec.recall.twotower.infer import TwoTowerChunks


def build_statistical_channel(
    channel: str, raw_config: Mapping[str, Any], base: BaseConfig
) -> RecallChannel:
    """Parse a raw R1-R5 config and build its recall channel.

    Args:
        channel: "r1_popularity", "r2_repurchase", "r3_sibling", "r4_itemcf" or "r5_wals".
        raw_config: The channel config, with the keys of configs/recall/{channel}.yaml.
        base: configs/base.yaml (chunk sizes, threads).

    Raises:
        ValueError: If `channel` is not one of R1-R5 (R6 needs a model and is
            built by its own code), or the config is invalid.
    """
    chunking = base.chunking
    if channel == "r1_popularity":
        return PopularityChannel(parse_popularity_config(raw_config))
    if channel == "r2_repurchase":
        return RepurchaseChannel(parse_repurchase_config(raw_config))
    if channel == "r3_sibling":
        return SiblingChannel(parse_sibling_config(raw_config))
    if channel == "r4_itemcf":
        config = parse_itemcf_config(raw_config)
        return ItemCFChannel(config, chunking.item_chunk_size, chunking.query_chunk_size)
    if channel == "r5_wals":
        config = parse_wals_config(raw_config)
        return WalsChannel(
            config,
            base.num_threads,
            chunking.dense_score_query_chunk_size,
            chunking.pair_chunk_size,
        )
    raise ValueError(f"{channel!r} is not one of the channels R1-R5")


def twotower_chunks(base: BaseConfig) -> TwoTowerChunks:
    """R6 prediction chunk sizes from configs/base.yaml."""
    return TwoTowerChunks(
        encode_queries=base.chunking.dense_score_query_chunk_size,
        score_queries=base.chunking.twotower_score_query_chunk_size,
        pairs=base.chunking.pair_chunk_size,
    )
