"""R6 Two-Tower as a recall channel (proposal §5 R6), channel "twotower".

For a fold with anchor a:
  1. History of each query customer: the last 20 purchases with t_dat_day <
     anchor_day, with the SAME rules as training (one per (day, article), most
     recent day first, same day in ascending article_idx, padding n_articles).
     The purchase lines are first cut to the fold's query customers.
  2. User vectors from the history and the profile (a customer with no history
     gets a profile-only vector, so every query gets candidates).
  3. Item vectors for the active pool A(a) only (outputs are limited to A(a), §2.4).
  4. Top-quota per query with hmrec.recall.vector_scoring (shared with R5):
     selection by a float32 matrix product per chunk of
     twotower_score_query_chunk_size queries, stored score from vectors.dot_pairs
     (so it equals the tt_dot feature exactly), order score DESC, article_idx ASC.

The model is a fixed checkpoint (config.base_epoch); nothing is trained here.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
import torch

from hmrec.config import PreprocessConfig, TwoTowerConfig
from hmrec.data.metadata import scan_processed_parquet
from hmrec.logging_utils import get_logger
from hmrec.memory import log_memory
from hmrec.paths import ProjectPaths
from hmrec.recall.base import ARTICLE_IDX, CUSTOMER_IDX, OutputSpec, RecallInputs
from hmrec.recall.twotower.data import (
    DeviceItemFeatures,
    build_customer_profiles,
    build_histories,
    daily_purchases,
    items_to_device,
    load_customer_lines_before,
    load_item_feature_arrays,
    profiles_to_device,
)
from hmrec.recall.twotower.model import TwoTowerModel, build_two_tower_model
from hmrec.recall.twotower.train import check_model_identity, load_checkpoint, model_identity
from hmrec.recall.vector_scoring import candidates_from_vectors
from hmrec.vectors import UserItemVectors

TWOTOWER_SPEC = OutputSpec(channel="twotower")

LOGGER = get_logger("recall.twotower.infer")


@dataclass(frozen=True)
class TwoTowerResources:
    """A loaded checkpoint with the tables it reads, all on one device.

    Attributes:
        model: The model in eval mode.
        features: Item feature table (padding row last).
        profiles: Profile codes int32 [n_customers, 3].
        n_articles: Real articles (= padding index of histories).
        device: Where everything lives.
        identity: The code vocabularies the model was trained with (train.model_identity);
            stored again in every chain checkpoint.
    """

    model: TwoTowerModel
    features: DeviceItemFeatures
    profiles: torch.Tensor
    n_articles: int
    device: torch.device
    identity: dict[str, Any]


def load_twotower_resources(
    paths: ProjectPaths, preprocess: PreprocessConfig, config: TwoTowerConfig
) -> TwoTowerResources:
    """Build the model, load checkpoint config.base_epoch, move it and its tables to the device.

    Raises:
        ValueError: If the checkpoint was trained with other article / profile codes.
    """
    metadata_file = paths.processed_metadata_file
    customers = scan_processed_parquet(paths.customers_file, metadata_file, preprocess).collect()
    articles = scan_processed_parquet(paths.articles_file, metadata_file, preprocess).collect()
    profiles = build_customer_profiles(customers)
    items = load_item_feature_arrays(paths, preprocess, articles)
    n_articles = articles.height
    model = build_two_tower_model(
        n_articles,
        preprocess.embeddings.image_dim,
        preprocess.embeddings.text_dim,
        items.codes,
        profiles,
        config.history_length,
        config.model,
        config.training.seed,
    )
    checkpoint_file = paths.twotower_checkpoint_file(config.base_epoch)
    metadata = load_checkpoint(model, checkpoint_file)
    expected = model_identity(n_articles, items.codes, profiles)
    check_model_identity(metadata["model_identity"], expected, checkpoint_file)
    device = torch.device(config.training.device)
    model = model.to(device).eval()
    return TwoTowerResources(
        model=model,
        features=items_to_device(items, device),
        profiles=profiles_to_device(profiles, device),
        n_articles=n_articles,
        device=device,
        identity=expected,
    )


def query_histories(
    inputs: RecallInputs, customer_idx: np.ndarray, history_length: int, n_articles: int
) -> tuple[np.ndarray, np.ndarray]:
    """History int32 [Q, L] and mask bool [Q, L] of each customer as of inputs.anchor.

    Args:
        inputs: The fold (anchor, transactions).
        customer_idx: int32 [Q], distinct query customers.
        history_length: L.
        n_articles: Padding index.
    """
    lines = load_customer_lines_before(
        inputs.transactions, customer_idx, inputs.anchor, inputs.day_zero
    )
    anchor_days = np.full(customer_idx.shape, inputs.anchor_day, dtype=np.int32)
    return build_histories(
        daily_purchases(lines), customer_idx, anchor_days, history_length, n_articles
    )


@torch.no_grad()
def encode_customers(
    resources: TwoTowerResources,
    customer_idx: np.ndarray,
    history: np.ndarray,
    history_mask: np.ndarray,
    chunk_size: int,
) -> np.ndarray:
    """User vectors float32 [Q, D], computed in chunks of `chunk_size` customers."""
    device = resources.device
    parts: list[np.ndarray] = []
    for start in range(0, customer_idx.size, chunk_size):
        stop = start + chunk_size
        customers = torch.from_numpy(customer_idx[start:stop].astype(np.int64)).to(device)
        chunk_history = torch.from_numpy(history[start:stop].astype(np.int64)).to(device)
        chunk_mask = torch.from_numpy(history_mask[start:stop]).to(device)
        profile = resources.profiles[customers].long()
        vectors = resources.model.encode_users(
            chunk_history, chunk_mask, profile, resources.features
        )
        parts.append(vectors.cpu().numpy())
    return np.concatenate(parts).astype(np.float32, copy=False)


@torch.no_grad()
def encode_articles(
    resources: TwoTowerResources, article_idx: np.ndarray, chunk_size: int
) -> np.ndarray:
    """Item vectors float32 [n, D] of the given real articles, in chunks."""
    parts: list[np.ndarray] = []
    for start in range(0, article_idx.size, chunk_size):
        chunk = torch.from_numpy(article_idx[start : start + chunk_size].astype(np.int64))
        vectors = resources.model.item_tower(chunk.to(resources.device), resources.features)
        parts.append(vectors.cpu().numpy())
    return np.concatenate(parts).astype(np.float32, copy=False)


@dataclass(frozen=True)
class TwoTowerChunks:
    """Chunk sizes of R6 prediction (configs/base.yaml, chunking). They change memory, not results.

    Attributes:
        encode_queries: Customers (or articles) per model forward pass
            (chunking.dense_score_query_chunk_size).
        score_queries: Queries per chunk of the user x active-article score
            matrix (chunking.twotower_score_query_chunk_size).
        pairs: (customer, article) pairs per dot_pairs chunk (chunking.pair_chunk_size).
    """

    encode_queries: int
    score_queries: int
    pairs: int


def query_vectors(
    config: TwoTowerConfig,
    resources: TwoTowerResources,
    queries: pl.DataFrame,
    inputs: RecallInputs,
    chunk_size: int,
) -> UserItemVectors:
    """User vectors of the query customers and item vectors of A(a), with the current model."""
    customers = np.unique(queries[CUSTOMER_IDX].to_numpy()).astype(np.int32)
    history, history_mask = query_histories(
        inputs, customers, config.history_length, resources.n_articles
    )
    log_memory(LOGGER, f"{inputs.anchor}: histories of {customers.size} customers")
    user_vectors = encode_customers(resources, customers, history, history_mask, chunk_size)
    pool = np.unique(inputs.active_pool[ARTICLE_IDX].to_numpy()).astype(np.int32)
    item_vectors = encode_articles(resources, pool, chunk_size)
    log_memory(LOGGER, f"{inputs.anchor}: vectors of {customers.size} users, {pool.size} articles")
    return UserItemVectors(customers, user_vectors, pool, item_vectors)


def add_other_catalog_articles(
    vectors: UserItemVectors, resources: TwoTowerResources, chunk_size: int
) -> UserItemVectors:
    """Add the item vectors of every catalog article that `vectors` does not have yet.

    query_vectors encodes only the active pool A(a), which is all that R6 needs
    for its own candidates. The tt_dot feature (proposal §8.3) is computed for
    EVERY candidate, and a candidate of another channel (R2) can be outside
    A(a), so the saved vectors must cover the whole catalog.

    The rows that are already there are kept as they are (not encoded again),
    so scoring with the result gives exactly the same R6 candidates and scores.

    Returns:
        Vectors with the same users and one item row per catalog article
        (article_idx 0 .. n_articles - 1, strictly increasing).
    """
    all_articles = np.arange(resources.n_articles, dtype=np.int32)
    missing = np.setdiff1d(all_articles, vectors.item_article_idx).astype(np.int32)
    if missing.size == 0:
        return vectors
    missing_vectors = encode_articles(resources, missing, chunk_size)
    article_idx = np.concatenate([vectors.item_article_idx, missing])
    item_vectors = np.concatenate([vectors.item_vectors, missing_vectors])
    order = np.argsort(article_idx, kind="stable")
    return UserItemVectors(
        user_customer_idx=vectors.user_customer_idx,
        user_vectors=vectors.user_vectors,
        item_article_idx=article_idx[order],
        item_vectors=np.ascontiguousarray(item_vectors[order]),
    )


def candidates_from_query_vectors(
    config: TwoTowerConfig,
    queries: pl.DataFrame,
    inputs: RecallInputs,
    vectors: UserItemVectors,
    chunks: TwoTowerChunks,
) -> pl.DataFrame:
    """Top-quota of A(a) per query from already computed vectors.

    `vectors` may hold more articles than A(a) (add_other_catalog_articles);
    only the articles in inputs.active_pool can become candidates.
    """
    return candidates_from_vectors(
        queries,
        vectors,
        inputs.active_pool,
        config.quota,
        chunks.score_queries,
        chunks.pairs,
        TWOTOWER_SPEC,
    )


def twotower_candidates(
    config: TwoTowerConfig,
    resources: TwoTowerResources,
    queries: pl.DataFrame,
    inputs: RecallInputs,
    chunks: TwoTowerChunks,
) -> pl.DataFrame:
    """Top-quota of A(a) per query with the CURRENT model (no training here).

    Args:
        config: R6 settings (history length, quota).
        resources: Model and tables.
        queries: One row per query; columns query_id, customer_idx.
        inputs: The fold.
        chunks: Chunk sizes from configs/base.yaml.
    """
    if queries.height == 0 or inputs.active_pool.height == 0:
        return TWOTOWER_SPEC.schema.to_frame()
    vectors = query_vectors(config, resources, queries, inputs, chunks.encode_queries)
    candidates = candidates_from_query_vectors(config, queries, inputs, vectors, chunks)
    log_memory(LOGGER, f"{inputs.anchor}: {candidates.height} candidates")
    return candidates


class TwoTowerChannel:
    """R6 recall channel with a fixed checkpoint ("frozen"), see the module docstring."""

    def __init__(
        self,
        config: TwoTowerConfig,
        resources: TwoTowerResources,
        chunks: TwoTowerChunks,
    ) -> None:
        """Store the config, the loaded model and the chunk sizes (configs/base.yaml)."""
        self.config = config
        self.resources = resources
        self.chunks = chunks

    @property
    def output_spec(self) -> OutputSpec:
        """R6 output: base columns only, default order."""
        return TWOTOWER_SPEC

    def vectors(self, queries: pl.DataFrame, inputs: RecallInputs) -> UserItemVectors:
        """User vectors of the query customers and item vectors of A(a)."""
        return query_vectors(
            self.config, self.resources, queries, inputs, self.chunks.encode_queries
        )

    def generate(self, queries: pl.DataFrame, inputs: RecallInputs) -> pl.DataFrame:
        """R6 candidates for `queries` (Top-quota of A(a) per query)."""
        return twotower_candidates(self.config, self.resources, queries, inputs, self.chunks)
