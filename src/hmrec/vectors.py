"""User and article vectors shared by R5 WALS and R6 Two-Tower (proposal §5, §8.3).

Both models give each user and each article a vector; the score of a
(user, article) pair is the dot product u . v. The same numbers are needed in
two places:

    - recall (R5 / R6): the channel's `score` column;
    - features (§8.3): wals_dot / tt_dot for every candidate.

recall/ and features/ may not import each other (AGENTS.md §5), so the vectors
live in this lower-layer module: recall/ fits and saves them, features/ loads
them, and both compute dot products with `dot_pairs`. Using one function for
both guarantees that a candidate's R5 score equals its wals_dot exactly.

Files (one directory per model and fold, e.g. artifacts/models/wals/{fold}/):
    user_customer_idx.npy  int32 [n_users], strictly increasing
    user_vectors.npy       float32 [n_users, dim]
    item_article_idx.npy   int32 [n_items], strictly increasing
    item_vectors.npy       float32 [n_items, dim]
    vectors.json           source ("wals" / "twotower"), fold, dim, row counts
A saved file may hold only some users (e.g. the fold's query customers); a user
or article that is not stored has no vector, and its dot product is NaN.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

VECTOR_SOURCES = ("wals", "twotower")

USER_INDEX_FILE = "user_customer_idx.npy"
USER_VECTORS_FILE = "user_vectors.npy"
ITEM_INDEX_FILE = "item_article_idx.npy"
ITEM_VECTORS_FILE = "item_vectors.npy"
METADATA_FILE = "vectors.json"


# ---------------------------------------------------------------- the vectors


def _check_index(name: str, index: np.ndarray) -> None:
    """An index must be a 1-D, strictly increasing int32 array (so lookups can use searchsorted)."""
    if index.dtype != np.int32 or index.ndim != 1:
        raise ValueError(
            f"{name} must be a 1-D int32 array, got {index.dtype} with shape {index.shape}"
        )
    if index.size > 1 and not bool(np.all(np.diff(index) > 0)):
        raise ValueError(f"{name} must be strictly increasing (sorted, no duplicates)")


def _check_matrix(name: str, matrix: np.ndarray, n_rows: int) -> None:
    """A vector matrix must be 2-D float32, with one finite row per index entry."""
    if matrix.dtype != np.float32 or matrix.ndim != 2:
        raise ValueError(f"{name} must be a 2-D float32 array, got {matrix.dtype} {matrix.shape}")
    if matrix.shape[0] != n_rows:
        raise ValueError(f"{name} has {matrix.shape[0]} rows, but its index has {n_rows} entries")
    if not bool(np.all(np.isfinite(matrix))):
        raise ValueError(f"{name} contains NaN or infinite values")


@dataclass(frozen=True)
class UserItemVectors:
    """User vectors and article vectors of one model for one fold.

    Row r of user_vectors belongs to customer user_customer_idx[r]; row r of
    item_vectors belongs to article item_article_idx[r].

    Attributes:
        user_customer_idx: int32 [n_users], strictly increasing.
        user_vectors: float32 [n_users, dim].
        item_article_idx: int32 [n_items], strictly increasing.
        item_vectors: float32 [n_items, dim].
    """

    user_customer_idx: np.ndarray
    user_vectors: np.ndarray
    item_article_idx: np.ndarray
    item_vectors: np.ndarray

    def __post_init__(self) -> None:
        """Check dtypes, shapes and index order (fail early, AGENTS.md §6.5)."""
        _check_index("user_customer_idx", self.user_customer_idx)
        _check_index("item_article_idx", self.item_article_idx)
        _check_matrix("user_vectors", self.user_vectors, self.user_customer_idx.size)
        _check_matrix("item_vectors", self.item_vectors, self.item_article_idx.size)
        if self.user_vectors.shape[1] != self.item_vectors.shape[1]:
            raise ValueError(
                f"user dim {self.user_vectors.shape[1]} != item dim {self.item_vectors.shape[1]}"
            )

    @property
    def dim(self) -> int:
        """Vector length."""
        return int(self.user_vectors.shape[1])


def lookup_rows(index: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """Row of each key in a sorted index; -1 where the key is not in the index.

    Args:
        index: Strictly increasing int32 array (e.g. user_customer_idx).
        keys: Any int array, shape [n].

    Returns:
        int64 array, shape [n].
    """
    if index.size == 0:
        return np.full(keys.shape, -1, dtype=np.int64)
    positions = np.searchsorted(index, keys)
    # searchsorted returns index.size for keys above the last entry; clip before comparing.
    clipped = np.minimum(positions, index.size - 1)
    found = index[clipped] == keys
    return np.where(found, clipped, -1).astype(np.int64)


def select_users(vectors: UserItemVectors, keep_customer_idx: np.ndarray) -> UserItemVectors:
    """Keep only the users in `keep_customer_idx` (all articles are kept).

    Customers in `keep_customer_idx` without a vector are simply absent.
    """
    keep = np.isin(vectors.user_customer_idx, keep_customer_idx)
    return UserItemVectors(
        user_customer_idx=vectors.user_customer_idx[keep],
        user_vectors=vectors.user_vectors[keep],
        item_article_idx=vectors.item_article_idx,
        item_vectors=vectors.item_vectors,
    )


# ---------------------------------------------------------------- dot products


def dot_pairs(
    vectors: UserItemVectors,
    customer_idx: np.ndarray,
    article_idx: np.ndarray,
    chunk_size: int,
) -> np.ndarray:
    """Dot product u . v for each (customer_idx[k], article_idx[k]) pair; NaN if no vector.

    This is THE dot product of the project: R5 / R6 scores and the §8.3
    wals_dot / tt_dot features all come from here, so they are identical.
    Each pair is a float32 sum of `dim` products (np.einsum), computed the same
    way whatever the chunk size or the other pairs in the call.

    Args:
        vectors: The model's vectors.
        customer_idx: int array, shape [n_pairs].
        article_idx: int array, shape [n_pairs].
        chunk_size: Pairs per chunk (chunking.pair_chunk_size); bounds the
            memory of the gathered rows to 2 * chunk_size * dim * 4 bytes.

    Returns:
        float32 array, shape [n_pairs].
    """
    if customer_idx.shape != article_idx.shape or customer_idx.ndim != 1:
        raise ValueError(
            f"customer_idx {customer_idx.shape} and article_idx {article_idx.shape} "
            "must be 1-D arrays of the same length"
        )
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    result = np.full(customer_idx.size, np.nan, dtype=np.float32)
    for start in range(0, customer_idx.size, chunk_size):
        stop = start + chunk_size
        user_rows = lookup_rows(vectors.user_customer_idx, customer_idx[start:stop])
        item_rows = lookup_rows(vectors.item_article_idx, article_idx[start:stop])
        valid = (user_rows >= 0) & (item_rows >= 0)
        user_part = vectors.user_vectors[user_rows[valid]]
        item_part = vectors.item_vectors[item_rows[valid]]
        chunk_result = result[start:stop]  # a view: writing into it fills `result`
        chunk_result[valid] = np.einsum("ij,ij->i", user_part, item_part)
    return result


# ---------------------------------------------------------------- save / load


def save_vectors(
    vectors: UserItemVectors,
    directory: Path,
    source: str,
    fold: str,
    keep_customer_idx: np.ndarray | None,
    overwrite: bool,
) -> None:
    """Write the vectors and a vectors.json with their source and fold.

    Args:
        vectors: The model's vectors.
        directory: e.g. paths.model_dir("wals", fold).
        source: "wals" or "twotower".
        fold: Fold name, e.g. "F1".
        keep_customer_idx: Save only these users (e.g. the fold's query
            customers); None saves every user. Articles are always all saved.
        overwrite: If False, refuse to replace existing files.
    """
    if source not in VECTOR_SOURCES:
        raise ValueError(f"source must be one of {list(VECTOR_SOURCES)}, got {source!r}")
    metadata_path = directory / METADATA_FILE
    if metadata_path.exists() and not overwrite:
        raise FileExistsError(f"{metadata_path} exists; pass overwrite=True to replace it")
    to_save = vectors if keep_customer_idx is None else select_users(vectors, keep_customer_idx)
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / USER_INDEX_FILE, to_save.user_customer_idx)
    np.save(directory / USER_VECTORS_FILE, to_save.user_vectors)
    np.save(directory / ITEM_INDEX_FILE, to_save.item_article_idx)
    np.save(directory / ITEM_VECTORS_FILE, to_save.item_vectors)
    metadata = {
        "source": source,
        "fold": fold,
        "dim": to_save.dim,
        "n_users": int(to_save.user_customer_idx.size),
        "n_items": int(to_save.item_article_idx.size),
        "all_users_saved": keep_customer_idx is None,
    }
    # Written last: a directory with vectors.json is complete.
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))


def load_vectors(directory: Path, source: str, fold: str) -> UserItemVectors:
    """Read vectors saved by save_vectors and check that they are the expected ones.

    Raises:
        FileNotFoundError: If vectors.json is missing.
        ValueError: If the stored source or fold differs from the expected one,
            or the arrays do not match the stored row counts.
    """
    metadata = json.loads((directory / METADATA_FILE).read_text())
    if metadata["source"] != source or metadata["fold"] != fold:
        raise ValueError(
            f"{directory} holds {metadata['source']} vectors of fold {metadata['fold']}, "
            f"expected {source} / {fold}"
        )
    vectors = UserItemVectors(
        user_customer_idx=np.load(directory / USER_INDEX_FILE, allow_pickle=False),
        user_vectors=np.load(directory / USER_VECTORS_FILE, allow_pickle=False),
        item_article_idx=np.load(directory / ITEM_INDEX_FILE, allow_pickle=False),
        item_vectors=np.load(directory / ITEM_VECTORS_FILE, allow_pickle=False),
    )
    if vectors.user_customer_idx.size != metadata["n_users"]:
        raise ValueError(
            f"{directory}: {vectors.user_customer_idx.size} users, "
            f"metadata says {metadata['n_users']}"
        )
    if vectors.item_article_idx.size != metadata["n_items"]:
        raise ValueError(
            f"{directory}: {vectors.item_article_idx.size} items, "
            f"metadata says {metadata['n_items']}"
        )
    return vectors
