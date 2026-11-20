"""Score candidates with a DeepFM model and keep the Top-K of every query (proposal §16.1).

Order inside a query: score DESC, then article_idx ASC (the project's tie-break).
The score is the model's logit (float32, stored as float64): the sigmoid would
not change the order.
"""

import numpy as np
import polars as pl
import torch

from hmrec.deepfm.inputs import EncodedRows
from hmrec.deepfm.model import DeepFM
from hmrec.deepfm.train import DeviceRows, rows_to_device

QUERY_ID = "query_id"
ARTICLE_IDX = "article_idx"
SCORE = "score"
RANK = "rank"


@torch.no_grad()
def score_device_rows(
    model: DeepFM, rows: DeviceRows, n_rows: int, rows_per_chunk: int, rows_per_block: int
) -> torch.Tensor:
    """The logit of the first `n_rows` rows that are already on the device: float32 [n_rows].

    Args:
        model: The model (put into eval mode here).
        rows: The encoded rows on the device.
        n_rows: Rows to score, from the first one.
        rows_per_chunk: Rows per forward pass.
        rows_per_block: Rows per block inside a forward pass, for building the
            field embeddings (hmrec.deepfm.model: memory only, the scores do
            not depend on it).
    """
    if rows_per_chunk <= 0:
        raise ValueError(f"rows_per_chunk must be > 0, got {rows_per_chunk}")
    if rows_per_block <= 0:
        raise ValueError(f"rows_per_block must be > 0, got {rows_per_block}")
    model.eval()
    device = rows.codes.device
    scores = []
    for start in range(0, n_rows, rows_per_chunk):
        chunk = torch.arange(start, min(start + rows_per_chunk, n_rows), device=device)
        scores.append(model(*rows.batch(chunk), block_rows=rows_per_block))
    if not scores:
        return torch.zeros(0, dtype=torch.float32, device=device)
    return torch.cat(scores).float()


def score_rows(
    model: DeepFM,
    rows: EncodedRows,
    rows_per_chunk: int,
    rows_per_block: int,
    device: torch.device,
) -> np.ndarray:
    """The logit of every row, computed in chunks of `rows_per_chunk` rows.

    The model has no batch statistics, so a row's score does not depend on the
    other rows of its chunk as a formula. In floating point the chunk size can
    still move a score by about 1e-7 (the first-order matrix-vector product of
    value fields), so it is a fixed setting. `rows_per_block` changes memory
    only: the scores are bit-identical for every block size.

    Returns:
        float32 [n_rows].
    """
    on_device = rows_to_device(rows, device)
    scores = score_device_rows(model, on_device, rows.n_rows, rows_per_chunk, rows_per_block)
    return scores.cpu().numpy().astype(np.float32, copy=False)


def top_k_per_query(
    query_ids: np.ndarray, article_idx: np.ndarray, scores: np.ndarray, top_k: int
) -> pl.DataFrame:
    """Rank every query's candidates and keep the Top-`top_k`.

    Args:
        query_ids: int32 [n_rows].
        article_idx: int32 [n_rows], unique within a query.
        scores: float32 [n_rows].
        top_k: Rows kept per query (fewer if the query has fewer candidates).

    Returns:
        One row per kept (query_id, article_idx). Columns: query_id (int32),
        article_idx (int32), score (float64), rank (int32, 1 = best); sorted by
        (query_id, rank).

    Raises:
        ValueError: If top_k <= 0 or a score is NaN (the order would not be defined).
    """
    if top_k <= 0:
        raise ValueError(f"top_k must be > 0, got {top_k}")
    if np.isnan(scores).any():
        raise ValueError("scores contain NaN: the order would not be defined")
    table = pl.DataFrame(
        {QUERY_ID: query_ids, ARTICLE_IDX: article_idx, SCORE: scores.astype(np.float64)},
        schema={QUERY_ID: pl.Int32, ARTICLE_IDX: pl.Int32, SCORE: pl.Float64},
    )
    ordered = table.sort([QUERY_ID, SCORE, ARTICLE_IDX], descending=[False, True, False])
    ranked = ordered.with_columns(
        pl.int_range(1, pl.len() + 1, dtype=pl.Int32).over(QUERY_ID).alias(RANK)
    )
    return ranked.filter(pl.col(RANK) <= top_k)
