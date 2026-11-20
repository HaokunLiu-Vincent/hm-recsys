"""The two training losses of DeepFM (proposal §16.2, §16.3).

pointwise   binary cross-entropy of every row, mean over the batch:
                -[y log p + (1 - y) log(1 - p)],  p = sigmoid(logit)
listwise    softmax cross-entropy per query, mean over the batch's queries:
                loss_q = -(1 / |P_q|) * sum_{i in P_q} log softmax(s_q)_i
            where s_q are the logits of the query's rows and P_q its positive
            rows. A batch holds WHOLE queries only.
"""

import torch
from torch.nn import functional


def pointwise_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean binary cross-entropy.

    Args:
        logits: float32 [B].
        labels: float32 [B], 1 = positive.
    """
    return functional.binary_cross_entropy_with_logits(logits, labels)


def listwise_loss(
    logits: torch.Tensor, labels: torch.Tensor, query_index: torch.Tensor, n_queries: int
) -> torch.Tensor:
    """Mean over the queries of the softmax cross-entropy of their positive rows.

    Args:
        logits: float32 [B].
        labels: float32 [B], 1 = positive.
        query_index: int64 [B]: which query of the batch (0 ... n_queries - 1)
            each row belongs to. All rows of a query are in the batch.
        n_queries: Queries in the batch.

    Raises:
        ValueError: If a query of the batch has no positive row.
    """
    # log sum exp per query, shifted by the query's largest logit so that exp cannot overflow.
    largest = torch.full((n_queries,), -torch.inf, dtype=logits.dtype, device=logits.device)
    largest = largest.scatter_reduce(0, query_index, logits, reduce="amax")
    shifted = torch.exp(logits - largest[query_index])
    sums = torch.zeros(n_queries, dtype=logits.dtype, device=logits.device)
    log_sum_exp = largest + torch.log(sums.index_add(0, query_index, shifted))
    log_softmax = logits - log_sum_exp[query_index]
    zeros = torch.zeros(n_queries, dtype=logits.dtype, device=logits.device)
    n_positives = zeros.index_add(0, query_index, labels)
    if bool((n_positives == 0).any()):
        raise ValueError("a query of the batch has no positive row")
    positive_log_likelihood = zeros.index_add(0, query_index, labels * log_softmax)
    return (-positive_log_likelihood / n_positives).mean()
