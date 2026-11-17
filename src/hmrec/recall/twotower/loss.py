"""Multi-positive, duplicate-aware in-batch softmax with LogQ correction (proposal §5 R6).

For a batch of B samples whose targets cover K DISTINCT articles:

    columns      = the K distinct target articles (duplicate-aware: an article
                   bought by several samples of the batch is ONE column, so it is
                   never counted twice as a negative)
    logit[i, k]  = (u_i . v_k) / tau - log Q(k)          for every column k
    Q(k)         = (training samples whose target is k) / (training samples)
    other positives of row i: columns that are the target of another sample of
                   the SAME group (same customer, same day) -> logit = -inf, so
                   they are neither the target nor a negative of row i
    loss         = mean over rows i of cross-entropy(logit[i, :], target column of i)

Q(k) is the chance that a random batch slot holds article k when the training
samples are shuffled uniformly; subtracting log Q(k) from every column (the
target included) removes the popularity bias of in-batch negatives (Yi et al.
2019, "sampling-bias-corrected"). Q is counted on the CURRENT training set only:
all base samples for the base model; the new week + replay for a fine-tune.
"""

import numpy as np
import torch
import torch.nn.functional as functional


def article_log_q(target_article_idx: np.ndarray, n_rows: int) -> np.ndarray:
    """Return log Q(k) for every article row of the feature table.

    Args:
        target_article_idx: int [N], the target article of each training sample.
        n_rows: Rows of the item feature table (n_articles + 1).

    Returns:
        float32 [n_rows]. Rows never used as a target get 0.0: they can never be
        a loss column (columns are targets of the batch), so the value is unused.
    """
    if target_article_idx.size == 0:
        raise ValueError("no training samples: cannot estimate Q(k)")
    counts = np.bincount(target_article_idx, minlength=n_rows).astype(np.float64)
    if counts.size != n_rows:
        raise ValueError(f"article_idx {target_article_idx.max()} is outside {n_rows} rows")
    log_q = np.zeros(n_rows, dtype=np.float64)
    seen = counts > 0
    # float64 for the division and log (CPU, small); stored as float32 for the device.
    log_q[seen] = np.log(counts[seen] / target_article_idx.size)
    return log_q.astype(np.float32)


def other_positive_mask(
    group_id: torch.Tensor, target_column: torch.Tensor, n_columns: int
) -> torch.Tensor:
    """Mask (bool [B, K]): True where column k is the target of ANOTHER article of row i's group.

    Args:
        group_id: int64 [B].
        target_column: int64 [B], column of each sample's target.
        n_columns: K.
    """
    same_group = (group_id.unsqueeze(1) == group_id.unsqueeze(0)).float()  # [B, B]
    target_one_hot = functional.one_hot(target_column, n_columns).float()  # [B, K]
    # reached[i, k] > 0 if some sample j of i's group (j = i included) has target k.
    reached = same_group @ target_one_hot > 0
    return reached & ~target_one_hot.bool()


def multi_positive_logq_loss(
    users: torch.Tensor,
    columns: torch.Tensor,
    target_column: torch.Tensor,
    group_id: torch.Tensor,
    column_log_q: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """The loss of the module docstring (a scalar).

    Args:
        users: float32 [B, D], L2-normalised.
        columns: float32 [K, D], L2-normalised vectors of the distinct targets.
        target_column: int64 [B].
        group_id: int64 [B].
        column_log_q: float32 [K], log Q of each column's article.
        temperature: tau.
    """
    logits = users @ columns.T / temperature - column_log_q.unsqueeze(0)  # [B, K]
    hidden = other_positive_mask(group_id, target_column, columns.shape[0])
    logits = logits.masked_fill(hidden, float("-inf"))
    return functional.cross_entropy(logits, target_column)
