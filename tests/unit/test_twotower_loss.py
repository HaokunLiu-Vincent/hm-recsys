"""Tests for recall.twotower.loss: duplicate columns, LogQ, same-group masking (hand-computed)."""

import math

import numpy as np
import pytest
import torch

from hmrec.recall.twotower.loss import (
    article_log_q,
    multi_positive_logq_loss,
    other_positive_mask,
)


def test_article_log_q_counts_targets() -> None:
    # Targets 1, 1, 2, 5 -> Q = 2/4, 1/4, 1/4; never-target rows get 0.
    log_q = article_log_q(np.array([1, 1, 2, 5]), n_rows=6)
    expected = [0.0, math.log(0.5), math.log(0.25), 0.0, 0.0, math.log(0.25)]
    assert log_q.tolist() == pytest.approx(expected)
    assert log_q.dtype == np.float32


def test_other_positive_mask() -> None:
    # Samples: s0 (group 0, column 0), s1 (group 0, column 1), s2 (group 1, column 0).
    mask = other_positive_mask(torch.tensor([0, 0, 1]), torch.tensor([0, 1, 0]), n_columns=2)
    assert mask.tolist() == [[False, True], [True, False], [False, False]]


def test_loss_hand_computed_with_duplicates_logq_and_groups() -> None:
    # Samples s0 = (group 0, article 10), s1 = (group 0, article 11), s2 = (group 1, article 10).
    # Duplicate-aware columns: article 10 -> column 0 (targets of s0 and s2), 11 -> column 1.
    # v10 = (1, 0), v11 = (0, 1); u0 = (1, 0), u1 = u2 = (0, 1); tau = 1;
    # log Q = (ln 0.5, ln 0.25).
    #   row 0: column 1 is another positive of group 0 -> hidden; only its target -> loss 0
    #   row 1: column 0 hidden -> loss 0
    #   row 2: logits (0 - ln 0.5, 1 - ln 0.25) = (ln 2, 1 + ln 4); target column 0
    #          loss = ln(1 + e^(1 + ln 4 - ln 2)) = ln(1 + 2e)
    #   mean = ln(1 + 2e) / 3
    users = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    columns = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    loss = multi_positive_logq_loss(
        users,
        columns,
        target_column=torch.tensor([0, 1, 0]),
        group_id=torch.tensor([0, 0, 1]),
        column_log_q=torch.tensor([math.log(0.5), math.log(0.25)]),
        temperature=1.0,
    )
    assert loss.item() == pytest.approx(math.log(1 + 2 * math.e) / 3, rel=1e-6)


def test_logq_lowers_the_pull_of_popular_negatives() -> None:
    # One user, two columns with equal dot products; the target is column 0.
    # Without LogQ the loss is ln 2. If the NEGATIVE (column 1) is popular
    # (log Q larger), its logit drops and the loss gets smaller.
    users = torch.tensor([[1.0, 0.0]])
    columns = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    args = {"target_column": torch.tensor([0]), "group_id": torch.tensor([0]), "temperature": 1.0}
    flat = multi_positive_logq_loss(users, columns, column_log_q=torch.zeros(2), **args)
    popular_negative = multi_positive_logq_loss(
        users, columns, column_log_q=torch.tensor([math.log(0.1), math.log(0.9)]), **args
    )
    assert flat.item() == pytest.approx(math.log(2))
    # logits (-ln .1, -ln .9) -> loss = ln(1 + e^(ln .1 - ln .9)) = ln(1 + 1/9)
    assert popular_negative.item() == pytest.approx(math.log(1 + 1 / 9), rel=1e-6)
