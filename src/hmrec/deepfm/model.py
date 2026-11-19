"""The DeepFM model (proposal §16.2, §16.3).

For a row with field embeddings e_1 ... e_F (each of length d):

    first order    b + sum_f w_f                       one weight per (field, code)
    second order   1/2 * sum_k [ (sum_f e_fk)^2 - sum_f e_fk^2 ]
                   = sum over all pairs f < g of <e_f, e_g>   (the FM identity)
    deep           MLP(concat(e_1 ... e_F)) -> 1
    logit          = FM logit (first + second order) + deep logit

`components` switches the parts: "first_order", "fm" (first + second), "deep"
(the MLP only), "full".

A code field gets its embedding and its first-order weight from two lookup
tables that hold all fields one after another (row = the field's offset + the
code). A value field (standardised numeric input) has one vector v and one
vector u: e = z * v + m * u, first-order term w * z + c * m, where z is the
standardised value and m the missing indicator.

Shapes: codes int64 [B, n_code_fields]; values / missing float32
[B, n_value_fields]; output float32 [B].
"""

import torch
from torch import nn

from hmrec.config import DeepFMModelConfig
from hmrec.deepfm.inputs import FieldLayout

FIRST_ORDER = "first_order"
FM = "fm"
DEEP = "deep"
FULL = "full"


def fm_second_order(embeddings: torch.Tensor) -> torch.Tensor:
    """Sum of the dot products of all pairs of field embeddings.

    Uses 1/2 * sum_k [(sum_f e_fk)^2 - sum_f e_fk^2], which equals
    sum_{f < g} <e_f, e_g> without looping over the pairs.

    Args:
        embeddings: float32 [B, F, d].

    Returns:
        float32 [B].
    """
    sum_of_fields = embeddings.sum(dim=1)  # [B, d]
    sum_of_squares = (embeddings * embeddings).sum(dim=1)  # [B, d]
    return 0.5 * (sum_of_fields * sum_of_fields - sum_of_squares).sum(dim=1)


class DeepFM(nn.Module):
    """DeepFM over code fields and (optionally) value fields; see the module docstring."""

    def __init__(self, layout: FieldLayout, config: DeepFMModelConfig) -> None:
        """Build the model. Initial values depend on torch's global seed (set it first).

        Args:
            layout: The fields (codes per code field, number of value fields).
            config: Embedding size, MLP layers, dropout, components.
        """
        super().__init__()
        self.components = config.components
        self.n_value_fields = layout.n_value_fields
        dim = config.embedding_dim
        sizes = torch.tensor(layout.code_field_sizes, dtype=torch.int64)
        # Row of (field f, code c) in the two tables: offset[f] + c.
        offsets = torch.cumsum(sizes, dim=0) - sizes
        self.offsets: torch.Tensor
        self.register_buffer("offsets", offsets)
        n_codes = int(sizes.sum().item())
        self.embedding = nn.Embedding(n_codes, dim)
        self.first_order = nn.Embedding(n_codes, 1)
        self.bias = nn.Parameter(torch.zeros(1))
        self.value_vectors = nn.Parameter(torch.empty(layout.n_value_fields, dim))
        self.missing_vectors = nn.Parameter(torch.empty(layout.n_value_fields, dim))
        self.value_weights = nn.Parameter(torch.zeros(layout.n_value_fields))
        self.missing_weights = nn.Parameter(torch.zeros(layout.n_value_fields))
        layers: list[nn.Module] = []
        width = layout.n_fields * dim
        for units in config.hidden_units:
            layers += [nn.Linear(width, units), nn.ReLU(), nn.Dropout(config.dropout)]
            width = units
        layers.append(nn.Linear(width, 1))
        self.mlp = nn.Sequential(*layers)
        for table in (self.embedding.weight, self.value_vectors, self.missing_vectors):
            nn.init.normal_(table, mean=0.0, std=config.embedding_init_std)
        nn.init.zeros_(self.first_order.weight)

    def field_embeddings(
        self, codes: torch.Tensor, values: torch.Tensor | None, missing: torch.Tensor | None
    ) -> torch.Tensor:
        """The embedding of every field of every row: float32 [B, n_fields, d]."""
        parts = [self.embedding(codes + self.offsets)]
        if self.n_value_fields > 0:
            if values is None or missing is None:
                raise ValueError("this model has value fields: values and missing are needed")
            # e = z * v + m * u, for every value field.
            parts.append(
                values.unsqueeze(2) * self.value_vectors
                + missing.unsqueeze(2) * self.missing_vectors
            )
        return torch.cat(parts, dim=1)

    def first_order_logit(
        self, codes: torch.Tensor, values: torch.Tensor | None, missing: torch.Tensor | None
    ) -> torch.Tensor:
        """The bias plus the first-order weights of the row's codes (and values): float32 [B]."""
        logit = self.bias + self.first_order(codes + self.offsets).squeeze(2).sum(dim=1)
        if self.n_value_fields > 0 and values is not None and missing is not None:
            logit = logit + values @ self.value_weights + missing @ self.missing_weights
        return logit

    def forward(
        self,
        codes: torch.Tensor,
        values: torch.Tensor | None = None,
        missing: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """The logit of every row: float32 [B]."""
        if self.components == FIRST_ORDER:
            # No embedding is used by this variant: do not look them up at all.
            return self.first_order_logit(codes, values, missing)
        embeddings = self.field_embeddings(codes, values, missing)
        logit = torch.zeros(codes.shape[0], dtype=embeddings.dtype, device=embeddings.device)
        if self.components in (FM, FULL):
            logit = logit + self.first_order_logit(codes, values, missing)
        if self.components in (FM, FULL):
            logit = logit + fm_second_order(embeddings)
        if self.components in (DEEP, FULL):
            logit = logit + self.mlp(embeddings.flatten(start_dim=1)).squeeze(1)
        return logit
