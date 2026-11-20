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

Scoring in blocks (memory only; proposal §16.10). Building the tensor of all
field embeddings, float32 [B, F, d], is what a forward pass needs most memory
for: with the standardised input three more tensors of almost that size exist
while it is built, and the second-order term squares it. forward(...,
block_rows=n) builds the embeddings and the second-order term for n rows at a
time. These are element-wise operations, look-ups and sums over a row's own
fields, so every row's numbers are the same bits as in one pass
(tests/unit/test_deepfm_model.py).

The two matrix products are NOT split: the MLP and, for value fields, the
first-order term (values @ weights). The last bits of a matrix product depend on
how many rows the call holds (measured on this machine: a Linear layer gives
other bits for the same row in a call of 1 row, and Linear(64, 1) still in a
call of 1,000 rows). So both are always computed in ONE call for the whole
batch, on the same input as without blocks; the MLP reads the block-built
embeddings from one tensor of the full size.
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
        self.n_fields = layout.n_fields
        dim = config.embedding_dim
        self.embedding_dim = dim
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

    def embedding_logits(
        self, codes: torch.Tensor, values: torch.Tensor | None, missing: torch.Tensor | None
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """The two terms computed from the field embeddings: (second order, deep).

        Each is float32 [B], or None if the model's components do not include it.
        """
        embeddings = self.field_embeddings(codes, values, missing)
        second_order = fm_second_order(embeddings) if self.components in (FM, FULL) else None
        deep = None
        if self.components in (DEEP, FULL):
            deep = self.mlp(embeddings.flatten(start_dim=1)).squeeze(1)
        return second_order, deep

    def embedding_logits_in_blocks(
        self,
        codes: torch.Tensor,
        values: torch.Tensor | None,
        missing: torch.Tensor | None,
        block_rows: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """embedding_logits with the embeddings built `block_rows` rows at a time (memory only).

        The field embeddings and the second-order term are computed block by
        block; the MLP runs ONCE on all rows, as in embedding_logits (module
        docstring). The results are bit-identical to embedding_logits. For
        scoring only (no gradient is needed and dropout is off).

        Raises:
            ValueError: If block_rows <= 0.
        """
        if block_rows <= 0:
            raise ValueError(f"block_rows must be > 0, got {block_rows}")
        n_rows = int(codes.shape[0])
        if n_rows <= block_rows:
            return self.embedding_logits(codes, values, missing)
        with_second_order = self.components in (FM, FULL)
        with_deep = self.components in (DEEP, FULL)
        # The MLP needs every row's embeddings in one tensor: float32 [B, F, d].
        embeddings = None
        if with_deep:
            shape = (n_rows, self.n_fields, self.embedding_dim)
            embeddings = torch.empty(shape, dtype=self.bias.dtype, device=codes.device)
        second_orders: list[torch.Tensor] = []
        for start in range(0, n_rows, block_rows):
            stop = start + block_rows
            block = self.field_embeddings(
                codes[start:stop],
                None if values is None else values[start:stop],
                None if missing is None else missing[start:stop],
            )
            if with_second_order:
                second_orders.append(fm_second_order(block))
            if embeddings is not None:
                embeddings[start:stop] = block
        second_order = torch.cat(second_orders) if with_second_order else None
        deep = None
        if embeddings is not None:
            # One call for all rows: the same matrix products as without blocks.
            deep = self.mlp(embeddings.flatten(start_dim=1)).squeeze(1)
        return second_order, deep

    def forward(
        self,
        codes: torch.Tensor,
        values: torch.Tensor | None = None,
        missing: torch.Tensor | None = None,
        block_rows: int | None = None,
    ) -> torch.Tensor:
        """The logit of every row: float32 [B].

        Args:
            codes: int64 [B, n_code_fields].
            values: float32 [B, n_value_fields], or None without value fields.
            missing: float32 [B, n_value_fields], or None.
            block_rows: None (training): one pass over all rows. A number
                (scoring): the field embeddings and the second-order term are
                built for that many rows at a time, with the same result
                (module docstring).
        """
        if self.components == FIRST_ORDER:
            # No embedding is used by this variant: do not look them up at all.
            return self.first_order_logit(codes, values, missing)
        if block_rows is None:
            second_order, deep = self.embedding_logits(codes, values, missing)
        else:
            second_order, deep = self.embedding_logits_in_blocks(codes, values, missing, block_rows)
        # The terms are added in a fixed order: first order, second order, deep.
        logit = torch.zeros(codes.shape[0], dtype=self.bias.dtype, device=codes.device)
        if self.components in (FM, FULL):
            # One call for the whole batch, never in blocks (module docstring).
            logit = logit + self.first_order_logit(codes, values, missing)
        if second_order is not None:
            logit = logit + second_order
        if deep is not None:
            logit = logit + deep
        return logit
