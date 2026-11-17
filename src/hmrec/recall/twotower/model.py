"""ItemTower, UserTower and the TwoTowerModel around them (proposal §5 R6).

All sizes come from configs/recall/r6_twotower.yaml (model section). With the
current data and config:

Item tower, one article -> concatenation of
    image          Linear(512 -> 128), output x image_mask (a missing image gives 0,
                   bias included)                                              128
    text           Linear(384 -> 128), output x text_mask                     128
    article ID     Embedding(n_articles + 1, 64, padding_idx = n_articles), init 0  64
    product_code   Embedding(47,225, 32, padding_idx = 0), init 0              32
    product_type_no      133 codes -> Embedding(133, 16)                       16
    product_group_name    20 codes -> one-hot                                  20
    garment_group_no      22 codes -> Embedding(22, 16)                        16
    index_group_no         6 codes -> one-hot                                   6
    section_no            58 codes -> Embedding(58, 16)                        16
    colour_group_code     51 codes -> Embedding(51, 16)                        16
                                                                     total    442
  -> Linear(442 -> 256) -> ReLU -> Linear(256 -> 128) -> L2-normalise.

Category rule (both towers): a field with <= one_hot_max_cardinality (20) codes,
the missing code 0 included, is one-hot; a larger field uses an Embedding of
category_embedding_dim (16), padding_idx = 0, init N(0, 0.02^2). Code counts are
taken from the data (CategoryCodes.cardinalities), so this table is an example.

User tower:
    history: the 20 history articles go through the SAME item tower (padding
             slots are never encoded); pooled = sum_j a_j h_j, where
             a = softmax of 20 learned position logits over the FILLED slots
             only (logits start at 0, so training starts from a masked mean);
             an empty history gives pooled = 0.
    profile: age_bucket (8 codes), club_member_status (4), fashion_news_frequency
             (4) -> one-hot each -> 16.
    [pooled 128 ; profile 16] = 144 -> Linear(144 -> 256) -> ReLU
    -> Linear(256 -> 128) -> L2-normalise.

Each article is encoded once per step: the model collects the distinct articles
of a batch (targets + filled history slots), runs the item tower on them, and
gathers the results back by index.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import nn

from hmrec.config import TwoTowerModelConfig
from hmrec.recall.twotower.data import Batch, CategoryCodes, DeviceItemFeatures

# Smallest denominator of the position weights; an empty history has total 0,
# and 0 / 1e-12 = 0 keeps the pooled vector at 0 without NaN in the gradient.
EMPTY_HISTORY_EPSILON = 1e-12


class CategoryEncoder(nn.Module):
    """One category field: one-hot if it has few codes, else an embedding (see module docstring)."""

    def __init__(
        self,
        cardinality: int,
        one_hot_max_cardinality: int,
        embedding_dim: int,
        init_std: float,
    ) -> None:
        """Choose the encoding from the number of codes (code 0 included).

        Args:
            cardinality: Number of codes, 0..cardinality-1.
            one_hot_max_cardinality: Largest cardinality that is one-hot coded.
            embedding_dim: Width of the embedding for larger fields.
            init_std: Std of the normal init of the embedding.
        """
        super().__init__()
        self.cardinality = cardinality
        self.embedding: nn.Embedding | None = None
        if cardinality <= one_hot_max_cardinality:
            self.output_dim = cardinality
            return
        self.output_dim = embedding_dim
        self.embedding = nn.Embedding(cardinality, embedding_dim, padding_idx=0)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=init_std)
        # normal_ also overwrote the padding row; padding must stay exactly 0.
        with torch.no_grad():
            self.embedding.weight[0].zero_()

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """int64 codes of any shape [...] -> float32 [..., output_dim]."""
        if self.embedding is None:
            return functional.one_hot(codes, self.cardinality).float()
        return self.embedding(codes)


def _two_layer_mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    """Linear(input -> hidden) -> ReLU -> Linear(hidden -> output)."""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim)
    )


class ItemTower(nn.Module):
    """Article -> 128-d L2-normalised vector (see the module docstring)."""

    def __init__(
        self,
        n_articles: int,
        image_dim: int,
        text_dim: int,
        article_codes: CategoryCodes,
        config: TwoTowerModelConfig,
    ) -> None:
        """Build the layers.

        Args:
            n_articles: Real articles; row n_articles of the feature table is padding.
            image_dim: Width of the CLIP vectors (512).
            text_dim: Width of the MiniLM vectors (384).
            article_codes: Code columns of the item table; column 0 must be
                product_code, the rest are category fields.
            config: Model sizes.
        """
        super().__init__()
        if article_codes.columns[0] != "product_code":
            raise ValueError(f"code column 0 must be product_code, got {article_codes.columns}")
        cardinalities = article_codes.cardinalities
        self.image_projection = nn.Linear(image_dim, config.image_projection_dim)
        self.text_projection = nn.Linear(text_dim, config.text_projection_dim)
        self.article_embedding = nn.Embedding(
            n_articles + 1, config.article_embedding_dim, padding_idx=n_articles
        )
        self.product_code_embedding = nn.Embedding(
            cardinalities[0], config.product_code_embedding_dim, padding_idx=0
        )
        # Zero init: an article (or product) never seen in training keeps a zero
        # ID part, so its vector comes from content and categories, not noise.
        nn.init.zeros_(self.article_embedding.weight)
        nn.init.zeros_(self.product_code_embedding.weight)
        encoders = [
            CategoryEncoder(
                size,
                config.one_hot_max_cardinality,
                config.category_embedding_dim,
                config.embedding_init_std,
            )
            for size in cardinalities[1:]
        ]
        self.category_encoders = nn.ModuleList(encoders)
        self.concat_dim = (
            config.image_projection_dim
            + config.text_projection_dim
            + config.article_embedding_dim
            + config.product_code_embedding_dim
            + sum(encoder.output_dim for encoder in encoders)
        )
        self.mlp = _two_layer_mlp(self.concat_dim, config.hidden_dim, config.output_dim)

    def forward(self, article_idx: torch.Tensor, features: DeviceItemFeatures) -> torch.Tensor:
        """int64 article_idx [N] -> float32 [N, output_dim], L2-normalised."""
        image_mask = features.image_mask[article_idx].unsqueeze(-1).float()
        text_mask = features.text_mask[article_idx].unsqueeze(-1).float()
        codes = features.codes[article_idx].long()  # [N, 7]
        parts = [
            # Multiplying the OUTPUT by the mask zeroes the bias too: a missing
            # image contributes nothing (proposal §5 R6: masked, not a zero input).
            self.image_projection(features.image[article_idx]) * image_mask,
            self.text_projection(features.text[article_idx]) * text_mask,
            self.article_embedding(article_idx),
            self.product_code_embedding(codes[:, 0]),
        ]
        for column, encoder in enumerate(self.category_encoders, start=1):
            parts.append(encoder(codes[:, column]))
        return functional.normalize(self.mlp(torch.cat(parts, dim=-1)), dim=-1)


class UserTower(nn.Module):
    """(history vectors, mask, profile) -> 128-d L2-normalised vector (see the module docstring)."""

    def __init__(
        self,
        history_length: int,
        item_dim: int,
        profile_codes: CategoryCodes,
        config: TwoTowerModelConfig,
    ) -> None:
        """Build the position logits, the profile encoders and the MLP.

        Args:
            history_length: L (20).
            item_dim: Width of the item vectors (128).
            profile_codes: Profile code columns (age_bucket, club_member_status,
                fashion_news_frequency).
            config: Model sizes.
        """
        super().__init__()
        # One logit per history position; 0 at start = equal weights (masked mean).
        self.position_logits = nn.Parameter(torch.zeros(history_length))
        encoders = [
            CategoryEncoder(
                size,
                config.one_hot_max_cardinality,
                config.category_embedding_dim,
                config.embedding_init_std,
            )
            for size in profile_codes.cardinalities
        ]
        self.profile_encoders = nn.ModuleList(encoders)
        self.profile_dim = sum(encoder.output_dim for encoder in encoders)
        self.mlp = _two_layer_mlp(item_dim + self.profile_dim, config.hidden_dim, config.output_dim)

    def position_weights(self, history_mask: torch.Tensor) -> torch.Tensor:
        """Softmax of the position logits over the filled slots; float32 [B, L].

        a_j = mask_j exp(w_j) / sum_k mask_k exp(w_k). A row with no filled slot
        gets all zeros (its denominator is clamped, so there is no 0 / 0).
        """
        # Subtracting the max is the usual softmax stabilisation; it cancels out.
        unnormalised = torch.exp(self.position_logits - self.position_logits.max())
        masked = unnormalised.unsqueeze(0) * history_mask.float()
        total = masked.sum(dim=1, keepdim=True).clamp(min=EMPTY_HISTORY_EPSILON)
        return masked / total

    def pool(self, history_vectors: torch.Tensor, history_mask: torch.Tensor) -> torch.Tensor:
        """Pooled = sum_j a_j h_j with the position weights; float32 [B, D] (0 for no history)."""
        weights = self.position_weights(history_mask)  # [B, L]
        return (weights.unsqueeze(-1) * history_vectors).sum(dim=1)

    def forward(
        self, history_vectors: torch.Tensor, history_mask: torch.Tensor, profile: torch.Tensor
    ) -> torch.Tensor:
        """User vectors.

        Args:
            history_vectors: float32 [B, L, D]; padded slots may hold anything
                (their weight is 0).
            history_mask: bool [B, L].
            profile: int64 [B, n_profile_columns].

        Returns:
            float32 [B, output_dim], L2-normalised.
        """
        parts = [self.pool(history_vectors, history_mask)]
        for column, encoder in enumerate(self.profile_encoders):
            parts.append(encoder(profile[:, column]))
        return functional.normalize(self.mlp(torch.cat(parts, dim=-1)), dim=-1)


@dataclass(frozen=True)
class BatchEncoding:
    """Model outputs for one training batch of B samples with K distinct target articles.

    Attributes:
        users: float32 [B, D], one user vector per sample.
        columns: float32 [K, D], one vector per DISTINCT target article (the
            loss columns; duplicate-aware).
        column_article_idx: int64 [K], the article of each column.
        target_column: int64 [B], the column of each sample's target.
    """

    users: torch.Tensor
    columns: torch.Tensor
    column_article_idx: torch.Tensor
    target_column: torch.Tensor


def scatter_history(
    filled_vectors: torch.Tensor, history_mask: torch.Tensor, dim: int
) -> torch.Tensor:
    """Place vectors of the filled slots into a [B, L, dim] tensor of zeros.

    Args:
        filled_vectors: float32 [n_filled, dim], in the row-major order of
            history[history_mask].
        history_mask: bool [B, L].
        dim: Vector width.
    """
    batch_size, history_length = history_mask.shape
    empty = filled_vectors.new_zeros(batch_size, history_length, dim)
    # Out-of-place index_put keeps the autograd graph to filled_vectors.
    return empty.index_put((history_mask,), filled_vectors)


class TwoTowerModel(nn.Module):
    """Item tower + user tower; the item feature table is passed in, not stored."""

    def __init__(self, item_tower: ItemTower, user_tower: UserTower, output_dim: int) -> None:
        """Keep both towers (history and target articles share the item tower)."""
        super().__init__()
        self.item_tower = item_tower
        self.user_tower = user_tower
        self.output_dim = output_dim

    def encode_users(
        self,
        history: torch.Tensor,
        history_mask: torch.Tensor,
        profile: torch.Tensor,
        features: DeviceItemFeatures,
    ) -> torch.Tensor:
        """User vectors [B, D] from int64 history [B, L], bool mask [B, L], profile [B, 3].

        Each distinct history article is encoded once; padded slots never are.
        """
        uniq, inverse = torch.unique(history[history_mask], return_inverse=True)
        item_vectors = self.item_tower(uniq, features)
        history_vectors = scatter_history(item_vectors[inverse], history_mask, self.output_dim)
        return self.user_tower(history_vectors, history_mask, profile)

    def encode_batch(self, batch: Batch, features: DeviceItemFeatures) -> BatchEncoding:
        """Encode a training batch, running the item tower once per distinct article.

        The distinct articles are the targets plus the filled history slots
        (padding excluded). Their vectors are gathered back for the history
        (-> user tower) and for the targets (-> loss columns, one per distinct
        target article).
        """
        batch_size = batch.article_idx.shape[0]
        filled_history = batch.history[batch.history_mask]
        uniq, inverse = torch.unique(
            torch.cat([batch.article_idx, filled_history]), return_inverse=True
        )
        item_vectors = self.item_tower(uniq, features)
        history_vectors = scatter_history(
            item_vectors[inverse[batch_size:]], batch.history_mask, self.output_dim
        )
        users = self.user_tower(history_vectors, batch.history_mask, batch.profile)
        # Distinct target articles: rows of `uniq` used as targets, in uniq order.
        column_rows, target_column = torch.unique(inverse[:batch_size], return_inverse=True)
        return BatchEncoding(
            users=users,
            columns=item_vectors[column_rows],
            column_article_idx=uniq[column_rows],
            target_column=target_column,
        )


def build_two_tower_model(
    n_articles: int,
    image_dim: int,
    text_dim: int,
    article_codes: CategoryCodes,
    profile_codes: CategoryCodes,
    history_length: int,
    config: TwoTowerModelConfig,
    seed: int,
) -> TwoTowerModel:
    """Build the model with weights seeded by `seed` (on the CPU; move it with .to())."""
    torch.manual_seed(seed)
    item_tower = ItemTower(n_articles, image_dim, text_dim, article_codes, config)
    user_tower = UserTower(history_length, config.output_dim, profile_codes, config)
    return TwoTowerModel(item_tower, user_tower, config.output_dim)
