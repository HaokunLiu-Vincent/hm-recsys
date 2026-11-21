"""R5 WALS: time-decayed implicit ALS (proposal §5 R5), channel "wals".

Fit window [a - window_days, a) (default 180 days), configs/recall/r5_wals.yaml.

Formulas (§5 R5):
    r_ui = sum of exp(-age_days / decay_tau_days) over u's purchase lines of i   (W2)
           (repeat purchase lines each add their own weight)
         = 1 if u bought i at least once in the window                         (W0)
    c_ui = 1 + alpha_conf * r_ui                                               (confidence)

The CSR matrix passed to `implicit` holds c_ui itself, and the library's own
`alpha` is set to 1.0. implicit multiplies the matrix by its `alpha` and then
treats each stored value as the confidence C_ui (it uses C_ui - 1 internally).
So we store c_ui = 1 + alpha_conf * r_ui and leave the library alpha at 1.0;
setting both would apply the amplification twice.

Matrix axes: only users and articles with at least one purchase line in the
window get a row / column (and so a vector). A user without a purchase in the
window has no vector and gets no candidates; an article without a purchase has
no vector and is never a candidate (every article of A(a) has purchases in the
last 28 days, so it always has one).

Inference: score = user vector . article vector over the active pool A(a),
Top-quota per query, with hmrec.recall.vector_scoring (shared with R6): the
stored score comes from hmrec.vectors.dot_pairs, so it equals wals_dot exactly.
Already-bought articles are NOT excluded.

Uses the CPU version of implicit, num_threads from configs/base.yaml, and a
fixed random_seed for the initial factors.

Threads (measured 2026-11-21, docs/m10_calibration_plan.md §8): on the laptop
the macOS wheel of implicit has no OpenMP, so the fit runs on ONE thread and
num_threads has no effect. On Linux the fit runs on num_threads OpenMP threads;
there the BLAS of scipy is limited to one thread for the duration of the fit
(hmrec.blas_threads, decision of 2026-11-22). Nothing is limited on the laptop.
"""

import sys
from dataclasses import dataclass

import numpy as np
import polars as pl
from implicit.cpu.als import AlternatingLeastSquares
from scipy import sparse

from hmrec.blas_threads import single_threaded_scipy_blas
from hmrec.config import WalsConfig
from hmrec.data.asof import history_window
from hmrec.logging_utils import get_logger
from hmrec.recall.base import (
    ARTICLE_IDX,
    CUSTOMER_IDX,
    OutputSpec,
    RecallInputs,
    add_age_days,
    decay_weight_expr,
)
from hmrec.recall.vector_scoring import candidates_from_vectors
from hmrec.vectors import UserItemVectors

STRENGTH = "interaction_strength"
CONFIDENCE = "confidence"

WALS_SPEC = OutputSpec(channel="wals")

LOGGER = get_logger("recall.r5_wals")


# ---------------------------------------------------------------- confidence matrix


def interaction_strength(
    history: pl.DataFrame, variant: str, decay_tau_days: float
) -> pl.DataFrame:
    """r_ui per (customer, article) (§5 R5).

    Args:
        history: Purchase lines of the fit window with age_days (from add_age_days);
            columns customer_idx, article_idx, age_days.
        variant: "W2" = sum of exp(-age_days / decay_tau_days) over the purchase
            lines (two identical lines count twice); "W0" = 1 per pair.
        decay_tau_days: Decay time constant (W2 only).

    Returns:
        One row per distinct (customer_idx, article_idx); column interaction_strength (float64).
    """
    pairs = history.group_by(CUSTOMER_IDX, ARTICLE_IDX)
    if variant == "W2":
        return pairs.agg(decay_weight_expr(decay_tau_days).sum().alias(STRENGTH))
    if variant == "W0":
        return pairs.agg(pl.lit(1.0, dtype=pl.Float64).alias(STRENGTH))
    raise ValueError(f"unknown WALS variant {variant!r}, expected 'W0' or 'W2'")


@dataclass(frozen=True)
class ConfidenceMatrix:
    """The user x article confidence matrix given to implicit.

    Attributes:
        matrix: scipy CSR matrix, shape [n_users, n_items], float32; entry (r, c)
            = c_ui of user user_customer_idx[r] and article item_article_idx[c].
        user_customer_idx: int32 [n_users], strictly increasing.
        item_article_idx: int32 [n_items], strictly increasing.
    """

    matrix: sparse.csr_matrix
    user_customer_idx: np.ndarray
    item_article_idx: np.ndarray


def build_confidence_matrix(strength: pl.DataFrame, alpha_conf: float) -> ConfidenceMatrix:
    """c_ui = 1 + alpha_conf * r_ui as a CSR matrix over users / articles that have a purchase.

    Args:
        strength: From interaction_strength (one row per pair).
        alpha_conf: Confidence scale.

    Returns:
        The confidence matrix and its row / column index arrays.
    """
    user_customer_idx = np.unique(strength[CUSTOMER_IDX].to_numpy()).astype(np.int32)
    item_article_idx = np.unique(strength[ARTICLE_IDX].to_numpy()).astype(np.int32)
    rows = np.searchsorted(user_customer_idx, strength[CUSTOMER_IDX].to_numpy())
    columns = np.searchsorted(item_article_idx, strength[ARTICLE_IDX].to_numpy())
    # See proposal §5 R5: c_ui = 1 + alpha_conf * r_ui. Computed in float64, stored
    # as float32 because implicit trains in float32.
    confidence = (1.0 + alpha_conf * strength[STRENGTH].to_numpy()).astype(np.float32)
    matrix = sparse.csr_matrix(
        (confidence, (rows, columns)), shape=(user_customer_idx.size, item_article_idx.size)
    )
    matrix.sort_indices()
    return ConfidenceMatrix(matrix, user_customer_idx, item_article_idx)


# ---------------------------------------------------------------- fitting


def build_als_model(config: WalsConfig, num_threads: int) -> AlternatingLeastSquares:
    """The implicit CPU ALS model for `config`.

    alpha=1.0 on purpose: the CSR values already are c_ui = 1 + alpha_conf * r_ui
    (see the module docstring); a library alpha != 1 would scale them again.
    """
    return AlternatingLeastSquares(
        factors=config.factors,
        regularization=config.regularization,
        alpha=1.0,
        dtype=np.float32,
        iterations=config.iterations,
        calculate_training_loss=False,
        num_threads=num_threads,
        random_state=config.random_seed,
    )


def fit_wals(confidence: ConfidenceMatrix, config: WalsConfig, num_threads: int) -> UserItemVectors:
    """Fit ALS and return the user / article vectors.

    Returns:
        UserItemVectors with float32 [n_users, factors] and [n_items, factors].
    """
    model = build_als_model(config, num_threads)
    # Linux only: scipy's BLAS runs on one thread inside the block, because implicit calls
    # it from each of its own OpenMP threads. On macOS the block changes nothing.
    with single_threaded_scipy_blas(sys.platform) as limited_libraries:
        if limited_libraries:
            LOGGER.info("BLAS limited to 1 thread during the fit: %s", limited_libraries)
        model.fit(confidence.matrix, show_progress=False)
    return UserItemVectors(
        user_customer_idx=confidence.user_customer_idx,
        user_vectors=np.ascontiguousarray(model.user_factors, dtype=np.float32),
        item_article_idx=confidence.item_article_idx,
        item_vectors=np.ascontiguousarray(model.item_factors, dtype=np.float32),
    )


# ---------------------------------------------------------------- channel


class WalsChannel:
    """R5 recall channel, see the module docstring."""

    def __init__(
        self,
        config: WalsConfig,
        num_threads: int,
        query_chunk_size: int,
        pair_chunk_size: int,
    ) -> None:
        """Store the R5 config, the thread count and the chunk sizes (configs/base.yaml)."""
        self.config = config
        self.num_threads = num_threads
        self.query_chunk_size = query_chunk_size
        self.pair_chunk_size = pair_chunk_size

    @property
    def output_spec(self) -> OutputSpec:
        """R5 output: base columns only, default order."""
        return WALS_SPEC

    def build_confidence(self, inputs: RecallInputs) -> ConfidenceMatrix:
        """The confidence matrix of the fit window (no fitting)."""
        history = history_window(
            inputs.transactions,
            inputs.anchor,
            inputs.day_zero,
            self.config.window_days,
            columns=[CUSTOMER_IDX, ARTICLE_IDX],
        )
        history = add_age_days(history, inputs.anchor_day)
        strength = interaction_strength(history, self.config.variant, self.config.decay_tau_days)
        return build_confidence_matrix(strength, self.config.alpha_conf)

    def fit(self, inputs: RecallInputs) -> UserItemVectors:
        """Fit ALS on the fit window and return the vectors."""
        confidence = self.build_confidence(inputs)
        LOGGER.info(
            "fitting WALS: %d users x %d articles, %d non-zeros",
            confidence.user_customer_idx.size,
            confidence.item_article_idx.size,
            confidence.matrix.nnz,
        )
        return fit_wals(confidence, self.config, self.num_threads)

    def candidates_from_vectors(
        self, queries: pl.DataFrame, inputs: RecallInputs, vectors: UserItemVectors
    ) -> pl.DataFrame:
        """Score `queries` with already-fitted vectors."""
        return candidates_from_vectors(
            queries,
            vectors,
            inputs.active_pool,
            self.config.quota,
            self.query_chunk_size,
            self.pair_chunk_size,
            WALS_SPEC,
        )

    def generate(self, queries: pl.DataFrame, inputs: RecallInputs) -> pl.DataFrame:
        """Fit on the fit window, then return R5 candidates for `queries`."""
        return self.candidates_from_vectors(queries, inputs, self.fit(inputs))
