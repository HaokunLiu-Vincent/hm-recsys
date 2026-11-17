"""R6 weekly training chain (proposal §6.4): predict the week, THEN fine-tune on it.

Starting from the base checkpoint (config.base_epoch), for each fold in time order:

    1. predict the fold with the current model (scored by the runner). The model
       has only seen purchases before the fold's anchor a, so the score is a
       true "predict first, train after" (progressive validation) score;
    2. fine-tune on the fold's week, then save the model to
       artifacts/models/twotower/chain/{trial_id}/{fold}.pt. After the last fold
       (D4) this is M4, the model F1 starts from.

Fine-tune set of the fold with anchor a (built per fold and released after):

    new rows:    samples with purchase day in [a, a + week_days) (7 days), with the
                 same sample, history and dedup rules as base training (data.py);
    replay rows: n_replay = round(n_new x replay / (1 - replay)) samples drawn
                 without replacement from purchase days [a - replay_window_days, a)
                 (the 4 weeks before the week), seeded by (chain_seed, a);
                 replay = 0.5 -> n_replay = n_new (new : old = 1 : 1); replay = 0 -> none;
    alarm users: the same 1% of users as in base training (drawn again from the
                 base arrays with the base seed) are left out of both; their
                 new-week loss is logged after the update.

Every week gets a NEW Adam (lr = fine_tune_learning_rate), and log Q is counted on
that week's fine-tune set (train.fine_tune).

Dates are filtered only through hmrec.data.asof (load_window_lines /
load_customer_lines_before); histories use purchases before each sample's day.
"""

from collections.abc import Callable
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from hmrec.config import MemoryConfig, TwoTowerConfig
from hmrec.data.asof import TRANSACTION_DAY
from hmrec.data.day_index import date_to_day_index
from hmrec.data.id_maps import ARTICLE_IDX, CUSTOMER_IDX
from hmrec.logging_utils import get_logger
from hmrec.recall.base import OutputSpec, RecallInputs
from hmrec.recall.twotower.data import (
    arrays_from_samples,
    build_samples,
    daily_purchases,
    load_customer_lines_before,
    load_window_lines,
)
from hmrec.recall.twotower.infer import TWOTOWER_SPEC, TwoTowerResources, twotower_candidates
from hmrec.recall.twotower.train import (
    DeviceTables,
    FineTuneArrays,
    FineTuneResult,
    FineTuneSchedule,
    fine_tune,
    save_checkpoint,
)

LOGGER = get_logger("recall.twotower.chain")

SAMPLE_COLUMNS = [CUSTOMER_IDX, TRANSACTION_DAY, ARTICLE_IDX]
# Second number of the replay seed, so replay draws never share a stream with shuffles.
REPLAY_STREAM = 1


# ---------------------------------------------------------------- fine-tune data


def replay_count(n_new: int, replay: float) -> int:
    """Replay rows for n_new new rows: replay is the share of the whole fine-tune set.

    n_replay / (n_new + n_replay) = replay  ->  n_replay = n_new x replay / (1 - replay).
    replay = 0.5 gives n_replay = n_new (1 : 1).
    """
    if not 0 <= replay < 1:
        raise ValueError(f"replay must be in [0, 1), got {replay}")
    return round(n_new * replay / (1 - replay))


def draw_replay_rows(pool: pl.DataFrame, n_rows: int, seed: int, week_day: int) -> pl.DataFrame:
    """n_rows rows of `pool` drawn without replacement, seeded by (seed, week_day).

    Raises:
        ValueError: If the pool is smaller than n_rows.
    """
    if n_rows > pool.height:
        raise ValueError(f"replay needs {n_rows} rows but the replay window has {pool.height}")
    rng = np.random.default_rng([seed, week_day, REPLAY_STREAM])
    chosen = np.sort(rng.choice(pool.height, size=n_rows, replace=False))
    return pool[chosen].select(SAMPLE_COLUMNS)


def _split_by_customers(
    samples: pl.DataFrame, customers: np.ndarray
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(rows of other customers, rows of `customers`)."""
    listed = pl.Series(CUSTOMER_IDX, customers, dtype=pl.Int32).implode()
    is_listed = pl.col(CUSTOMER_IDX).is_in(listed)
    return samples.filter(~is_listed), samples.filter(is_listed)


def fine_tune_rows(
    transactions: pl.LazyFrame,
    anchor: date,
    day_zero: date,
    config: TwoTowerConfig,
    alarm_customers: np.ndarray,
) -> tuple[pl.DataFrame, pl.DataFrame, int]:
    """(training rows, alarm rows, n_new) of the week starting at `anchor`.

    Rows have columns customer_idx, t_dat_day, article_idx (one per daily purchase).
    Training rows = new rows of non-alarm users + replay rows (non-alarm users).
    """
    week_end = anchor + timedelta(days=config.chain.week_days)
    new_lines = load_window_lines(transactions, week_end, day_zero, config.chain.week_days)
    new_rows, alarm_rows = _split_by_customers(build_samples(new_lines), alarm_customers)
    n_replay = replay_count(new_rows.height, config.replay)
    rows = [new_rows.select(SAMPLE_COLUMNS)]
    if n_replay > 0:
        window = config.chain.replay_window_days
        old_lines = load_window_lines(transactions, anchor, day_zero, window)
        old_rows, _ = _split_by_customers(build_samples(old_lines), alarm_customers)
        week_day = date_to_day_index(anchor, day_zero)
        rows.append(draw_replay_rows(old_rows, n_replay, config.chain_seed, week_day))
    return pl.concat(rows), alarm_rows.select(SAMPLE_COLUMNS), new_rows.height


def build_fine_tune_arrays(
    transactions: pl.LazyFrame,
    anchor: date,
    day_zero: date,
    config: TwoTowerConfig,
    n_articles: int,
    alarm_customers: np.ndarray,
) -> FineTuneArrays:
    """The fine-tune set (new + replay) and the alarm rows of the week at `anchor`, with histories.

    Histories read only the involved customers' purchases before the week's end
    (each history then keeps only days before its own sample day).
    """
    train_rows, alarm_rows, n_new = fine_tune_rows(
        transactions, anchor, day_zero, config, alarm_customers
    )
    week_end = anchor + timedelta(days=config.chain.week_days)
    customers = np.unique(
        np.concatenate([train_rows[CUSTOMER_IDX].to_numpy(), alarm_rows[CUSTOMER_IDX].to_numpy()])
    ).astype(np.int32)
    lines = load_customer_lines_before(transactions, customers, week_end, day_zero)
    events = daily_purchases(lines)
    length = config.history_length
    return FineTuneArrays(
        train=arrays_from_samples(build_samples(train_rows), events, length, n_articles),
        alarm=arrays_from_samples(build_samples(alarm_rows), events, length, n_articles),
        n_new=n_new,
        n_replay=train_rows.height - n_new,
    )


# ---------------------------------------------------------------- the chain channel


class TwoTowerChainChannel:
    """R6 with weekly fine-tuning: generate() predicts the fold, then fine-tunes on it.

    The model changes after every call, so the folds must come in time order
    (the runner evaluates D1, D2, D3, D4 in that order); an earlier or repeated
    anchor raises.
    """

    def __init__(
        self,
        config: TwoTowerConfig,
        resources: TwoTowerResources,
        chunk_sizes: tuple[int, int],
        checkpoint_file: Callable[[str], Path],
        alarm_customers: np.ndarray,
        memory: MemoryConfig,
        checkpoint_metadata: dict[str, Any],
    ) -> None:
        """Store the settings; `resources.model` is the starting (base) model.

        Args:
            config: R6 settings with e_update > 0.
            resources: Model (changed in place week by week) and tables.
            chunk_sizes: (query chunk size, pair chunk size) for prediction.
            checkpoint_file: fold name -> path of the model after that fold's fine-tune.
            alarm_customers: Users kept out of every fine-tune.
            memory: Memory limit and sampling interval.
            checkpoint_metadata: Extra metadata saved in every chain checkpoint.
        """
        if config.e_update <= 0:
            raise ValueError("the chain channel needs e_update > 0; use TwoTowerChannel")
        self.config = config
        self.resources = resources
        self.chunk_sizes = chunk_sizes
        self.checkpoint_file = checkpoint_file
        self.alarm_customers = alarm_customers
        self.memory = memory
        self.checkpoint_metadata = checkpoint_metadata
        self.last_anchor: date | None = None
        self.fine_tune_results: dict[str, FineTuneResult] = {}

    @property
    def output_spec(self) -> OutputSpec:
        """R6 output: base columns only, default order."""
        return TWOTOWER_SPEC

    def _check_order(self, inputs: RecallInputs) -> None:
        """Folds must come strictly later each time (the model only moves forward)."""
        if self.last_anchor is not None and inputs.anchor <= self.last_anchor:
            raise ValueError(
                f"chain folds must come in time order: {inputs.anchor} after {self.last_anchor}"
            )
        self.last_anchor = inputs.anchor

    def generate(self, queries: pl.DataFrame, inputs: RecallInputs) -> pl.DataFrame:
        """Predict the fold with the current model, then fine-tune on the fold's week."""
        folds = queries["fold"].unique().to_list()
        if len(folds) != 1:
            raise ValueError(f"queries must belong to one fold, got {folds}")
        self._check_order(inputs)
        # Predict FIRST: the model has not seen any purchase on or after inputs.anchor.
        candidates = twotower_candidates(
            self.config, self.resources, queries, inputs, self.chunk_sizes
        )
        self.fine_tune_on_week(str(folds[0]), inputs)
        return candidates

    def fine_tune_on_week(self, fold: str, inputs: RecallInputs) -> FineTuneResult:
        """Fine-tune on the week [anchor, anchor + week_days) and save the new model."""
        arrays = build_fine_tune_arrays(
            inputs.transactions,
            inputs.anchor,
            inputs.day_zero,
            self.config,
            self.resources.n_articles,
            self.alarm_customers,
        )
        schedule = FineTuneSchedule(self.config.e_update, self.config.chain_seed, inputs.anchor_day)
        tables = DeviceTables(self.resources.profiles, self.resources.features)
        result = fine_tune(
            self.resources.model, arrays, tables, self.config.training, schedule, self.memory
        )
        metadata = {
            **self.checkpoint_metadata,
            "fold": fold,
            "anchor": inputs.anchor,
            "n_new": arrays.n_new,
            "n_replay": arrays.n_replay,
            "fine_tune": asdict(result),
        }
        save_checkpoint(self.resources.model, self.checkpoint_file(fold), metadata)
        self.fine_tune_results[fold] = result
        LOGGER.info(
            "%s fine-tune: %d new + %d replay rows, %d epoch(s), %.1f s, loss %.4f, "
            "alarm loss %.4f, peak rss %.2f GB",
            fold,
            arrays.n_new,
            arrays.n_replay,
            result.epochs,
            result.seconds,
            result.mean_train_loss,
            result.alarm_loss,
            result.peak_rss_gb,
        )
        return result
