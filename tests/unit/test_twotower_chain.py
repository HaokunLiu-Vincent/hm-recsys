"""Tests for recall.twotower.chain: fine-tune data windows, 1:1 replay, predict-before-train, Adam.

Tiny data on FIXTURE_DAY_ZERO; lines are (customer, article, day). Alarm user: customer 3.
    before the replay window: (2, 1, 5)
    replay window of D1 [12, 40): (0,1,20) (1,2,25) (2,3,30) (3,4,35) (0,5,38) (1,0,39)
    D1 week [40, 47):             (0,2,41) (1,3,42) (2,4,43) (3,1,44) (0,2,41) <- repeat line
    D2 week [47, 54):             (0,4,48) (1,5,49) (2,0,50) (3,2,51)
D1 fine-tune: new rows (non-alarm) = (0,2,41) (1,3,42) (2,4,43) -> n_new = 3 (the repeat line
is one row); replay 0.5 -> 3 rows drawn from the 5 non-alarm rows of [12, 40); the alarm
row (3,1,44) is kept apart.
"""

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
import torch
from conftest import FIXTURE_DAY_ZERO, make_queries, make_recall_inputs
from test_twotower_infer import CHUNKS, _resources
from test_twotower_infer import CONFIG as INFER_CONFIG

from hmrec.config import TwoTowerChainConfig, TwoTowerConfig, load_base_config
from hmrec.data.day_index import day_index_to_date
from hmrec.recall.base import RecallInputs
from hmrec.recall.twotower import train as train_module
from hmrec.recall.twotower.chain import (
    TwoTowerChainChannel,
    build_fine_tune_arrays,
    fine_tune_rows,
    replay_count,
)
from hmrec.recall.twotower.infer import TwoTowerChannel
from hmrec.recall.twotower.train import load_checkpoint

REPO = Path(__file__).resolve().parents[2]
MEMORY = load_base_config(REPO / "configs" / "base.yaml").memory
BEFORE = [(2, 1, 5)]
REPLAY_WINDOW = [(0, 1, 20), (1, 2, 25), (2, 3, 30), (3, 4, 35), (0, 5, 38), (1, 0, 39)]
D1_WEEK = [(0, 2, 41), (1, 3, 42), (2, 4, 43), (3, 1, 44), (0, 2, 41)]
D2_WEEK = [(0, 4, 48), (1, 5, 49), (2, 0, 50), (3, 2, 51)]
LINES = BEFORE + REPLAY_WINDOW + D1_WEEK + D2_WEEK
ALARM = np.array([3], dtype=np.int32)
POOL = [0, 1, 2, 3, 4, 5]
CONFIG = dataclasses.replace(
    INFER_CONFIG,
    e_update=1,
    replay=0.5,
    chain_seed=42,
    chain=TwoTowerChainConfig(week_days=7, replay_window_days=28),
)


def _rows(
    config: TwoTowerConfig = CONFIG, lines: list[tuple[int, int, int]] = LINES
) -> tuple[pl.DataFrame, pl.DataFrame, int]:
    transactions = make_recall_inputs(lines, 40).transactions
    return fine_tune_rows(
        transactions, day_index_to_date(40, FIXTURE_DAY_ZERO), FIXTURE_DAY_ZERO, config, ALARM
    )


def _as_tuples(rows: pl.DataFrame) -> list[tuple[int, int, int]]:
    return sorted(rows.select("customer_idx", "article_idx", "t_dat_day").iter_rows())


# ---------------------------------------------------------------- fine-tune data


def test_replay_count() -> None:
    assert replay_count(100, 0.5) == 100  # 1 : 1
    assert replay_count(100, 0.0) == 0
    assert replay_count(3, 0.25) == 1  # 3 x 0.25 / 0.75 = 1
    with pytest.raises(ValueError, match="replay must be in"):
        replay_count(10, 1.0)


def test_new_rows_are_the_week_and_alarm_users_are_kept_apart() -> None:
    train_rows, alarm_rows, n_new = _rows(dataclasses.replace(CONFIG, replay=0.0))
    assert n_new == 3
    assert _as_tuples(train_rows) == [(0, 2, 41), (1, 3, 42), (2, 4, 43)]
    assert _as_tuples(alarm_rows) == [(3, 1, 44)]


def test_replay_is_one_to_one_from_the_four_weeks_before() -> None:
    train_rows, _, n_new = _rows()
    replay = train_rows.filter(pl.col("t_dat_day") < 40)
    assert replay.height == n_new == 3
    # Only [12, 40), never the alarm user, never day 5 (before the window).
    assert (replay["t_dat_day"].to_numpy() >= 12).all()
    assert 3 not in replay["customer_idx"].to_list()
    assert set(_as_tuples(replay)) <= {(0, 1, 20), (1, 2, 25), (2, 3, 30), (0, 5, 38), (1, 0, 39)}


def test_replay_draw_is_reproducible_and_seed_dependent() -> None:
    first, _, _ = _rows()
    again, _, _ = _rows()
    assert first.equals(again)
    draws = {
        tuple(_as_tuples(_rows(dataclasses.replace(CONFIG, chain_seed=s))[0])) for s in range(8)
    }
    assert len(draws) > 1  # another seed can draw other rows


def test_purchases_after_the_week_change_nothing() -> None:
    # D2-week lines (days 48-51) are after the D1 week's end (47).
    without_later, _, _ = _rows(lines=BEFORE + REPLAY_WINDOW + D1_WEEK)
    assert without_later.equals(_rows()[0])
    transactions = make_recall_inputs(LINES, 40).transactions
    arrays = build_fine_tune_arrays(
        transactions, day_index_to_date(40, FIXTURE_DAY_ZERO), FIXTURE_DAY_ZERO, CONFIG, 6, ALARM
    )
    assert arrays.train.t_dat_day.max() < 47
    # Every filled history slot is a purchase before its own sample day: customer 0 on
    # day 41 has history [5, 1] (days 38, 20), not article 4 bought on day 48.
    row = int(np.flatnonzero((arrays.train.customer_idx == 0) & (arrays.train.t_dat_day == 41))[0])
    assert arrays.train.history[row][arrays.train.history_mask[row]].tolist() == [5, 1]
    assert (arrays.n_new, arrays.n_replay, arrays.alarm.n_samples) == (3, 3, 1)


# ---------------------------------------------------------------- the chain channel


def _chain(tmp_path: Path) -> TwoTowerChainChannel:
    return TwoTowerChainChannel(
        CONFIG,
        _resources(),
        CHUNKS,
        lambda fold: tmp_path / f"{fold}.pt",
        ALARM,
        MEMORY,
        {"model_identity": {}},
    )


def _fold(name: str, anchor_day: int) -> tuple[pl.DataFrame, RecallInputs]:
    queries = make_queries([(1, 0), (2, 1), (3, 2)], fold=name)
    return queries, make_recall_inputs(LINES, anchor_day, active_pool=POOL)


def test_each_week_is_predicted_before_the_model_trains_on_it(tmp_path: Path) -> None:
    chain = _chain(tmp_path)
    frozen = TwoTowerChannel(CONFIG, _resources(), CHUNKS)  # same start weights, never updated
    d1_queries, d1_inputs = _fold("D1", 40)
    assert chain.generate(d1_queries, d1_inputs).equals(frozen.generate(d1_queries, d1_inputs))
    # D2 is predicted by the model fine-tuned on D1 (saved as D1.pt), before D2 training.
    after_d1 = _resources()
    load_checkpoint(after_d1.model, tmp_path / "D1.pt")
    after_d1_channel = TwoTowerChannel(CONFIG, after_d1, CHUNKS)
    d2_queries, d2_inputs = _fold("D2", 47)
    expected = after_d1_channel.generate(d2_queries, d2_inputs)
    assert chain.generate(d2_queries, d2_inputs).equals(expected)
    assert (tmp_path / "D2.pt").exists()
    assert chain.fine_tune_results["D1"].n_train_samples == 6  # 3 new + 3 replay


def test_folds_must_come_in_time_order(tmp_path: Path) -> None:
    chain = _chain(tmp_path)
    chain.generate(*_fold("D2", 47))
    with pytest.raises(ValueError, match="time order"):
        chain.generate(*_fold("D1", 40))


def test_every_week_gets_a_new_adam(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[torch.optim.Optimizer] = []
    original = train_module.make_optimizer

    def recording(*args: Any, **kwargs: Any) -> torch.optim.Adam:
        optimizer = original(*args, **kwargs)
        assert not optimizer.state  # fresh: no moments carried over
        created.append(optimizer)
        return optimizer

    monkeypatch.setattr(train_module, "make_optimizer", recording)
    chain = _chain(tmp_path)
    chain.generate(*_fold("D1", 40))
    chain.generate(*_fold("D2", 47))
    assert len(created) == 2 and created[0] is not created[1]
    assert all(
        group["lr"] == CONFIG.training.fine_tune_learning_rate for group in created[1].param_groups
    )


def test_checkpoints_are_never_overwritten(tmp_path: Path) -> None:
    _chain(tmp_path).generate(*_fold("D1", 40))
    with pytest.raises(FileExistsError):
        _chain(tmp_path).generate(*_fold("D1", 40))
