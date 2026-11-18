"""Tests for hmrec.ranker.identity: model_config and the 12-character configuration id."""

from dataclasses import replace
from pathlib import Path

from hmrec.config import RankerTunedParams, load_ranker_config
from hmrec.ranker.identity import CONFIG_ID_LENGTH, model_config, ranker_config_id

REPO = Path(__file__).resolve().parents[2]
RANKER = load_ranker_config(REPO / "configs" / "ranker.yaml")
DEFAULT = RankerTunedParams(num_leaves=31, min_data_in_leaf=500, negatives_per_positive=40)


def test_model_config_lists_everything_that_defines_a_model() -> None:
    config = model_config(RANKER, DEFAULT, "full", "row_wise")
    assert set(config) == {
        "experiment",
        "tuned",
        "window_weeks",
        "seed",
        "sampling_seed",
        "histogram_mode",
        "dataset_params",
        "train_params",
    }
    assert config["tuned"] == {
        "num_leaves": 31,
        "min_data_in_leaf": 500,
        "negatives_per_positive": 40,
    }
    # The number of trees is not part of the identity: the first k trees of a
    # model do not depend on how many trees follow.
    assert "num_trees" not in config


def test_the_default_configuration_has_the_recorded_id() -> None:
    # The id every row of the default configuration carries in results/v_phase*.csv.
    # If this changes, model_config or the fixed parameters of configs/ranker.yaml
    # changed, and the recorded trials can no longer be found.
    config = model_config(RANKER, DEFAULT, "full", "row_wise")
    assert ranker_config_id(config) == "ca1a3cdfce49"
    assert len(ranker_config_id(config)) == CONFIG_ID_LENGTH
    # The id does not depend on the order of the keys.
    assert ranker_config_id(dict(reversed(list(config.items())))) == "ca1a3cdfce49"


def test_any_changed_setting_gives_another_id() -> None:
    base_id = ranker_config_id(model_config(RANKER, DEFAULT, "full", "row_wise"))
    slower = replace(RANKER, train_params={**RANKER.train_params, "learning_rate": 0.1})
    others = [
        model_config(RANKER, replace(DEFAULT, num_leaves=63), "full", "row_wise"),
        model_config(RANKER, replace(DEFAULT, min_data_in_leaf=100), "full", "row_wise"),
        model_config(RANKER, replace(DEFAULT, negatives_per_positive=60), "full", "row_wise"),
        model_config(RANKER, DEFAULT, "minus_r6", "row_wise"),
        model_config(RANKER, DEFAULT, "full", "col_wise"),
        model_config(replace(RANKER, seed=43), DEFAULT, "full", "row_wise"),
        model_config(replace(RANKER, window_weeks=5), DEFAULT, "full", "row_wise"),
        model_config(slower, DEFAULT, "full", "row_wise"),
    ]
    ids = {ranker_config_id(other) for other in others}
    assert len(ids) == len(others) and base_id not in ids
    # 0.0 and 0 are different settings (the id is a hash of the JSON text).
    as_int = replace(RANKER, train_params={**RANKER.train_params, "lambda_l1": 0})
    assert ranker_config_id(model_config(as_int, DEFAULT, "full", "row_wise")) != base_id
