"""The identity of a Ranker model configuration: model_config and its 12-character id.

A model is defined by everything in model_config: the experiment, the three
tuned values, the training window, the seeds, the histogram mode and every
LightGBM parameter. The id is a hash of that mapping, so two trials with the
same id were trained with the same settings (on different folds).

This lives in ranker/ (not hpo/) because two places need it and must agree:
the V-phase trials (hmrec.hpo.v_trial) and the locked configuration
(hmrec.ranker.locked), which recomputes the id from configs/locks/ranker.lock.yaml
and refuses a lock whose settings do not give the recorded id.
"""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

from hmrec.config import RankerConfig, RankerTunedParams

CONFIG_ID_LENGTH = 12


def model_config(
    ranker: RankerConfig, tuned: RankerTunedParams, experiment: str, histogram_mode: str
) -> dict[str, Any]:
    """Everything that defines a trained model, apart from its number of trees and its fold.

    The number of trees is left out on purpose: training is deterministic, so
    the first k trees do not depend on how many trees follow.
    """
    return {
        "experiment": experiment,
        "tuned": asdict(tuned),
        "window_weeks": ranker.window_weeks,
        "seed": ranker.seed,
        "sampling_seed": ranker.sampling.seed,
        "histogram_mode": histogram_mode,
        "dataset_params": dict(ranker.dataset_params),
        "train_params": dict(ranker.train_params),
    }


def ranker_config_id(config: Mapping[str, Any]) -> str:
    """12 hex characters identifying a model_config."""
    text = json.dumps(config, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:CONFIG_ID_LENGTH]
