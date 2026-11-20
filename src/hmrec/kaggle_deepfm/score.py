"""Score the shards of a Kaggle run with the run's DeepFM model. Torch side.

The model and its input transform are read from the files the training wrote
(hmrec.kaggle_deepfm.train). A shard's wide table is the one the LightGBM
submission wrote (artifacts/kaggle/{run}/shards/{shard}/features/); it is
verified with hmrec.feature_files.verify_feature_files before a part is read, and
read one part at a time. Every candidate is scored with the scoring functions of
stage 2 (hmrec.deepfm.score) and the scoring settings of configs/deepfm.yaml;
the Top-K of every query is kept (score DESC, article_idx ASC).

Output of a shard: artifacts/kaggle/deepfm/{run}/shards/{shard}/top.parquet with
query_id (int32), rank (int32, 1 = best), article_idx (int32), score (float64),
sorted by (query_id, rank). An existing file is skipped, never replaced.

No label is read. This module must not import LightGBM or a module that loads it.
"""

import polars as pl
import torch

from hmrec.deepfm.data import feature_columns, read_candidate_part
from hmrec.deepfm.inputs import InputTransform, encode_rows, load_transform
from hmrec.deepfm.model import DeepFM
from hmrec.deepfm.score import ARTICLE_IDX, QUERY_ID, RANK, SCORE, score_rows, top_k_per_query
from hmrec.deepfm.trial import TRANSFORM_FILE, TrialInputs, load_final_model
from hmrec.feature_files import verify_feature_files
from hmrec.kaggle_deepfm.context import DeepFMKaggleContext
from hmrec.kaggle_deepfm.train import read_training_summary
from hmrec.logging_utils import get_logger
from hmrec.ranker.ablation import wide_table_columns

LOGGER = get_logger("kaggle_deepfm.score")

TOP_COLUMNS = [QUERY_ID, RANK, ARTICLE_IDX, SCORE]


def load_run_model(
    context: DeepFMKaggleContext, inputs: TrialInputs
) -> tuple[InputTransform, DeepFM]:
    """The run's input transform and its saved model (eval mode), read from their files.

    Raises:
        FileNotFoundError: If the training is not finished.
        ValueError: If the saved model is of another configuration or seed.
    """
    model_dir = context.paths.kaggle_deepfm_model_dir(context.run)
    summary = read_training_summary(model_dir)
    wanted = (context.settings.config_id, context.settings.seed)
    if (summary.config_id, summary.seed) != wanted:
        raise ValueError(
            f"{model_dir} holds a model of (configuration, seed) = "
            f"{(summary.config_id, summary.seed)}, expected {wanted}"
        )
    transform = load_transform(model_dir / TRANSFORM_FILE)
    return transform, load_final_model(model_dir, transform, inputs, summary.epochs)


def score_shard(
    context: DeepFMKaggleContext,
    inputs: TrialInputs,
    transform: InputTransform,
    model: DeepFM,
    shard: int,
) -> int:
    """Score ALL candidates of one shard and write the Top-K of every query.

    Args:
        context: The process's context.
        inputs: The run's configuration (hmrec.kaggle_deepfm.train.chosen_inputs).
        transform: The run's input transform.
        model: The run's model.
        shard: 0-based shard number.

    Returns:
        The number of candidate rows scored; 0 if the shard's file existed already.
    """
    paths, config = context.paths, inputs.config
    top_file = paths.kaggle_deepfm_top_file(context.run, shard)
    if top_file.exists():
        LOGGER.warning("%s shard %d: %s exists; SKIPPED", context.run, shard, top_file)
        return 0
    features_dir = paths.kaggle_features_dir(context.run, shard)
    feature_files = verify_feature_files(features_dir, wide_table_columns(inputs.lock))
    columns = feature_columns(inputs.lock)
    device = torch.device(config.training.device)
    tops: list[pl.DataFrame] = []
    n_rows = 0
    for part in feature_files.parts:
        candidates = read_candidate_part(features_dir / part.name, columns)
        encoded = encode_rows(candidates.raw, transform)
        scores = score_rows(
            model, encoded, config.scoring_rows_per_chunk, config.scoring_rows_per_block, device
        )
        tops.append(
            top_k_per_query(candidates.query_ids, candidates.article_idx, scores, config.top_k)
        )
        n_rows += encoded.n_rows
    top = pl.concat(tops).select(TOP_COLUMNS).sort(QUERY_ID, RANK)
    top_file.parent.mkdir(parents=True, exist_ok=True)
    top.write_parquet(top_file)
    LOGGER.info(
        "%s shard %d: scored %d rows, Top-%d of %d queries -> %s",
        context.run,
        shard,
        n_rows,
        config.top_k,
        top[QUERY_ID].n_unique(),
        top_file,
    )
    return n_rows


def score_run(context: DeepFMKaggleContext, inputs: TrialInputs, n_shards: int) -> int:
    """Score every shard of the run that has no Top-K file yet.

    Returns:
        The number of candidate rows scored now.
    """
    # As in a stage-2 trial: the thread count is set before anything is computed.
    torch.set_num_threads(inputs.base.num_threads)
    transform, model = load_run_model(context, inputs)
    return sum(score_shard(context, inputs, transform, model, shard) for shard in range(n_shards))
