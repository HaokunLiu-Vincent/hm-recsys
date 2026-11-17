"""CLI for M3: train the Two-Tower base model (proposal §6.2).

Usage:
    caffeinate -i uv run python scripts/m3_twotower_base.py [--run-id m3_twotower_base]

Reads the base-window arrays (artifacts/twotower/data/base/, refused if they were
built from uncommitted code), trains `training.epochs` epochs on
`training.device` (configs/recall/r6_twotower.yaml), and after every epoch saves
artifacts/models/twotower/base/epoch_{k}.pt and logs time, training loss,
alarm-user loss and memory peak. Epoch numbers also go to
artifacts/runs/{run_id}/epochs.jsonl. Existing checkpoints are never overwritten.
E_base is chosen afterwards by scripts/m2_recall_hpo.py --channels r6_twotower.
"""

# Thread setup. This MUST run before polars (or any hmrec module that imports
# polars) is imported: polars reads POLARS_MAX_THREADS only once, at import time.
from hmrec.threads import check_polars_thread_pool, configure_polars_threads

configure_polars_threads()

import argparse  # noqa: E402  (after the thread setup above, see comment)
import json  # noqa: E402
import logging  # noqa: E402
from dataclasses import asdict  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import torch  # noqa: E402

from hmrec.config import (  # noqa: E402
    TwoTowerConfig,
    load_base_config,
    load_preprocess_config,
    load_twotower_config,
)
from hmrec.data.metadata import scan_processed_parquet  # noqa: E402
from hmrec.hpo.runner import git_state, warn_uncommitted_changes  # noqa: E402
from hmrec.logging_utils import get_logger, setup_logging  # noqa: E402
from hmrec.paths import ProjectPaths, get_paths  # noqa: E402
from hmrec.recall.twotower.data import (  # noqa: E402
    build_customer_profiles,
    items_to_device,
    load_item_feature_arrays,
    profiles_to_device,
    read_training_arrays,
    samples_to_device,
)
from hmrec.recall.twotower.loss import article_log_q  # noqa: E402
from hmrec.recall.twotower.model import build_two_tower_model  # noqa: E402
from hmrec.recall.twotower.train import (  # noqa: E402
    EpochResult,
    TrainingContext,
    epoch_metadata,
    model_identity,
    run_base_training,
    save_checkpoint,
    split_alarm_users,
)

LOGGER = get_logger("m3_twotower_base")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id", default="m3_twotower_base", help="log goes to artifacts/runs/{run_id}/"
    )
    return parser.parse_args()


def check_inputs(paths: ProjectPaths, config: TwoTowerConfig, n_articles: int) -> dict[str, Any]:
    """Refuse dirty or mismatching arrays and existing checkpoints; return the arrays' metadata."""
    data_dir = paths.twotower_data_dir("base")
    data_metadata = json.loads((data_dir / "metadata.json").read_text())
    if data_metadata["git_dirty"]:
        raise ValueError(f"{data_dir} was built from uncommitted code; rebuild it first")
    if data_metadata["history_length"] != config.history_length:
        raise ValueError(f"{data_dir}: history_length differs from the config")
    if data_metadata["padding_idx"] != n_articles:
        raise ValueError(f"{data_dir}: padding_idx {data_metadata['padding_idx']} != {n_articles}")
    for epoch in range(1, config.training.epochs + 1):
        if paths.twotower_checkpoint_file(epoch).exists():
            raise FileExistsError(f"{paths.twotower_checkpoint_file(epoch)} already exists")
    return data_metadata


def main() -> None:
    """Train and checkpoint the base model."""
    args = parse_args()
    paths = get_paths()
    run_dir = paths.run_dir(args.run_id)
    setup_logging(logging.INFO, log_file=run_dir / "run.log")
    base = load_base_config(paths.base_config_file)
    LOGGER.info("polars thread pool size = %d", check_polars_thread_pool(base.num_threads))
    torch.set_num_threads(base.num_threads)
    commit, dirty = git_state(paths.project_root, paths.results_root)
    if dirty:
        warn_uncommitted_changes(commit)
    config = load_twotower_config(paths.recall_config_file("r6_twotower"))
    preprocess = load_preprocess_config(paths.preprocess_config_file)
    metadata_file = paths.processed_metadata_file
    customers = scan_processed_parquet(paths.customers_file, metadata_file, preprocess).collect()
    articles = scan_processed_parquet(paths.articles_file, metadata_file, preprocess).collect()
    n_articles = articles.height
    data_metadata = check_inputs(paths, config, n_articles)
    profiles = build_customer_profiles(customers)
    items = load_item_feature_arrays(paths, preprocess, articles)
    arrays = read_training_arrays(paths.twotower_data_dir("base"))
    split = split_alarm_users(
        arrays.customer_idx, config.training.alarm_user_share, config.training.seed
    )
    # Q(k) from the TRAINING samples only (alarm users excluded): §5 R6 LogQ.
    log_q = article_log_q(arrays.article_idx[split.train_indices], n_articles + 1)
    LOGGER.info(
        "%d training samples, %d alarm samples (%d alarm users)",
        split.train_indices.size,
        split.alarm_indices.size,
        split.n_alarm_users,
    )
    device = torch.device(config.training.device)
    context = TrainingContext(
        samples=samples_to_device(arrays, device),
        profiles=profiles_to_device(profiles, device),
        features=items_to_device(items, device),
        log_q=torch.from_numpy(log_q).to(device),
        temperature=config.training.temperature,
        device=device,
    )
    model = build_two_tower_model(
        n_articles,
        preprocess.embeddings.image_dim,
        preprocess.embeddings.text_dim,
        items.codes,
        profiles,
        config.history_length,
        config.model,
        config.training.seed,
    ).to(device)
    identity = model_identity(n_articles, items.codes, profiles)
    del arrays, customers, articles

    def save_epoch(epoch: int, result: EpochResult) -> None:
        path = paths.twotower_checkpoint_file(epoch)
        metadata = {
            "epoch": epoch_metadata(result),
            "config": asdict(config),
            "model_identity": identity,
            "data_metadata": data_metadata,
            "n_train_samples": int(split.train_indices.size),
            "device": str(device),
            "torch_version": torch.__version__,
            "git_commit": commit,
            "git_dirty": dirty,
        }
        save_checkpoint(model, path, metadata)
        with (run_dir / "epochs.jsonl").open("a") as handle:
            handle.write(json.dumps(epoch_metadata(result)) + "\n")
        LOGGER.info(
            "epoch %d: %.1f s, train loss %.4f, alarm loss %.4f, peak rss %.2f GB -> %s",
            epoch,
            result.seconds,
            result.mean_train_loss,
            result.alarm_loss,
            result.peak_rss_gb,
            path,
        )

    run_base_training(model, split, context, config.training, base.memory, save_epoch)
    LOGGER.info("base training done: %s", Path(paths.model_dir("twotower", "base")))


if __name__ == "__main__":
    main()
