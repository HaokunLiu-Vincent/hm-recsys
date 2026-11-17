"""CLI for the Two-Tower speed benchmark: 200 steps on one device (proposal §1.4).

Usage:
    uv run python scripts/m0_benchmark_twotower.py --device mps --model real [--run-id ...]
    uv run python scripts/m0_benchmark_twotower.py --device cpu --model real

Loads the base-window arrays (artifacts/twotower/data/base/, built by
scripts/m3_build_twotower_data.py), the profile codes and the item feature
table, moves them to the device, and trains for `benchmark.steps` steps:
    --model placeholder: the placeholder of hmrec.recall.twotower.benchmark (M0);
    --model real:        the R6 model and loss (model.py, loss.py, train.train_step),
                         used to choose the training device (docs/decisions.md).
Writes artifacts/runs/{run_id}/benchmark_{model}_{device}.json. Run one device per
process, so the memory numbers of the two devices do not mix.
"""

# Thread setup. This MUST run before polars (or any hmrec module that imports
# polars) is imported: polars reads POLARS_MAX_THREADS only once, at import time.
from hmrec.threads import check_polars_thread_pool, configure_polars_threads

configure_polars_threads()

import argparse  # noqa: E402  (after the thread setup above, see comment)
import json  # noqa: E402
import logging  # noqa: E402
from dataclasses import asdict  # noqa: E402

import torch  # noqa: E402

from hmrec.config import (  # noqa: E402
    TwoTowerConfig,
    load_base_config,
    load_preprocess_config,
    load_twotower_config,
)
from hmrec.data.metadata import scan_processed_parquet  # noqa: E402
from hmrec.logging_utils import get_logger, setup_logging  # noqa: E402
from hmrec.memory import MemoryTracker, bytes_to_gb, get_current_rss_bytes  # noqa: E402
from hmrec.paths import get_paths  # noqa: E402
from hmrec.recall.twotower.benchmark import (  # noqa: E402
    StepFunction,
    make_placeholder_model,
    placeholder_step_function,
    run_benchmark,
    summarize_benchmark,
)
from hmrec.recall.twotower.data import (  # noqa: E402
    CategoryCodes,
    DeviceItemFeatures,
    DeviceSamples,
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
    TrainingContext,
    make_optimizer,
    train_step,
)

LOGGER = get_logger("m0_benchmark_twotower")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["mps", "cpu"], required=True)
    parser.add_argument("--model", choices=["placeholder", "real"], default="placeholder")
    parser.add_argument(
        "--run-id",
        default="m0_benchmark_twotower",
        help="log and results go to artifacts/runs/{run_id}/",
    )
    return parser.parse_args()


def mps_memory_gb() -> dict[str, float]:
    """Memory held by PyTorch on MPS: tensors in use, and all memory the driver holds."""
    return {
        "mps_allocated_gb": bytes_to_gb(torch.mps.current_allocated_memory()),
        "mps_driver_gb": bytes_to_gb(torch.mps.driver_allocated_memory()),
    }


def real_step_function(
    config: TwoTowerConfig,
    samples: DeviceSamples,
    profile_tensor: torch.Tensor,
    features: DeviceItemFeatures,
    codes: tuple[CategoryCodes, CategoryCodes, int],
    device: torch.device,
) -> StepFunction:
    """The R6 training step (same code as base training) on `device`."""
    article_codes, profile_codes, n_articles = codes
    model = build_two_tower_model(
        n_articles,
        int(features.image.shape[1]),
        int(features.text.shape[1]),
        article_codes,
        profile_codes,
        config.history_length,
        config.model,
        config.training.seed,
    ).to(device)
    targets = samples.article_idx.cpu().numpy()
    log_q = torch.from_numpy(article_log_q(targets, n_articles + 1)).to(device)
    context = TrainingContext(
        samples, profile_tensor, features, log_q, config.training.temperature, device
    )
    optimizer = make_optimizer(model, config.training, config.training.learning_rate)
    return lambda batch: train_step(model, optimizer, batch, context)


def main() -> None:
    """Load the data onto the device, run the benchmark, write the numbers."""
    args = parse_args()
    paths = get_paths()
    run_dir = paths.run_dir(args.run_id)
    setup_logging(logging.INFO, log_file=run_dir / "run.log")
    base = load_base_config(paths.base_config_file)
    LOGGER.info("polars thread pool size = %d", check_polars_thread_pool(base.num_threads))
    torch.set_num_threads(base.num_threads)
    device = torch.device(args.device)
    output_file = run_dir / f"benchmark_{args.model}_{args.device}.json"
    if output_file.exists():
        raise FileExistsError(f"{output_file} exists; use another --run-id")
    twotower = load_twotower_config(paths.recall_config_file("r6_twotower"))
    config = twotower.benchmark
    preprocess = load_preprocess_config(paths.preprocess_config_file)
    metadata_file = paths.processed_metadata_file
    customers = scan_processed_parquet(paths.customers_file, metadata_file, preprocess).collect()
    articles = scan_processed_parquet(paths.articles_file, metadata_file, preprocess).collect()
    profiles = build_customer_profiles(customers)
    items = load_item_feature_arrays(paths, preprocess, articles)
    arrays = read_training_arrays(paths.twotower_data_dir("base"))
    device_samples = samples_to_device(arrays, device)
    device_items = items_to_device(items, device)
    device_profiles = profiles_to_device(profiles, device)
    article_codes, n_articles = items.codes, articles.height
    # The CPU copies are no longer needed; drop them before timing.
    del arrays, items, customers, articles
    if args.model == "real":
        codes = (article_codes, profiles, n_articles)
        step_function = real_step_function(
            twotower, device_samples, device_profiles, device_items, codes, device
        )
    else:
        model = make_placeholder_model(
            device_items, article_codes.cardinalities, profiles.cardinalities, config
        )
        step_function = placeholder_step_function(model, config.learning_rate, config.temperature)
    with MemoryTracker(
        f"benchmark {args.model} {args.device}",
        LOGGER,
        limit_gb=base.memory.limit_gb,
        sample_interval_seconds=base.memory.sample_interval_seconds,
    ) as tracker:
        times = run_benchmark(step_function, device_samples, device_profiles, config, device)
    summary = summarize_benchmark(times, config.warmup_steps)
    n_samples = int(device_samples.article_idx.shape[0])
    steps_per_epoch = -(-n_samples // config.batch_size)  # ceiling division
    seconds_per_step = summary.steady_fetch_mean + summary.steady_compute_mean
    result = {
        "device": args.device,
        "model": args.model,
        "torch_version": torch.__version__,
        "n_samples": n_samples,
        "steps_per_epoch": steps_per_epoch,
        "estimated_epoch_minutes": steps_per_epoch * seconds_per_step / 60,
        "summary": asdict(summary),
        "fetch_seconds": times.fetch_seconds,
        "compute_seconds": times.compute_seconds,
        "losses": times.losses,
        "peak_rss_gb": bytes_to_gb(tracker.peak_rss_bytes),
        "end_rss_gb": bytes_to_gb(get_current_rss_bytes()),
        **(mps_memory_gb() if device.type == "mps" else {}),
    }
    output_file.write_text(json.dumps(result, indent=2))
    LOGGER.info("summary %s %s: %s", args.model, args.device, summary)
    LOGGER.info(
        "steps/epoch %d, estimated epoch %.1f min -> %s",
        steps_per_epoch,
        result["estimated_epoch_minutes"],
        output_file,
    )


if __name__ == "__main__":
    main()
