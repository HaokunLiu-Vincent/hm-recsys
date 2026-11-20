"""CLI for M9: ONE DeepFM training with its evaluations (proposal §16).

    uv run python scripts/m9_deepfm_trial.py --area speed_test
    uv run python scripts/m9_deepfm_trial.py --area v_phase --target V2 --seed 43

Trains the configuration of configs/deepfm.yaml for one V fold on the same
training rows as LightGBM, saves the model after the last epoch, scores all
candidates of the fold with it and evaluates it. Appends rows to
results/deepfm_metrics.csv and results/deepfm_trials.csv. A T fold is refused.

--area speed_test runs the speed test's one training (target fold and epochs
from the config) and then stops with an error if its MAP@12 is below the
recorded RRF baseline of that fold. One training per process; clean tree only.

The study (scripts/m9_deepfm_study.py) starts this script with the settings of
one configuration: --learning-rate, --embedding-dim, --dropout, --components,
--numeric-input, --loss and --epochs replace the values of the config. The
input-representation extension (scripts/m9_deepfm_extension.py) also passes
--n-bins, the number of quantile bins of the binned input.
"""

# Thread setup must run before polars is imported (see hmrec.threads).
from hmrec.threads import configure_polars_threads

configure_polars_threads()

import argparse  # noqa: E402
import time  # noqa: E402
from dataclasses import replace  # noqa: E402

from hmrec.config import (  # noqa: E402
    DEEPFM_COMPONENTS,
    DEEPFM_LOSSES,
    DEEPFM_NUMERIC_INPUTS,
    DeepFMConfig,
)
from hmrec.deepfm.process import (  # noqa: E402
    below_baseline,
    finish,
    record_trial,
    speed_test_spec,
    start_process,
    with_epochs,
    with_n_bins,
    with_seed,
)
from hmrec.deepfm.records import read_reference_scores  # noqa: E402
from hmrec.deepfm.study import Variant, apply_variant, variant_of  # noqa: E402
from hmrec.deepfm.trial import (  # noqa: E402
    AREA_SPEED_TEST,
    AREAS,
    TrialSpec,
    check_scores_are_finite,
    run_trial,
)
from hmrec.logging_utils import get_logger  # noqa: E402

LOGGER = get_logger("m9_deepfm_trial")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--area", required=True, choices=AREAS)
    parser.add_argument("--target", help="the V fold to predict (not for --area speed_test)")
    parser.add_argument("--seed", type=int, help="training seed (default: the config's)")
    parser.add_argument("--epochs", type=int, help="epochs to train (default: the config's)")
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--embedding-dim", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--components", choices=DEEPFM_COMPONENTS)
    parser.add_argument("--numeric-input", choices=DEEPFM_NUMERIC_INPUTS)
    parser.add_argument("--loss", choices=DEEPFM_LOSSES)
    parser.add_argument("--n-bins", type=int, help="quantile bins (default: the config's)")
    return parser.parse_args()


def configuration(config: DeepFMConfig, args: argparse.Namespace) -> DeepFMConfig:
    """The config with the study settings given on the command line (the others unchanged)."""
    default = variant_of(config)
    given = {
        "learning_rate": args.learning_rate,
        "embedding_dim": args.embedding_dim,
        "dropout": args.dropout,
        "components": args.components,
        "numeric_input": args.numeric_input,
        "loss": args.loss,
        "epochs": args.epochs,
    }
    values = {
        name: getattr(default, name) if value is None else value for name, value in given.items()
    }
    changed = apply_variant(config, Variant(**values))
    if args.n_bins is None:
        return changed
    return with_n_bins(changed, args.n_bins)


def main() -> None:
    """Run one training, record it, and apply the speed test's stop rule."""
    started = time.perf_counter()
    args = parse_args()
    label = f"trial_{args.area}_{args.target or 'default'}_seed{args.seed or 'default'}"
    inputs, stamp = start_process(f"{label}_{time.strftime('%Y%m%dT%H%M%S')}")
    if args.area == AREA_SPEED_TEST:
        if args.target is not None or args.seed is not None or args.epochs is not None:
            raise SystemExit("--area speed_test takes its fold, seed and epochs from the config")
        spec = speed_test_spec(inputs)
        inputs = replace(inputs, config=with_epochs(inputs.config, spec.epochs))
    else:
        if args.target is None:
            raise SystemExit(f"--target is required for --area {args.area}")
        config = configuration(inputs.config, args)
        seed = config.training.seed if args.seed is None else args.seed
        inputs = replace(inputs, config=with_seed(config, seed))
        spec = TrialSpec(args.area, args.target, seed, config.training.epochs)
    result = run_trial(inputs, spec)
    peak = finish(inputs, label)
    record_trial(inputs, spec, result, started, peak, stamp)
    check_scores_are_finite(result)
    metric = inputs.config.metric
    reference = read_reference_scores(inputs.paths, spec.target_fold, metric)
    last = result.metrics[-1]
    LOGGER.info(
        "%s seed %d, %d epochs (input cache %s): %s = %.6f (recorded RRF baseline %.6f, "
        "LightGBM %.6f)",
        spec.target_fold,
        spec.seed,
        last.epoch,
        result.input_cache,
        metric,
        last.scores[metric],
        reference.rrf_baseline,
        reference.lightgbm,
    )
    if args.area == AREA_SPEED_TEST and below_baseline(result, reference):
        raise SystemExit(
            f"speed test: {metric} {last.scores[metric]:.6f} is below the RRF baseline "
            f"{reference.rrf_baseline:.6f}. Stop and report; the model is not changed."
        )


if __name__ == "__main__":
    main()
