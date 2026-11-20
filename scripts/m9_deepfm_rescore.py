"""CLI for M9: check that the current scoring code reproduces a finished trial (proposal §16.10).

    uv run python scripts/m9_deepfm_rescore.py --target V1 --seed 42 --epochs 1 \
        --numeric-input standardised --expect-config-id 08775a5d6efb

Loads the saved model of a finished trial, scores all candidates of its V fold
again with the current code and compares the Top-K with the trial's saved
predictions, bit for bit. Used after the scoring was changed to use less memory:
the trial named above ran before that change. Exits with an error if a single
row differs. Reads no label, trains nothing, writes nothing but its log; its
process peak is the peak of scoring alone. A T fold is refused. Clean tree only.
"""

# Thread setup must run before polars is imported (see hmrec.threads).
from hmrec.threads import configure_polars_threads

configure_polars_threads()

import argparse  # noqa: E402
import time  # noqa: E402
from dataclasses import replace  # noqa: E402

import torch  # noqa: E402

from hmrec.config import DEEPFM_LOSSES, DEEPFM_NUMERIC_INPUTS  # noqa: E402
from hmrec.deepfm.process import finish, start_process, with_n_bins, with_seed  # noqa: E402
from hmrec.deepfm.records import deepfm_config_id  # noqa: E402
from hmrec.deepfm.rescore import rescore_trial  # noqa: E402
from hmrec.deepfm.study import apply_variant, variant_of  # noqa: E402
from hmrec.deepfm.trial import AREA_V_PHASE, TrialSpec  # noqa: E402
from hmrec.logging_utils import get_logger  # noqa: E402
from hmrec.run_records import check_memory_limit  # noqa: E402

LOGGER = get_logger("m9_deepfm_rescore")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, help="the V fold of the trial")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--numeric-input", required=True, choices=DEEPFM_NUMERIC_INPUTS)
    parser.add_argument("--loss", choices=DEEPFM_LOSSES, help="default: the config's")
    parser.add_argument("--n-bins", type=int, help="quantile bins (default: the config's)")
    parser.add_argument(
        "--expect-config-id", required=True, help="the id the trial is recorded with"
    )
    return parser.parse_args()


def main() -> None:
    """Score one finished trial again and fail unless its saved Top-K is reproduced exactly."""
    args = parse_args()
    label = f"rescore_{args.target}_seed{args.seed}_{time.strftime('%Y%m%dT%H%M%S')}"
    inputs, _ = start_process(label)
    default = variant_of(inputs.config)
    variant = replace(
        default,
        numeric_input=args.numeric_input,
        loss=default.loss if args.loss is None else args.loss,
        epochs=args.epochs,
    )
    config = apply_variant(inputs.config, variant)
    if args.n_bins is not None:
        config = with_n_bins(config, args.n_bins)
    config_id = deepfm_config_id(config)
    if config_id != args.expect_config_id:
        raise SystemExit(
            f"these settings are configuration {config_id}, not {args.expect_config_id}"
        )
    inputs = replace(inputs, config=with_seed(config, args.seed))
    torch.set_num_threads(inputs.base.num_threads)
    result = rescore_trial(inputs, TrialSpec(AREA_V_PHASE, args.target, args.seed, args.epochs))
    peak = finish(inputs, label)
    check_memory_limit(peak, inputs.base.memory.limit_gb, f"deepfm rescore {args.target}")
    if not result.identical:
        raise SystemExit(
            f"NOT IDENTICAL: {result.n_different_rows} of {result.n_saved_rows} saved Top-K rows "
            f"differ (largest score difference {result.max_abs_score_difference!r}). "
            "Stop and report; nothing is changed."
        )
    LOGGER.info(
        "OK: %s seed %d %s: all %d saved Top-K rows are reproduced bit for bit (%d candidate "
        "rows scored again)",
        result.config_id,
        args.seed,
        args.target,
        result.n_saved_rows,
        result.n_scored_rows,
    )


if __name__ == "__main__":
    main()
