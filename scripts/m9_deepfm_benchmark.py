"""CLI for M9: the DeepFM device benchmark on ONE device (proposal §16.6).

    uv run python scripts/m9_deepfm_benchmark.py --device cpu
    uv run python scripts/m9_deepfm_benchmark.py --device mps

Runs the default DeepFM configuration on the training rows of the speed test's
target week: 300 training steps (the first 20 not timed), the same steps again
from the same seed, and the scoring throughput on 1 million rows. Appends one
row to results/deepfm_benchmark.csv. One device per process; run it on a clean
working tree. No label file is opened (the sample files carry their labels).
"""

# Thread setup must run before polars is imported (see hmrec.threads).
from hmrec.threads import configure_polars_threads

configure_polars_threads()

import argparse  # noqa: E402

from hmrec.deepfm.benchmark import BENCHMARK_DEVICES, run_benchmark  # noqa: E402
from hmrec.deepfm.process import finish, record_benchmark, start_process  # noqa: E402
from hmrec.run_records import check_memory_limit  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True, choices=BENCHMARK_DEVICES)
    return parser.parse_args()


def main() -> None:
    """Benchmark one device and record the row."""
    device = parse_args().device
    label = f"benchmark_{device}"
    inputs, stamp = start_process(label)
    result = run_benchmark(inputs, device)
    peak = finish(inputs, label)
    record_benchmark(inputs, result, peak, stamp)
    check_memory_limit(peak, inputs.base.memory.limit_gb, f"deepfm {label}")


if __name__ == "__main__":
    main()
