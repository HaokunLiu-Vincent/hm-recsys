"""The speed test that runs once before the Ranker grid (docs/decisions.md 2026-11-18).

What it does, each part in its own process (hmrec.hpo.v_phase starts them):
    1. samples the training folds of the target fold (V1 <- F1 ... F4) and builds the
       binary Dataset of the default configuration;
    2. trains `comparison_trees` trees with col-wise and with row-wise histograms,
       WITHOUT evaluation, and applies the rule of configs/ranker.yaml
       (hmrec.ranker.train.decide_histogram_mode): only speed and memory decide;
    3. trains the default configuration to `speed_test.num_trees` trees in the
       chosen mode and evaluates the target fold after every checkpoint;
    4. computes the RRF baseline of the target fold.
All rows go to results/m7_speed_test.csv and results/m7_speed_test_metrics.csv.

The summary then says whether the time may be extrapolated at all: if the model's
MAP@12 at the last checkpoint is BELOW the RRF baseline, something is wrong (a
bug is more likely than a weak model), and the run must be investigated first.

configs/ranker.yaml keeps histogram.mode = undecided during the test; the chosen
mode is written into the config by hand afterwards and committed before the grid.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from hmrec.config import HISTOGRAM_COL_WISE, HISTOGRAM_ROW_WISE
from hmrec.hpo.v_phase import GridContext, ensure_dataset, ensure_trial, run_baseline
from hmrec.hpo.v_records import (
    AREA_SPEED_TEST,
    NO_TREES,
    RRF_BASELINE,
    MetricRow,
    config_rows,
    find_trial,
    read_metric_rows,
    read_trial_rows,
    results_files,
)
from hmrec.hpo.v_trial import TrialSpec, model_config, ranker_config_id
from hmrec.ranker.ablation import FULL
from hmrec.ranker.sampling import SampleRecord, read_sample_record, record_file
from hmrec.ranker.train import decide_histogram_mode
from hmrec.ranker.window import m7_training_window

TIMING_STEPS = {
    HISTOGRAM_COL_WISE: "histogram_col_wise",
    HISTOGRAM_ROW_WISE: "histogram_row_wise",
}
EVALUATED_STEP = "speed_test"


@dataclass(frozen=True)
class HistogramTiming:
    """The comparison of the two histogram modes.

    Attributes:
        col_wise_seconds_per_tree: Mean seconds per tree, col-wise (after warm-up).
        row_wise_seconds_per_tree: The same, row-wise.
        col_wise_peak_gb: True peak RSS of the col-wise process.
        row_wise_peak_gb: True peak RSS of the row-wise process.
        chosen_mode: The mode the rule selects.
    """

    col_wise_seconds_per_tree: float
    row_wise_seconds_per_tree: float
    col_wise_peak_gb: float
    row_wise_peak_gb: float
    chosen_mode: str


@dataclass(frozen=True)
class SpeedTestSummary:
    """What the speed test found.

    Attributes:
        timing: The histogram mode comparison.
        trial: The row of the evaluated trial (results/m7_speed_test.csv).
        checkpoints: Its metric rows, one per checkpoint, ascending.
        baseline: The RRF baseline row of the same fold.
        below_baseline: True if the selection metric at the LAST checkpoint is
            below the baseline: stop and investigate, do not extrapolate.
        train_metric_rises: True if the training-set NDCG@12 grows from every
            checkpoint to the next.
    """

    timing: HistogramTiming
    trial: Mapping[str, str]
    checkpoints: tuple[MetricRow, ...]
    baseline: MetricRow
    below_baseline: bool
    train_metric_rises: bool


# ---------------------------------------------------------------- pure parts


def histogram_timing(
    col_wise_row: Mapping[str, str],
    row_wise_row: Mapping[str, str],
    memory_limit_gb: float,
    min_row_wise_speedup: float,
) -> HistogramTiming:
    """Apply the histogram rule to the two timing trials' rows."""
    col_seconds = float(col_wise_row["seconds_per_tree"])
    row_seconds = float(row_wise_row["seconds_per_tree"])
    row_peak = float(row_wise_row["process_max_rss_gb"])
    chosen = decide_histogram_mode(
        col_seconds, row_seconds, row_peak, memory_limit_gb, min_row_wise_speedup
    )
    return HistogramTiming(
        col_wise_seconds_per_tree=col_seconds,
        row_wise_seconds_per_tree=row_seconds,
        col_wise_peak_gb=float(col_wise_row["process_max_rss_gb"]),
        row_wise_peak_gb=row_peak,
        chosen_mode=chosen,
    )


def summarize(
    timing: HistogramTiming,
    trial: Mapping[str, str],
    metric_rows: Sequence[MetricRow],
    fold: str,
    metric: str,
) -> SpeedTestSummary:
    """Put the evaluated trial next to the baseline and derive the two checks.

    Args:
        timing: The histogram comparison.
        trial: The evaluated trial's row.
        metric_rows: All rows of results/m7_speed_test_metrics.csv.
        fold: The target fold.
        metric: The selection metric ("map@12").

    Raises:
        ValueError: If the baseline row or the trial's checkpoints are missing.
    """
    own = config_rows(metric_rows, trial["experiment"], trial["config_id"])
    checkpoint_rows = tuple(
        sorted((row for row in own if row.fold == fold), key=lambda row: row.num_trees)
    )
    baselines = [
        row
        for row in config_rows(metric_rows, RRF_BASELINE, RRF_BASELINE)
        if row.fold == fold and row.num_trees == NO_TREES
    ]
    if not checkpoint_rows or len(baselines) != 1:
        raise ValueError(f"speed test: fold {fold} needs its checkpoints and one baseline row")
    train_values = [row.train_metric for row in checkpoint_rows]
    rises = all(
        earlier is not None and later is not None and later > earlier
        for earlier, later in zip(train_values, train_values[1:], strict=False)
    )
    return SpeedTestSummary(
        timing=timing,
        trial=trial,
        checkpoints=checkpoint_rows,
        baseline=baselines[0],
        below_baseline=checkpoint_rows[-1].scores[metric] < baselines[0].scores[metric],
        train_metric_rises=rises,
    )


def training_data_lines(records: Sequence[SampleRecord]) -> list[str]:
    """One line per training fold: positive share and rows per query (mean, P50, P90)."""
    return [
        f"  {record.fold} r={record.ratio}: {record.n_rows:,} rows, {record.n_queries:,} queries, "
        f"positives {record.n_positives:,} ({record.positive_share:.4%}); rows per query "
        f"mean {record.rows_per_query_mean:.1f}, P50 {record.rows_per_query_p50:.0f}, "
        f"P90 {record.rows_per_query_p90:.0f}"
        for record in records
    ]


def report_lines(
    summary: SpeedTestSummary, records: Sequence[SampleRecord], metrics: Sequence[str]
) -> list[str]:
    """The speed test as plain log lines."""
    timing, trial = summary.timing, summary.trial
    lines = [
        f"histogram: col_wise {timing.col_wise_seconds_per_tree:.3f} s/tree "
        f"(peak {timing.col_wise_peak_gb:.2f} GB), row_wise "
        f"{timing.row_wise_seconds_per_tree:.3f} s/tree (peak {timing.row_wise_peak_gb:.2f} GB) "
        f"-> {timing.chosen_mode}",
        f"evaluated trial ({trial['histogram_mode']}, {trial['num_trees']} trees, "
        f"{int(trial['n_train_rows']):,} training rows): "
        f"{float(trial['seconds_per_tree']):.3f} s/tree, "
        f"train {float(trial['train_seconds']):.1f} s, "
        f"score + evaluate {float(trial['score_seconds']):.1f} s for "
        f"{int(trial['n_scored_rows']):,} rows, true peak "
        f"{float(trial['process_max_rss_gb']):.2f} GB",
        f"LightGBM {trial['lightgbm_version']}; OpenMP runtime {trial['libomp_path']} "
        f"(version {trial['libomp_version']})",
        "training data:",
        *training_data_lines(records),
        "target fold " + trial["fold"] + ":",
    ]
    baseline_text = ", ".join(f"{m} {summary.baseline.scores[m]:.6f}" for m in metrics)
    lines.append(f"  RRF baseline: {baseline_text}")
    for row in summary.checkpoints:
        text = ", ".join(f"{m} {row.scores[m]:.6f}" for m in metrics)
        train = "n/a" if row.train_metric is None else f"{row.train_metric:.6f}"
        lines.append(f"  {row.num_trees} trees: {text} | train ndcg@12 {train}")
    lines.append(f"training-set NDCG@12 rises at every checkpoint: {summary.train_metric_rises}")
    if summary.below_baseline:
        lines.append(
            "STOP: the Ranker is BELOW the RRF baseline at the last checkpoint. Investigate "
            "before anything else; do not extrapolate the time."
        )
    return lines


# ---------------------------------------------------------------- the run


def _spec(context: GridContext, step: str, mode: str, num_trees: int, evaluate: bool) -> TrialSpec:
    """A speed-test trial of the default configuration on the speed test's fold."""
    ranker = context.inputs.ranker
    return TrialSpec(
        area=AREA_SPEED_TEST,
        experiment=FULL,
        step=step,
        target_fold=ranker.speed_test.target_fold,
        tuned=ranker.tuned_defaults,
        num_trees=num_trees,
        histogram_mode=mode,
        evaluate=evaluate,
    )


def _trial_row(context: GridContext, spec: TrialSpec) -> Mapping[str, str]:
    """The recorded row of a speed-test trial."""
    inputs = context.inputs
    config = model_config(inputs.ranker, spec.tuned, spec.experiment, spec.histogram_mode)
    trial_file, _ = results_files(inputs.paths, AREA_SPEED_TEST)
    row = find_trial(
        read_trial_rows(trial_file),
        spec.experiment,
        ranker_config_id(config),
        spec.target_fold,
        spec.step,
    )
    if row is None:
        raise ValueError(f"speed test: the trial of step {spec.step} left no row")
    return row


def training_records(context: GridContext) -> list[SampleRecord]:
    """The sample records of the speed test's training folds (default ratio)."""
    inputs = context.inputs
    ranker = inputs.ranker
    folds = m7_training_window(inputs.manifest, ranker.speed_test.target_fold, ranker.window_weeks)
    ratio = ranker.tuned_defaults.negatives_per_positive
    return [
        read_sample_record(record_file(inputs.paths.ranker_sample_file(FULL, fold.name, ratio)))
        for fold in folds
    ]


def run_speed_test(context: GridContext) -> SpeedTestSummary:
    """Run every missing part of the speed test and return its summary."""
    inputs = context.inputs
    ranker = inputs.ranker
    target_fold = ranker.speed_test.target_fold
    ensure_dataset(context, FULL, target_fold, ranker.tuned_defaults.negatives_per_positive)
    timing_specs = {
        mode: _spec(context, step, mode, ranker.histogram.comparison_trees, evaluate=False)
        for mode, step in TIMING_STEPS.items()
    }
    for spec in timing_specs.values():
        ensure_trial(context, spec, match_step=True)
    timing = histogram_timing(
        _trial_row(context, timing_specs[HISTOGRAM_COL_WISE]),
        _trial_row(context, timing_specs[HISTOGRAM_ROW_WISE]),
        inputs.base.memory.limit_gb,
        ranker.histogram.min_row_wise_speedup,
    )
    evaluated = _spec(
        context, EVALUATED_STEP, timing.chosen_mode, ranker.speed_test.num_trees, evaluate=True
    )
    ensure_trial(context, evaluated, match_step=True)
    run_baseline(context, AREA_SPEED_TEST, [target_fold])
    _, metrics_file = results_files(inputs.paths, AREA_SPEED_TEST)
    return summarize(
        timing,
        _trial_row(context, evaluated),
        read_metric_rows(metrics_file, context.grid.metrics),
        target_fold,
        context.grid.metric,
    )
