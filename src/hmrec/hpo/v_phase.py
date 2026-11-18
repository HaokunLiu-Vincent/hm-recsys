"""The Ranker's V phase: grid, selection and the driver (proposal §9.4, §9.5).

Reads configs/hpo/v_phase.yaml. For each step, in order:
    1. every configuration of the step is run on every target fold (V1 ... V4);
       each (configuration, fold) is a trial in its own process, and a trial that
       is already recorded is not run again (the run can be resumed);
    2. every configuration gets its own number of trees k* from its pooled
       MAP@12 curve (hmrec.hpo.selection.choose_num_trees);
    3. the configurations are compared at their own k* with the §4.2
       multi-candidate rule (hmrec.hpo.selection.choose_winner);
    4. the step's rows are appended to results/v_phase_steps.csv; if the winner
       needs all trees the run stops for a human decision.
The winner of a step is the default of the next step.

Every decision is computed from results/v_phase_metrics.csv alone, so it can be
replayed from the files in git.

T folds: the grid's target folds must be V folds, and every process this module
starts checks its folds again (hmrec.ranker.window).
"""

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb

from hmrec.config import (
    HISTOGRAM_UNDECIDED,
    RANKER_TUNED_KEYS,
    RankerTunedParams,
    SelectionRule,
    check_keys,
    get_bool,
    get_float,
    get_str,
    get_str_list,
    load_base_config,
    load_ranker_config,
    load_selection_rule,
    load_yaml,
    parse_ranker_tuned_params,
)
from hmrec.csv_records import append_row, read_rows
from hmrec.data.folds import load_fold_manifest
from hmrec.eval.report import parse_metric
from hmrec.git_state import git_state, warn_uncommitted_changes
from hmrec.hpo.grid import Step, ordered_candidate_configs, parse_step
from hmrec.hpo.selection import (
    NumTreesChoice,
    StepDecision,
    TrialScores,
    choose_num_trees,
    choose_winner,
    count_folds_better,
)
from hmrec.hpo.v_records import (
    AREA_SPEED_TEST,
    AREA_V_PHASE,
    NO_TREES,
    RRF_BASELINE,
    SPEED_TEST_STEP,
    MetricRow,
    complete_tree_counts,
    config_rows,
    equal_fold_score,
    find_trial,
    pooled_curve,
    read_metric_rows,
    read_trial_rows,
    results_files,
    rows_at,
    scores_at,
)
from hmrec.hpo.v_trial import TrialInputs, TrialSpec, model_config, ranker_config_id
from hmrec.locks import file_sha256, load_features_lock
from hmrec.logging_utils import get_logger, setup_logging
from hmrec.paths import ProjectPaths, get_paths
from hmrec.ranker.ablation import FULL, MINUS_R6, build_experiment, wide_table_columns
from hmrec.ranker.dataset import dataset_is_built
from hmrec.ranker.predict import checkpoints
from hmrec.ranker.runtime import read_lightgbm_runtime
from hmrec.ranker.sampling import SampleJob, record_file
from hmrec.ranker.train import first_trees_sha256, gain_importance, load_model
from hmrec.ranker.window import (
    M7_TRAINING_PHASES,
    m7_sampling_folds,
    m7_training_window,
    require_m7_target_fold,
    require_m7_training_fold,
)
from hmrec.run_records import RunStamp

LOGGER = get_logger("hpo.v_phase")

PHASE = "V"
RANKER_COMPONENT = "ranker"
KNOWN_EXPERIMENTS = (RRF_BASELINE, MINUS_R6)

SAMPLE_SCRIPT = "m7_sample.py"
DATASET_SCRIPT = "m7_build_dataset.py"
TRIAL_SCRIPT = "m7_train_eval.py"
BASELINE_SCRIPT = "m7_rrf_baseline.py"

IMPORTANCE_COLUMNS = (
    "config_id",
    "fold",
    "num_trees",
    "feature",
    "gain",
    "gain_share",
    "git_commit",
)

# Runs a script in a new process: (script path, arguments) -> seconds (hmrec.pipeline.run_script).
Launcher = Callable[[Path, Sequence[str]], float]


# ---------------------------------------------------------------- the grid file


@dataclass(frozen=True)
class VPhaseGrid:
    """configs/hpo/v_phase.yaml.

    Attributes:
        target_folds: The folds every configuration is evaluated on (V1 ... V4).
        metric: The selection metric ("map@12").
        extra_metrics: Reported next to it, never used to select.
        stop_if_winner_uses_all_trees: Stop after a step whose winner's k* is the
            last checkpoint.
        steps: The steps in order; candidates are in priority order.
        experiments: System-level experiments ("rrf_baseline", "minus_r6").
        g1_min_pooled_score: Goal G1: the minimum pooled selection metric (0.032).
    """

    target_folds: tuple[str, ...]
    metric: str
    extra_metrics: tuple[str, ...]
    stop_if_winner_uses_all_trees: bool
    steps: tuple[Step, ...]
    experiments: tuple[str, ...]
    g1_min_pooled_score: float

    @property
    def metrics(self) -> tuple[str, ...]:
        """All computed metrics, the selection metric first."""
        return (self.metric, *self.extra_metrics)


def _parse_steps(raw_steps: Any) -> tuple[Step, ...]:
    """The steps: only "select" steps in listed order, overriding tuned parameters only."""
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError("v_phase.steps must be a non-empty list of steps")
    steps = tuple(
        parse_step(raw_step, f"v_phase.steps[{position}]")
        for position, raw_step in enumerate(raw_steps)
    )
    for step in steps:
        if step.kind != "select" or step.order != "listed" or step.requires:
            raise ValueError(f"step {step.name}: the V phase has only listed select steps")
        for candidate in step.candidates:
            unknown = set(candidate) - set(RANKER_TUNED_KEYS)
            if unknown:
                raise ValueError(f"step {step.name}: {sorted(unknown)} are not tuned parameters")
    names = [step.name for step in steps]
    if len(set(names)) != len(names):
        raise ValueError(f"v_phase.steps: step names repeat: {names}")
    return steps


def parse_v_phase_grid(raw: Mapping[str, Any]) -> VPhaseGrid:
    """Build a VPhaseGrid from the parsed YAML.

    Raises:
        ValueError: If a key is missing or unknown, or a value is not allowed.
    """
    where = "v_phase"
    expected = {
        "phase",
        "target_folds",
        "metric",
        "extra_metrics",
        "stop_if_winner_uses_all_trees",
        "steps",
        "experiments",
        "g1_min_pooled_score",
    }
    check_keys(raw, expected, where)
    if get_str(raw, "phase", where) != PHASE:
        raise ValueError(f"{where}.phase must be {PHASE!r}")
    grid = VPhaseGrid(
        target_folds=get_str_list(raw, "target_folds", where),
        metric=get_str(raw, "metric", where),
        extra_metrics=get_str_list(raw, "extra_metrics", where),
        stop_if_winner_uses_all_trees=get_bool(raw, "stop_if_winner_uses_all_trees", where),
        steps=_parse_steps(raw["steps"]),
        experiments=get_str_list(raw, "experiments", where),
        g1_min_pooled_score=get_float(raw, "g1_min_pooled_score", where),
    )
    for metric in grid.metrics:
        parse_metric(metric)
    unknown = set(grid.experiments) - set(KNOWN_EXPERIMENTS)
    if unknown:
        raise ValueError(f"{where}.experiments: unknown {sorted(unknown)}")
    return grid


def load_v_phase_grid(path: Path) -> VPhaseGrid:
    """Read and validate configs/hpo/v_phase.yaml."""
    return parse_v_phase_grid(load_yaml(path))


def check_grid(grid: VPhaseGrid, rule: SelectionRule, inputs: TrialInputs) -> None:
    """Checks that need the other configs: metric, number of folds, fold phases, ratios.

    Raises:
        ValueError: If the grid's metric is not the rule's Ranker metric, the
            number of target folds is not the rule's, a target fold is not a V
            fold, or a step uses a ratio that is not sampled.
    """
    if grid.metric != rule.selection_metric[RANKER_COMPONENT]:
        raise ValueError(
            f"v_phase.metric is {grid.metric!r}, but selection_rule.yaml says "
            f"{rule.selection_metric[RANKER_COMPONENT]!r}"
        )
    if len(grid.target_folds) != rule.n_folds:
        raise ValueError(f"v_phase.target_folds must name {rule.n_folds} folds")
    for fold_name in grid.target_folds:
        require_m7_target_fold(inputs.manifest.get(fold_name))
    for step in grid.steps:
        for candidate in step.candidates:
            ratio = candidate.get("negatives_per_positive")
            if ratio is not None and ratio not in inputs.ranker.sampling.ratios:
                raise ValueError(f"step {step.name}: ratio {ratio} is not in sampling.ratios")


# ---------------------------------------------------------------- selection (pure)


@dataclass(frozen=True)
class ConfigScores:
    """A configuration, its number of trees and its scores there.

    Attributes:
        tuned: The configuration.
        config_id: Its id.
        choice: Its number of trees k* and the curve's best point.
        scores: Per-fold and pooled selection metric with k* trees.
        equal_fold: Plain mean of the fold scores with k* trees (reported only).
    """

    tuned: RankerTunedParams
    config_id: str
    choice: NumTreesChoice
    scores: TrialScores
    equal_fold: float


def config_scores(
    metric_rows: Sequence[MetricRow],
    experiment: str,
    tuned: RankerTunedParams,
    config_id: str,
    grid: VPhaseGrid,
    tree_counts: Sequence[int],
    tolerance: float,
) -> ConfigScores:
    """Choose a configuration's number of trees and collect its scores there.

    Args:
        metric_rows: Rows of results/v_phase_metrics.csv.
        experiment: "full".
        tuned: The configuration.
        config_id: Its id.
        grid: The V-phase grid (folds, metric).
        tree_counts: The checkpoints every fold must have (100, ..., num_trees).
        tolerance: Relative tolerance of the number-of-trees rule (0.01).

    Raises:
        ValueError: If a (fold, checkpoint) is missing: a trial has not finished.
    """
    rows = config_rows(metric_rows, experiment, config_id)
    present = complete_tree_counts(rows, grid.target_folds)
    if present != sorted(tree_counts):
        raise ValueError(
            f"configuration {config_id}: every fold of {list(grid.target_folds)} needs the "
            f"checkpoints {list(tree_counts)}, complete are {present}"
        )
    curve = pooled_curve(rows, grid.target_folds, grid.metric)
    choice = choose_num_trees(curve, tolerance)
    fold_rows = rows_at(rows, grid.target_folds, choice.num_trees)
    return ConfigScores(
        tuned=tuned,
        config_id=config_id,
        choice=choice,
        scores=scores_at(rows, grid.target_folds, grid.metric, choice.num_trees, config_id),
        equal_fold=equal_fold_score(fold_rows, grid.metric),
    )


@dataclass(frozen=True)
class StepResult:
    """The outcome of one step.

    Attributes:
        step: Step name.
        default: The step's default configuration.
        candidates: The other configurations, in priority order.
        decision: The §4.2 decision (one check per candidate).
        winner: The selected configuration (the default if nothing passed).
        stop: True if the run must stop after this step (the winner uses all trees).
    """

    step: str
    default: ConfigScores
    candidates: tuple[ConfigScores, ...]
    decision: StepDecision
    winner: ConfigScores
    stop: bool


def decide_step(
    step: str,
    default: ConfigScores,
    candidates: Sequence[ConfigScores],
    rule: SelectionRule,
    stop_if_winner_uses_all_trees: bool,
) -> StepResult:
    """Compare the candidates with the default, each at its own number of trees."""
    decision = choose_winner(default.scores, [candidate.scores for candidate in candidates], rule)
    by_id = {config.config_id: config for config in (default, *candidates)}
    winner = by_id[decision.winner_trial_id]
    stop = stop_if_winner_uses_all_trees and winner.choice.uses_all_trees
    return StepResult(step, default, tuple(candidates), decision, winner, stop)


def step_columns(folds: Sequence[str]) -> tuple[str, ...]:
    """Header of results/v_phase_steps.csv for these folds."""
    return (
        "step",
        "role",
        "priority",
        "config_id",
        "params_json",
        "num_trees",
        "uses_all_trees",
        "best_num_trees",
        "score_best",
        "score_pooled",
        "score_equal_fold",
        *(f"score_{fold}" for fold in folds),
        "relative_improvement",
        "folds_better",
        "passes",
        "selected",
        "action",
        "reason",
        "git_commit",
        "decided_at",
    )


def step_rows(
    result: StepResult, folds: Sequence[str], git_commit: str, decided_at: str
) -> list[dict[str, str]]:
    """One row per configuration of a step, the default first (all values as text)."""
    checks = {check.trial_id: check for check in result.decision.checks}
    rows = []
    for priority, config in enumerate((result.default, *result.candidates)):
        check = checks.get(config.config_id)
        row = {
            "step": result.step,
            "role": "default" if priority == 0 else "candidate",
            "priority": str(priority),
            "config_id": config.config_id,
            "params_json": json.dumps(asdict(config.tuned), sort_keys=True),
            "num_trees": str(config.choice.num_trees),
            "uses_all_trees": str(config.choice.uses_all_trees),
            "best_num_trees": str(config.choice.best_num_trees),
            "score_best": repr(config.choice.best_score),
            "score_pooled": repr(config.scores.pooled_score),
            "score_equal_fold": repr(config.equal_fold),
            **{f"score_{fold}": repr(config.scores.fold_scores[fold]) for fold in folds},
            "relative_improvement": "" if check is None else repr(check.relative_improvement),
            "folds_better": "" if check is None else str(check.folds_better),
            "passes": "" if check is None else str(check.passes),
            "selected": str(config.config_id == result.winner.config_id),
            "action": result.decision.action,
            "reason": result.decision.reason,
            "git_commit": git_commit,
            "decided_at": decided_at,
        }
        rows.append(row)
    return rows


def record_step(path: Path, result: StepResult, folds: Sequence[str], git_commit: str) -> None:
    """Append a step's rows, unless the step is already recorded with the same winner.

    Raises:
        ValueError: If the step is recorded with ANOTHER winner: the results
            changed after a decision was written, which must never happen.
    """
    columns = step_columns(folds)
    recorded = [row for row in read_rows(path, columns) if row["step"] == result.step]
    if recorded:
        winners = [row["config_id"] for row in recorded if row["selected"] == "True"]
        if winners != [result.winner.config_id]:
            raise ValueError(
                f"step {result.step} is recorded with winner {winners}, but the results now "
                f"give {result.winner.config_id}"
            )
        return
    decided_at = datetime.now().isoformat(timespec="seconds")
    for row in step_rows(result, folds, git_commit, decided_at):
        append_row(path, columns, row)


# ---------------------------------------------------------------- driver: start processes


@dataclass(frozen=True)
class GridContext:
    """What the driver needs.

    Attributes:
        inputs: Paths and configs.
        grid: configs/hpo/v_phase.yaml.
        rule: configs/selection_rule.yaml.
        scripts_dir: The folder of the m7_*.py scripts.
        launch: Runs a script in a new process and waits (hmrec.pipeline.run_script).
        git_commit: HEAD of the driver.
        check_speed_test_model: True = when the grid has trained the default
            configuration on the speed test's fold, its first trees must equal the
            speed test's model (check_default_against_speed_test). Only tests that
            train no real model switch it off.
    """

    inputs: TrialInputs
    grid: VPhaseGrid
    rule: SelectionRule
    scripts_dir: Path
    launch: Launcher
    git_commit: str
    check_speed_test_model: bool = True


def refuse_to_launch(script: Path, arguments: Sequence[str]) -> float:
    """A launcher for read-only stages: any attempt to start a process is an error."""
    raise RuntimeError(
        f"the grid is not finished: {script.name} {' '.join(arguments)} has not run yet"
    )


def load_m7_context(paths: ProjectPaths, launch: Launcher, git_commit: str) -> GridContext:
    """Read every config M7 needs and check that they fit together.

    Args:
        paths: Project paths.
        launch: How the driver starts a script (hmrec.pipeline.run_script, or
            refuse_to_launch for a process that must not start others).
        git_commit: HEAD of this run.
    """
    grid = load_v_phase_grid(paths.v_phase_grid_file)
    inputs = TrialInputs(
        paths=paths,
        base=load_base_config(paths.base_config_file),
        ranker=load_ranker_config(paths.ranker_config_file),
        manifest=load_fold_manifest(paths.fold_manifest_file),
        lock=load_features_lock(paths.features_lock_file),
        metrics=grid.metrics,
    )
    rule = load_selection_rule(paths.selection_rule_file)
    check_grid(grid, rule, inputs)
    return GridContext(inputs, grid, rule, paths.project_root / "scripts", launch, git_commit)


def start_m7_run(run_name: str, launch: Launcher) -> tuple[GridContext, RunStamp]:
    """Common start of every M7 script: logging, git state, LightGBM / OpenMP runtime, configs.

    Args:
        run_name: Name of the run's log folder, artifacts/runs/{run_name}/.
        launch: How this process starts other scripts (refuse_to_launch if it must not).

    Returns:
        (the context, the code state of this run).
    """
    paths = get_paths()
    setup_logging(log_file=paths.run_dir(run_name) / "run.log")
    commit, dirty = git_state(paths.project_root, paths.results_root)
    LOGGER.info("git HEAD %s, uncommitted code/config changes: %s", commit, dirty)
    if dirty:
        warn_uncommitted_changes(commit)
    # Fails here if a second OpenMP runtime (e.g. torch's) is loaded in this process.
    runtime = read_lightgbm_runtime()
    LOGGER.info(
        "LightGBM %s; OpenMP runtime %s (version %s)",
        runtime.lightgbm_version,
        runtime.libomp_path,
        runtime.libomp_version,
    )
    return load_m7_context(paths, launch, commit), RunStamp(commit, dirty)


def sample_job(
    inputs: TrialInputs, fold_name: str, experiment: str, ratios: Sequence[int], git_commit: str
) -> SampleJob:
    """The sampling job of one M7 training fold. The fold's phase is checked first.

    Raises:
        ValueError: If the fold is not an F or V fold, or a ratio is not configured.
    """
    fold = inputs.manifest.get(fold_name)
    require_m7_training_fold(fold)
    unknown = [ratio for ratio in ratios if ratio not in inputs.ranker.sampling.ratios]
    if unknown:
        raise ValueError(f"ratios {unknown} are not in sampling.ratios of configs/ranker.yaml")
    paths = inputs.paths
    return SampleJob(
        fold=fold,
        features_dir=paths.features_dir(fold_name),
        labels_file=paths.labels_file(fold_name),
        experiment=build_experiment(experiment, inputs.lock),
        wide_columns=wide_table_columns(inputs.lock),
        sample_files={
            ratio: paths.ranker_sample_file(experiment, fold_name, ratio) for ratio in ratios
        },
        allowed_phases=M7_TRAINING_PHASES,
        git_commit=git_commit,
    )


def ensure_samples(
    context: GridContext, experiment: str, fold_names: Sequence[str], ratios: Sequence[int]
) -> None:
    """Start the sampling process of every fold that misses one of the ratios' samples."""
    paths = context.inputs.paths
    for fold_name in fold_names:
        finished = [
            record_file(paths.ranker_sample_file(experiment, fold_name, ratio)).exists()
            for ratio in ratios
        ]
        if all(finished):
            continue
        arguments = ["--fold", fold_name, "--experiment", experiment, "--ratios"]
        arguments += [str(ratio) for ratio in ratios]
        context.launch(context.scripts_dir / SAMPLE_SCRIPT, arguments)


def ensure_dataset(context: GridContext, experiment: str, target_fold: str, ratio: int) -> None:
    """Build the binary Dataset of (experiment, target fold, ratio) unless it exists."""
    inputs = context.inputs
    if dataset_is_built(inputs.paths.ranker_dataset_dir(experiment, target_fold, ratio)):
        return
    folds = m7_training_window(inputs.manifest, target_fold, inputs.ranker.window_weeks)
    # The full experiment writes all ratios in one pass over a fold; an ablation only its own.
    ratios = inputs.ranker.sampling.ratios if experiment == FULL else (ratio,)
    ensure_samples(context, experiment, [fold.name for fold in folds], ratios)
    arguments = ["--target", target_fold, "--experiment", experiment, "--ratio", str(ratio)]
    context.launch(context.scripts_dir / DATASET_SCRIPT, arguments)


def trial_arguments(spec: TrialSpec) -> list[str]:
    """Command-line arguments of scripts/m7_train_eval.py for one trial."""
    arguments = [
        "--area", spec.area,
        "--experiment", spec.experiment,
        "--step", spec.step,
        "--target", spec.target_fold,
        "--num-leaves", str(spec.tuned.num_leaves),
        "--min-data-in-leaf", str(spec.tuned.min_data_in_leaf),
        "--ratio", str(spec.tuned.negatives_per_positive),
        "--num-trees", str(spec.num_trees),
        "--histogram-mode", spec.histogram_mode,
    ]  # fmt: skip
    if not spec.evaluate:
        arguments.append("--no-eval")
    return arguments


def ensure_trial(context: GridContext, spec: TrialSpec, match_step: bool) -> None:
    """Run a trial in its own process unless its row is already recorded.

    Args:
        context: The driver's context.
        spec: The trial.
        match_step: True = a row of another step does not count (speed test).
    """
    inputs = context.inputs
    config = model_config(inputs.ranker, spec.tuned, spec.experiment, spec.histogram_mode)
    trial_file, _ = results_files(inputs.paths, spec.area)
    step = spec.step if match_step else None
    recorded = find_trial(
        read_trial_rows(trial_file),
        spec.experiment,
        ranker_config_id(config),
        spec.target_fold,
        step,
    )
    if recorded is not None:
        LOGGER.info("already recorded, skipped: %s %s", spec.step, " ".join(trial_arguments(spec)))
        return
    ensure_dataset(context, spec.experiment, spec.target_fold, spec.tuned.negatives_per_positive)
    context.launch(context.scripts_dir / TRIAL_SCRIPT, trial_arguments(spec))


# ---------------------------------------------------------------- driver: the grid


def decided_histogram_mode(context: GridContext) -> str:
    """The histogram mode of configs/ranker.yaml; the grid refuses to run while it is undecided."""
    mode = context.inputs.ranker.histogram.mode
    if mode == HISTOGRAM_UNDECIDED:
        raise ValueError(
            "histogram.mode in configs/ranker.yaml is still 'undecided'. Run the speed test, "
            "write the chosen mode into the config and commit before any grid trial."
        )
    return mode


def _config_id(context: GridContext, tuned: RankerTunedParams, experiment: str) -> str:
    """The id of a grid configuration."""
    mode = decided_histogram_mode(context)
    return ranker_config_id(model_config(context.inputs.ranker, tuned, experiment, mode))


def _read_metrics(context: GridContext) -> list[MetricRow]:
    """The grid's metric rows."""
    _, metrics_file = results_files(context.inputs.paths, AREA_V_PHASE)
    return read_metric_rows(metrics_file, context.grid.metrics)


def _recorded_model(
    context: GridContext, area: str, config_id: str, fold: str, step: str | None
) -> tuple[lgb.Booster, Mapping[str, str]]:
    """(the saved model, its trial row) of a recorded trial; the file must be the recorded one."""
    paths = context.inputs.paths
    trial_file, _ = results_files(paths, area)
    row = find_trial(read_trial_rows(trial_file), FULL, config_id, fold, step)
    if row is None:
        raise ValueError(
            f"no recorded trial of configuration {config_id} on {fold} in {trial_file}"
        )
    model_file = paths.ranker_model_file(area, FULL, config_id, fold)
    if file_sha256(model_file) != row["model_sha256"]:
        raise ValueError(f"{model_file} is not the model recorded in {trial_file}")
    return load_model(model_file), row


def check_default_against_speed_test(context: GridContext) -> str:
    """The grid's default model on the speed test's fold must START with the speed test's model.

    The speed test trained the default configuration to speed_test.num_trees trees
    (300). The grid trains the same configuration on the same fold to num_trees.
    Training is deterministic, so the first 300 trees must be the same trees. If
    they are not, training is not reproducible (another LightGBM or libomp, another
    Dataset, a changed parameter): the run stops before more trials are spent.

    Returns:
        The sha256 of the text of the common first trees.

    Raises:
        ValueError: If the speed test has no evaluated trial of the grid's default
            configuration (it ran with another histogram mode or not at all), a
            model file is not the recorded one, or the first trees differ.
    """
    ranker = context.inputs.ranker
    fold = ranker.speed_test.target_fold
    config_id = _config_id(context, ranker.tuned_defaults, FULL)
    speed_model, speed_row = _recorded_model(
        context, AREA_SPEED_TEST, config_id, fold, SPEED_TEST_STEP
    )
    grid_model, _ = _recorded_model(context, AREA_V_PHASE, config_id, fold, None)
    n_trees = int(speed_row["num_trees"])
    speed_sha256 = first_trees_sha256(speed_model, n_trees)
    grid_sha256 = first_trees_sha256(grid_model, n_trees)
    if speed_sha256 != grid_sha256:
        raise ValueError(
            f"the first {n_trees} trees of the grid's default model on {fold} ({grid_sha256[:12]}) "
            f"differ from the speed test's model ({speed_sha256[:12]}): training is not "
            "reproducible. STOPPED; no further trial is run."
        )
    LOGGER.info(
        "first %d trees of the grid's default model on %s equal the speed test's model (%s)",
        n_trees,
        fold,
        speed_sha256[:12],
    )
    return speed_sha256


def run_step(context: GridContext, step: Step, default: RankerTunedParams) -> StepResult:
    """Run every missing trial of a step, then decide it.

    Args:
        context: The driver's context.
        step: The step.
        default: The step's default configuration (the previous step's winner).
    """
    ranker = context.inputs.ranker
    mode = decided_histogram_mode(context)
    raw_candidates = ordered_candidate_configs(step, asdict(default))
    candidates = [parse_ranker_tuned_params(raw, step.name) for raw in raw_candidates]
    for tuned in (default, *candidates):
        for fold_name in context.grid.target_folds:
            spec = TrialSpec(
                AREA_V_PHASE, FULL, step.name, fold_name, tuned, ranker.num_trees, mode, True
            )
            ensure_trial(context, spec, match_step=False)
            is_speed_test_trial = (
                tuned == ranker.tuned_defaults and fold_name == ranker.speed_test.target_fold
            )
            if context.check_speed_test_model and is_speed_test_trial:
                # Right after this trial, before any other trial is spent.
                check_default_against_speed_test(context)
    metric_rows = _read_metrics(context)
    tree_counts = checkpoints(ranker.num_trees, ranker.checkpoint_every)
    scored = [
        config_scores(
            metric_rows,
            FULL,
            tuned,
            _config_id(context, tuned, FULL),
            context.grid,
            tree_counts,
            context.rule.min_relative_improvement,
        )
        for tuned in (default, *candidates)
    ]
    return decide_step(
        step.name, scored[0], scored[1:], context.rule, context.grid.stop_if_winner_uses_all_trees
    )


def run_grid(context: GridContext, last_step: str | None) -> list[StepResult]:
    """Run the steps in order up to `last_step` (None = all), recording every decision.

    Returns:
        The results of the steps that ran. The run stops early after a step
        whose winner uses all trees.
    """
    names = [step.name for step in context.grid.steps]
    if last_step is not None and last_step not in names:
        raise ValueError(f"unknown step {last_step!r}; the grid has {names}")
    current = context.inputs.ranker.tuned_defaults
    results = []
    for step in context.grid.steps:
        result = run_step(context, step, current)
        record_step(
            context.inputs.paths.v_phase_steps_file,
            result,
            context.grid.target_folds,
            context.git_commit,
        )
        results.append(result)
        LOGGER.info(
            "step %s: %s -> %s with %d trees (%s)",
            step.name,
            result.decision.action,
            asdict(result.winner.tuned),
            result.winner.choice.num_trees,
            result.decision.reason,
        )
        if result.stop:
            LOGGER.warning(
                "step %s: the winner's number of trees is the last checkpoint. STOPPED: a "
                "human decides whether to train more trees.",
                step.name,
            )
            break
        current = result.winner.tuned
        if step.name == last_step:
            break
    return results


# ---------------------------------------------------------------- driver: baseline, ablation


def run_baseline(context: GridContext, area: str, fold_names: Sequence[str]) -> None:
    """Compute the RRF baseline of every fold that has no baseline row in the area yet."""
    _, metrics_file = results_files(context.inputs.paths, area)
    rows = config_rows(
        read_metric_rows(metrics_file, context.grid.metrics), RRF_BASELINE, RRF_BASELINE
    )
    done = {row.fold for row in rows}
    for fold_name in fold_names:
        if fold_name not in done:
            arguments = ["--fold", fold_name, "--area", area]
            context.launch(context.scripts_dir / BASELINE_SCRIPT, arguments)


def run_minus_r6(context: GridContext, winner: ConfigScores) -> None:
    """Train and evaluate the final configuration without R6, to the selected number of trees."""
    inputs = context.inputs
    mode = decided_histogram_mode(context)
    ratio = winner.tuned.negatives_per_positive
    folds = m7_sampling_folds(
        inputs.manifest, context.grid.target_folds, inputs.ranker.window_weeks
    )
    ensure_samples(context, MINUS_R6, [fold.name for fold in folds], (ratio,))
    for fold_name in context.grid.target_folds:
        spec = TrialSpec(
            AREA_V_PHASE,
            MINUS_R6,
            MINUS_R6,
            fold_name,
            winner.tuned,
            winner.choice.num_trees,
            mode,
            True,
        )
        ensure_trial(context, spec, match_step=False)


@dataclass(frozen=True)
class SystemComparison:
    """The final configuration against the RRF baseline and the system without R6.

    Attributes:
        num_trees: The selected number of trees.
        full: The final configuration's scores with num_trees trees.
        baseline: The RRF baseline's scores.
        minus_r6: The scores without R6, or None if that experiment has not run.
        folds_above_baseline: Folds where `full` is strictly above the baseline.
    """

    num_trees: int
    full: TrialScores
    baseline: TrialScores
    minus_r6: TrialScores | None
    folds_above_baseline: int


def compare_systems(
    context: GridContext, winner: ConfigScores, metric_rows: Sequence[MetricRow]
) -> SystemComparison:
    """Collect the selection metric of the three systems on the target folds."""
    folds, metric = context.grid.target_folds, context.grid.metric
    k = winner.choice.num_trees
    baseline_rows = config_rows(metric_rows, RRF_BASELINE, RRF_BASELINE)
    baseline = scores_at(baseline_rows, folds, metric, NO_TREES, RRF_BASELINE)
    minus_r6_id = _config_id(context, winner.tuned, MINUS_R6)
    minus_r6_rows = config_rows(metric_rows, MINUS_R6, minus_r6_id)
    minus_r6 = None
    if k in complete_tree_counts(minus_r6_rows, folds):
        minus_r6 = scores_at(minus_r6_rows, folds, metric, k, minus_r6_id)
    return SystemComparison(
        num_trees=k,
        full=winner.scores,
        baseline=baseline,
        minus_r6=minus_r6,
        folds_above_baseline=count_folds_better(baseline, winner.scores),
    )


def goal_lines(comparison: SystemComparison, rule: SelectionRule, g1_target: float) -> list[str]:
    """Plain-language lines for G1 and G3 (proposal §0.1)."""
    full, baseline = comparison.full, comparison.baseline
    g1 = "met" if full.pooled_score >= g1_target else "NOT met"
    g3 = "met" if comparison.folds_above_baseline >= rule.min_folds_better else "NOT met"
    lines = [
        f"G1 {g1}: pooled MAP@12 {full.pooled_score:.6f} with {comparison.num_trees} trees "
        f"(target >= {g1_target})",
        f"G3 {g3}: above the RRF baseline on {comparison.folds_above_baseline} of "
        f"{len(full.fold_scores)} folds (baseline pooled {baseline.pooled_score:.6f})",
    ]
    for fold, score in full.fold_scores.items():
        lines.append(f"  {fold}: ranker {score:.6f} | rrf {baseline.fold_scores[fold]:.6f}")
    if comparison.minus_r6 is not None:
        lines.append(f"without R6: pooled MAP@12 {comparison.minus_r6.pooled_score:.6f}")
    return lines


def final_winner(context: GridContext) -> ConfigScores:
    """The winner of the last step, from the recorded results only.

    The grid is replayed with a launcher that refuses to start anything, so this
    fails (instead of training) if a trial of the grid is missing.

    Raises:
        RuntimeError: If the grid is not finished.
        ValueError: If the grid stopped early (a winner needs all trees).
    """
    replay = replace(context, launch=refuse_to_launch)
    results = run_grid(replay, None)
    if len(results) != len(context.grid.steps) or results[-1].stop:
        raise ValueError("the grid stopped early; there is no final configuration yet")
    return results[-1].winner


# ---------------------------------------------------------------- driver: feature importance


def importance_rows(context: GridContext, winner: ConfigScores) -> list[dict[str, str]]:
    """Gain of every feature in the final configuration's 4 models, first k* trees (§8.7).

    For analysis only: features are locked and nothing is selected from this.
    """
    paths = context.inputs.paths
    rows = []
    for fold_name in context.grid.target_folds:
        model_file = paths.ranker_model_file(AREA_V_PHASE, FULL, winner.config_id, fold_name)
        gains = gain_importance(load_model(model_file), winner.choice.num_trees)
        total = sum(gains.values())
        for feature, gain in gains.items():
            rows.append(
                {
                    "config_id": winner.config_id,
                    "fold": fold_name,
                    "num_trees": str(winner.choice.num_trees),
                    "feature": feature,
                    "gain": repr(gain),
                    "gain_share": repr(gain / total if total > 0 else 0.0),
                    "git_commit": context.git_commit,
                }
            )
    return rows


def write_importance(context: GridContext, winner: ConfigScores) -> None:
    """Append the importance rows unless this configuration's rows are already there."""
    path = context.inputs.paths.v_phase_feature_importance_file
    recorded = read_rows(path, IMPORTANCE_COLUMNS)
    if any(row["config_id"] == winner.config_id for row in recorded):
        return
    for row in importance_rows(context, winner):
        append_row(path, IMPORTANCE_COLUMNS, row)
