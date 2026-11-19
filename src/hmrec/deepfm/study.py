"""The DeepFM part of the M9 study on V1-V4: its rules, applied to recorded rows (proposal §16.8).

Nothing here trains or reads data: the functions take rows of
results/deepfm_metrics.csv and say what the rules decide. Training happens in
one process per (configuration, seed, fold); hmrec.deepfm.study_run starts them.

A configuration is a Variant: the seven settings the study changes. One of them
is the number of epochs T, the length of the run's cosine schedule: a run of T
epochs is its own model and is evaluated once, after its last epoch.

Scores. For a configuration, a seed s and a fold f, MAP_f(s) is the recorded
MAP@12 of the final model.

    pooled(s)   = sum_f n_f * MAP_f(s) / sum_f n_f        (query-weighted over the folds)
    pooled      = mean over the seeds of pooled(s)         (the configuration's score)
    fold score  = mean over the seeds of MAP_f(s)
    seed range  = (max_s pooled(s) - min_s pooled(s)) / min_s pooled(s)

With one seed the means are that seed's values.

Epoch rule. For the same settings with T = 1, 2, 3: the smallest T whose pooled
score is within `epoch_tolerance` of the best T (hmrec.hpo.selection.choose_num_trees).

Noise rule (step 0). The default configuration with the noise seeds at its
selected T: if its seed range is below `max_seed_range`, tuning uses the
selection seed only; otherwise every tuning candidate is run with all noise
seeds and compared on the seed means above. There is no stop.

Tuning, one factor at a time: hmrec.hpo.selection.choose_winner, the rule of
proposal §4.2 (at least 1% relative gain of the pooled score AND strictly better
in at least 3 of 4 folds; among passing candidates within 1% of the best, the
one with the highest priority).

This module imports neither torch nor LightGBM.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from hmrec.config import DeepFMConfig, DeepFMStudyConfig, DeepFMStudyStep, SelectionRule
from hmrec.deepfm.records import deepfm_config_id
from hmrec.hpo.selection import (
    StepDecision,
    TrialScores,
    choose_num_trees,
    choose_winner,
    relative_change,
)

LEARNING_RATE = "learning_rate"
EMBEDDING_DIM = "embedding_dim"
DROPOUT = "dropout"
LISTWISE = "listwise_softmax"
WITHIN_SEED_NOISE = "within seed noise"


@dataclass(frozen=True)
class Variant:
    """The settings of a configuration that the study changes.

    Attributes:
        learning_rate: Adam learning rate (of the first step of the schedule).
        embedding_dim: Length of every embedding.
        dropout: Dropout after every hidden layer.
        components: "first_order", "fm", "deep" or "full".
        numeric_input: "binned" or "standardised".
        loss: "pointwise_bce" or "listwise_softmax".
        epochs: T, the epochs of a run = the length of its schedule.
    """

    learning_rate: float
    embedding_dim: int
    dropout: float
    components: str
    numeric_input: str
    loss: str
    epochs: int


def variant_of(config: DeepFMConfig) -> Variant:
    """The seven study settings of a configuration."""
    return Variant(
        learning_rate=config.training.learning_rate,
        embedding_dim=config.model.embedding_dim,
        dropout=config.model.dropout,
        components=config.model.components,
        numeric_input=config.model.numeric_input,
        loss=config.training.loss,
        epochs=config.training.epochs,
    )


def apply_variant(config: DeepFMConfig, variant: Variant) -> DeepFMConfig:
    """`config` with the seven study settings replaced by those of `variant`."""
    model = replace(
        config.model,
        embedding_dim=variant.embedding_dim,
        dropout=variant.dropout,
        components=variant.components,
        numeric_input=variant.numeric_input,
    )
    training = replace(
        config.training,
        learning_rate=variant.learning_rate,
        loss=variant.loss,
        epochs=variant.epochs,
    )
    return replace(config, model=model, training=training)


def with_parameter(variant: Variant, parameter: str, value: float) -> Variant:
    """A variant with one tuned parameter set to `value` (the dimension as an integer).

    Raises:
        ValueError: If `parameter` is not one of the three tuned parameters.
    """
    if parameter == LEARNING_RATE:
        return replace(variant, learning_rate=float(value))
    if parameter == EMBEDDING_DIM:
        if float(value) != int(value):
            raise ValueError(f"embedding_dim must be a whole number, got {value}")
        return replace(variant, embedding_dim=int(value))
    if parameter == DROPOUT:
        return replace(variant, dropout=float(value))
    raise ValueError(f"{parameter!r} is not a tuned parameter")


def parameter_value(variant: Variant, parameter: str) -> float:
    """The value of one tuned parameter of a variant."""
    values = {
        LEARNING_RATE: variant.learning_rate,
        EMBEDDING_DIM: float(variant.embedding_dim),
        DROPOUT: variant.dropout,
    }
    if parameter not in values:
        raise ValueError(f"{parameter!r} is not a tuned parameter")
    return values[parameter]


def variant_config_id(config: DeepFMConfig, variant: Variant) -> str:
    """The configuration id of `config` with `variant` applied."""
    return deepfm_config_id(apply_variant(config, variant))


# ---------------------------------------------------------------- scores from metric rows


@dataclass(frozen=True)
class SeedScore:
    """A configuration's recorded result with one seed.

    Attributes:
        seed: The training seed.
        pooled: Query-weighted mean of the folds' values.
        fold_scores: Fold -> the final model's value.
        train_loss: Mean training loss of the last epoch, averaged over the folds.
    """

    seed: int
    pooled: float
    fold_scores: dict[str, float]
    train_loss: float


def _mean(values: Sequence[float]) -> float:
    """Arithmetic mean with math.fsum."""
    return math.fsum(values) / len(values)


@dataclass(frozen=True)
class ConfigurationScore:
    """A configuration's result with one or more seeds (module docstring).

    Attributes:
        config_id: The configuration (its number of epochs is part of the id).
        epochs: T of the configuration.
        seeds: One SeedScore per seed, in the order asked for.
    """

    config_id: str
    epochs: int
    seeds: tuple[SeedScore, ...]

    @property
    def pooled(self) -> float:
        """Mean over the seeds of their pooled values."""
        return _mean([seed.pooled for seed in self.seeds])

    @property
    def fold_scores(self) -> dict[str, float]:
        """Fold -> mean over the seeds of the fold's value."""
        folds = list(self.seeds[0].fold_scores)
        return {fold: _mean([seed.fold_scores[fold] for seed in self.seeds]) for fold in folds}

    @property
    def seed_range(self) -> float:
        """(max - min) / min of the seeds' pooled values.

        Raises:
            ValueError: With fewer than two seeds or a non-positive minimum.
        """
        pooled = [seed.pooled for seed in self.seeds]
        if len(pooled) < 2 or min(pooled) <= 0:
            raise ValueError("the range needs at least two positive pooled scores")
        return (max(pooled) - min(pooled)) / min(pooled)

    def trial_scores(self) -> TrialScores:
        """The same result in the shape the selection rule takes."""
        return TrialScores(self.config_id, self.fold_scores, self.pooled)

    def only_seed(self, seed: int) -> "ConfigurationScore":
        """The same configuration with one of its seeds only."""
        kept = tuple(score for score in self.seeds if score.seed == seed)
        if len(kept) != 1:
            raise ValueError(f"{self.config_id}: seed {seed} is not among its seeds")
        return replace(self, seeds=kept)


def pooled_value(values: Mapping[str, float], n_queries: Mapping[str, int]) -> float:
    """Query-weighted mean over the folds: sum_f n_f * v_f / sum_f n_f (math.fsum)."""
    total = sum(n_queries[fold] for fold in values)
    if total <= 0:
        raise ValueError("the folds have no queries")
    return math.fsum(n_queries[fold] * value for fold, value in values.items()) / total


def score_seed(
    rows: Sequence[Mapping[str, str]],
    area: str,
    config_id: str,
    epochs: int,
    seed: int,
    folds: Sequence[str],
    metric: str,
) -> SeedScore:
    """The recorded result of one configuration with one seed.

    Args:
        rows: Rows of results/deepfm_metrics.csv (any configurations).
        area: "v_phase".
        config_id: The configuration.
        epochs: Its number of epochs: the rows must be of this (last) epoch.
        seed: The seed.
        folds: The folds that must all be there (V1 ... V4), each exactly once.
        metric: The selection metric column, "map@12".

    Raises:
        ValueError: If a fold is missing or recorded twice, or a row is of
            another epoch than the run's last one.
    """
    own = [
        row
        for row in rows
        if row["area"] == area and row["config_id"] == config_id and int(row["seed"]) == seed
    ]
    values: dict[str, float] = {}
    losses: dict[str, float] = {}
    n_queries: dict[str, int] = {}
    for row in own:
        fold = row["fold"]
        if int(row["epoch"]) != epochs:
            raise ValueError(
                f"{config_id} seed {seed} {fold}: a row of epoch {row['epoch']} is recorded, "
                f"but the configuration has {epochs} epochs"
            )
        if fold in values:
            raise ValueError(f"{config_id} seed {seed}: fold {fold} is recorded twice")
        values[fold] = float(row[metric])
        losses[fold] = float(row["train_loss"])
        n_queries[fold] = int(row["n_queries"])
    if sorted(values) != sorted(folds):
        raise ValueError(
            f"{config_id} seed {seed} ({area}): recorded folds {sorted(values)}, "
            f"expected {sorted(folds)}"
        )
    ordered = {fold: values[fold] for fold in folds}
    return SeedScore(
        seed=seed,
        pooled=pooled_value(ordered, n_queries),
        fold_scores=ordered,
        train_loss=_mean([losses[fold] for fold in folds]),
    )


def score_configuration(
    rows: Sequence[Mapping[str, str]],
    area: str,
    config_id: str,
    epochs: int,
    seeds: Sequence[int],
    folds: Sequence[str],
    metric: str,
) -> ConfigurationScore:
    """The recorded result of one configuration with the given seeds (see score_seed)."""
    if not seeds:
        raise ValueError("at least one seed is needed")
    scores = tuple(score_seed(rows, area, config_id, epochs, seed, folds, metric) for seed in seeds)
    return ConfigurationScore(config_id, epochs, scores)


# ---------------------------------------------------------------- the schedule length


@dataclass(frozen=True)
class ScheduleChoice:
    """The epoch rule applied to one set of settings run with every schedule length.

    Attributes:
        scores: T -> the configuration's result with T epochs.
        selected_epochs: T*, the smallest T within the tolerance of the best.
        best_epochs: The smallest T with the highest pooled score.
        on_grid_edge: True if T* is the smallest or the largest T of the grid.
    """

    scores: dict[int, ConfigurationScore]
    selected_epochs: int
    best_epochs: int
    on_grid_edge: bool

    @property
    def selected(self) -> ConfigurationScore:
        """The result at T*."""
        return self.scores[self.selected_epochs]


def choose_schedule(scores: Mapping[int, ConfigurationScore], tolerance: float) -> ScheduleChoice:
    """Apply the epoch rule to the results of one set of settings (T -> result).

    The curve is T -> pooled score (the seed mean if there are several seeds).
    """
    curve = {epochs: score.pooled for epochs, score in scores.items()}
    choice = choose_num_trees(curve, tolerance)
    return ScheduleChoice(
        scores=dict(sorted(scores.items())),
        selected_epochs=choice.num_trees,
        best_epochs=choice.best_num_trees,
        on_grid_edge=choice.num_trees in (min(curve), max(curve)),
    )


# ---------------------------------------------------------------- step 0


class StudyStop(Exception):
    """A stop condition of the study is met: stop and report, do not work around it."""


def check_above_baseline(score: ConfigurationScore, baseline: float) -> None:
    """Stop if the default configuration's (seed-mean) pooled score is below the RRF baseline.

    Raises:
        StudyStop: If it is.
    """
    if score.pooled < baseline:
        raise StudyStop(
            f"the default configuration's seed-mean pooled score {score.pooled:.6f} is below "
            f"the RRF baseline {baseline:.6f}. The study stops."
        )


def tuning_seeds(study: DeepFMStudyConfig, seed_range: float) -> tuple[int, ...]:
    """The seeds every tuning candidate is run with, by the noise rule.

    Below `max_seed_range`: the selection seed only. Otherwise all noise seeds
    (candidates are then compared on seed means).
    """
    if seed_range < study.max_seed_range:
        return (study.selection_seed,)
    return study.noise_seeds


# ---------------------------------------------------------------- tuning


@dataclass(frozen=True)
class StepOutcome:
    """What one tuning step decided.

    Attributes:
        step: The step (parameter and candidate values in priority order).
        start: The configuration the step started from.
        variants: Candidate value -> the configuration with that value, at the T
            it is compared at.
        scores: Candidate value -> its result at that T (seeds of the selection).
        schedules: Learning-rate step only: candidate value -> its epoch rule
            over every T. Empty for the other steps (they use the T of `start`).
        decision: The outcome of the selection rule.
        selected_value: The value that goes on to the next step.
        on_grid_edge: True if the selected value is the smallest or largest of the grid.
    """

    step: DeepFMStudyStep
    start: Variant
    variants: dict[float, Variant]
    scores: dict[float, ConfigurationScore]
    schedules: dict[float, ScheduleChoice]
    decision: StepDecision
    selected_value: float
    on_grid_edge: bool

    @property
    def selected(self) -> Variant:
        """The configuration that goes on to the next step (with its T)."""
        return self.variants[self.selected_value]


def step_variants(step: DeepFMStudyStep, current: Variant) -> dict[float, Variant]:
    """The step's configurations: `current` with the tuned parameter set to each candidate.

    Raises:
        ValueError: If the step's first candidate is not the current value (the
            step would not start from the configuration the study has reached).
    """
    if parameter_value(current, step.parameter) != step.candidates[0]:
        raise ValueError(
            f"step {step.name}: its first candidate {step.candidates[0]} is not the current "
            f"value {parameter_value(current, step.parameter)}"
        )
    return {value: with_parameter(current, step.parameter, value) for value in step.candidates}


def decide_step(
    step: DeepFMStudyStep,
    current: Variant,
    variants: Mapping[float, Variant],
    scores: Mapping[float, ConfigurationScore],
    rule: SelectionRule,
    schedules: Mapping[float, ScheduleChoice] | None = None,
) -> StepOutcome:
    """Apply the selection rule to one tuning step.

    The first candidate is the current configuration (the default of the step);
    the others are compared with it, in priority order.

    Args:
        step: The step.
        current: The configuration the step starts from.
        variants: Candidate value -> its configuration at the T it is compared at.
        scores: Candidate value -> its result; every candidate must be there.
        rule: configs/selection_rule.yaml.
        schedules: The candidates' epoch rules (learning-rate step), or None.
    """
    default_value, *other_values = step.candidates
    default = scores[default_value].trial_scores()
    candidates = [scores[value].trial_scores() for value in other_values]
    decision = choose_winner(default, candidates, rule)
    by_trial = {scores[value].config_id: value for value in step.candidates}
    if len(by_trial) != len(step.candidates):
        raise ValueError(f"step {step.name}: two candidates have the same configuration id")
    selected_value = by_trial[decision.winner_trial_id]
    return StepOutcome(
        step=step,
        start=current,
        variants=dict(variants),
        scores=dict(scores),
        schedules=dict(schedules or {}),
        decision=decision,
        selected_value=selected_value,
        on_grid_edge=selected_value in (min(step.candidates), max(step.candidates)),
    )


# ---------------------------------------------------------------- the listwise arm and ablations


def listwise_step(study: DeepFMStudyConfig) -> DeepFMStudyStep:
    """The learning-rate step of the listwise arm (its candidates, the current value first)."""
    return DeepFMStudyStep("listwise_learning_rate", LEARNING_RATE, study.listwise_learning_rates)


def listwise_start(study: DeepFMStudyConfig, pointwise: Variant) -> Variant:
    """Where the listwise arm starts: the pointwise selection with the listwise loss.

    The embedding dimension and dropout are those selected for the pointwise arm;
    the learning rate is the first value of the listwise grid.
    """
    return replace(pointwise, loss=LISTWISE, learning_rate=study.listwise_learning_rates[0])


def ablation_variants(study: DeepFMStudyConfig, selected: Variant) -> dict[str, Variant]:
    """The ablations of the selected configuration: name -> configuration (same T).

    One setting is changed at a time; the variant equal to the selected
    configuration itself (components "full") is left out.
    """
    variants: dict[str, Variant] = {}
    for components in study.ablation_components:
        if components != selected.components:
            variants[f"components={components}"] = replace(selected, components=components)
    for numeric_input in study.ablation_numeric_inputs:
        if numeric_input != selected.numeric_input:
            variants[f"numeric_input={numeric_input}"] = replace(
                selected, numeric_input=numeric_input
            )
    return variants


def noise_mark(reference: float, value: float, seed_range: float) -> str:
    """The mark "within seed noise" if |relative change| is below the seed range, else "".

    Args:
        reference: The pooled score the value is compared with.
        value: The pooled score of the variant.
        seed_range: The default configuration's 3-seed range, (max - min) / min.
    """
    return WITHIN_SEED_NOISE if abs(relative_change(reference, value)) < seed_range else ""
