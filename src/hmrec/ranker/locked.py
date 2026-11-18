"""The ONLY entry to the Ranker configuration after the lock (proposal §9.6, §11).

After M7 the final evaluation (M8) gets everything about the Ranker from here:

    locked = load_locked_ranker(paths, base, read_lightgbm_runtime())
    params = training_params(locked.config, locked.tuned, locked.histogram_mode,
                             locked.num_threads)
    # train locked.num_trees trees on the locked.config.window_weeks weeks before
    # the predicted week, sample with locked.config.sampling.seed and
    # locked.tuned.negatives_per_positive, score with staged_raw_scores and
    # locked.config.checkpoint_every.

Code that runs after the lock must NOT read the tuned defaults or the fixed
parameters from configs/ranker.yaml, must NOT read configs/hpo/v_phase.yaml and
must NOT replay the selection on results/v_phase*.csv to get a configuration.
tests/unit/test_final_code_rules.py scans the M8 code for those names.

What comes from where:

    configs/locks/ranker.lock.yaml   every value that changes a result: the tuned
                                     values, the number of trees, all LightGBM
                                     parameters, the histogram mode, the seeds, the
                                     sampling ratio, the row groups of a sample file,
                                     the training window, the scoring segment length,
                                     Top-K
    configs/ranker.yaml              ONLY values that change memory or logging:
                                     scoring_query_chunk_size, train_metric_every.
                                     (The histogram rule's numbers and the speed-test
                                     section are copied too, because RankerConfig has
                                     those fields; nothing after the lock uses them.)

load_locked_ranker refuses to return a configuration unless check_ranker_lock
passes:

    1. the settings in the lock give the lock's own config_id (so the model
       settings were not edited after the id was recorded);
    2. the referenced lock files (features, channels, twotower) have the recorded
       sha256;
    3. this process runs the recorded LightGBM and OpenMP versions with the
       recorded number of threads. Another version or thread count can change
       the trees or the order of floating-point sums, so results would no longer
       be comparable with the V phase.

That the lock equals what the recorded V phase decided is checked by
hmrec.hpo.v_lock against results/v_phase*.csv (tests/unit/test_v_lock.py), not
here: code after the lock does not read those tables.
"""

from dataclasses import dataclass, replace
from pathlib import Path

from hmrec.config import (
    BaseConfig,
    RankerConfig,
    RankerSamplingConfig,
    RankerTunedParams,
    load_ranker_config,
)
from hmrec.locks import (
    CHANNELS_LOCK_REFERENCE,
    FEATURES_LOCK_REFERENCE,
    TWOTOWER_LOCK_REFERENCE,
    RankerLock,
    check_file_sha256,
    load_ranker_lock,
)
from hmrec.paths import ProjectPaths
from hmrec.ranker.ablation import FULL
from hmrec.ranker.identity import model_config, ranker_config_id
from hmrec.ranker.runtime import LightGBMRuntime


@dataclass(frozen=True)
class LockedRanker:
    """The locked Ranker configuration, ready to train and score with.

    Attributes:
        lock: The parsed lock file.
        config: A RankerConfig whose result-defining values all come from the
            lock (module docstring). config.num_trees is the LOCKED number of
            trees, so a model is trained to exactly that many trees.
        tuned: num_leaves, min_data_in_leaf, negatives_per_positive.
        num_trees: The locked number of trees (equal to config.num_trees).
        histogram_mode: "col_wise" or "row_wise".
        num_threads: CPU threads of training and scoring.
        config_id: Id of the configuration (equal to lock.selection.config_id).
    """

    lock: RankerLock
    config: RankerConfig
    tuned: RankerTunedParams
    num_trees: int
    histogram_mode: str
    num_threads: int
    config_id: str


def locked_tuned_params(lock: RankerLock) -> RankerTunedParams:
    """The three tuned values of the lock."""
    return RankerTunedParams(
        num_leaves=lock.model.num_leaves,
        min_data_in_leaf=lock.model.min_data_in_leaf,
        negatives_per_positive=lock.sampling.negatives_per_positive,
    )


def locked_ranker_config(lock: RankerLock, memory_settings: RankerConfig) -> RankerConfig:
    """Build the RankerConfig of the locked model.

    Args:
        lock: The parsed lock file.
        memory_settings: configs/ranker.yaml. Only scoring_query_chunk_size and
            train_metric_every are used as they are (memory and logging); the
            histogram rule's numbers and the speed-test section are copied but
            mean nothing after the lock. Every other field is replaced.

    Returns:
        The config: num_trees is the locked number of trees, sampling.ratios
        holds only the locked ratio, tuned_defaults are the locked tuned values.
    """
    tuned = locked_tuned_params(lock)
    return replace(
        memory_settings,
        window_weeks=lock.window_weeks,
        top_k=lock.scoring.top_k,
        num_trees=lock.model.num_trees,
        checkpoint_every=lock.scoring.checkpoint_every,
        seed=lock.model.seed,
        sampling=RankerSamplingConfig(
            seed=lock.sampling.seed,
            ratios=(tuned.negatives_per_positive,),
            row_group_rows=lock.sampling.row_group_rows,
        ),
        tuned_defaults=tuned,
        histogram=replace(memory_settings.histogram, mode=lock.model.histogram_mode),
        dataset_params=dict(lock.model.dataset_params),
        train_params=dict(lock.model.train_params),
    )


def locked_config_id(lock: RankerLock, config: RankerConfig) -> str:
    """The id the lock's own settings give (hmrec.ranker.identity), experiment "full"."""
    settings = model_config(config, locked_tuned_params(lock), FULL, lock.model.histogram_mode)
    return ranker_config_id(settings)


def check_lock_config_id(lock: RankerLock, config: RankerConfig) -> None:
    """Fail unless the lock's settings give the config_id the lock records.

    The id is a hash of the tuned values, the window, the seeds, the histogram
    mode and all LightGBM parameters. If one of them is edited in the lock file,
    the hash changes and no longer equals the recorded id.

    Raises:
        ValueError: If the two ids differ.
    """
    actual = locked_config_id(lock, config)
    if actual != lock.selection.config_id:
        raise ValueError(
            f"configs/locks/ranker.lock.yaml: its settings give config_id {actual}, but it "
            f"records {lock.selection.config_id}. The locked settings were changed."
        )


def _referenced_files(paths: ProjectPaths) -> dict[str, Path]:
    """Reference name -> the lock file it must point to (from hmrec.paths)."""
    return {
        FEATURES_LOCK_REFERENCE: paths.features_lock_file,
        CHANNELS_LOCK_REFERENCE: paths.channels_lock_file,
        TWOTOWER_LOCK_REFERENCE: paths.twotower_lock_file,
    }


def check_referenced_locks(lock: RankerLock, paths: ProjectPaths) -> None:
    """Fail unless the features, channels and twotower lock files are the recorded ones.

    The Ranker was tuned on candidates and features built from those three
    files (R1-R5 from the channels lock; the R6 candidates and the four tt_
    features from the twotower lock; the feature list from the features lock),
    so it is only valid together with exactly those files.

    Raises:
        ValueError: If a recorded path is not the file hmrec.paths names, or a
            file's sha256 differs from the recorded one.
        FileNotFoundError: If a referenced file does not exist.
    """
    for name, file in _referenced_files(paths).items():
        reference = lock.references[name]
        expected_path = file.relative_to(paths.project_root).as_posix()
        if reference.path != expected_path:
            raise ValueError(
                f"ranker.lock references.{name}.path is {reference.path!r}, "
                f"expected {expected_path!r}"
            )
        check_file_sha256(file, reference.sha256)


def check_runtime_matches_lock(
    lock: RankerLock, runtime: LightGBMRuntime, num_threads: int
) -> None:
    """Fail unless this process runs the locked LightGBM / OpenMP versions and thread count.

    Args:
        lock: The parsed lock file.
        runtime: hmrec.ranker.runtime.read_lightgbm_runtime() of THIS process.
        num_threads: num_threads of configs/base.yaml.

    Raises:
        ValueError: If any of the three differs; the message names each one.
    """
    locked = lock.runtime
    pairs = (
        ("LightGBM version", runtime.lightgbm_version, locked.lightgbm_version),
        ("libomp version", runtime.libomp_version, locked.libomp_version),
        ("num_threads", num_threads, locked.num_threads),
    )
    differences = [
        f"{name}: {actual!r} here, {recorded!r} in the lock"
        for name, actual, recorded in pairs
        if actual != recorded
    ]
    if differences:
        raise ValueError(
            "this process does not match configs/locks/ranker.lock.yaml (runtime): "
            + "; ".join(differences)
            + ". With another version or thread count the model can differ from the one "
            "the V phase selected (libomp at "
            f"{runtime.libomp_path})."
        )


def check_ranker_lock(
    lock: RankerLock,
    config: RankerConfig,
    paths: ProjectPaths,
    base: BaseConfig,
    runtime: LightGBMRuntime,
) -> None:
    """Run the three checks of the module docstring; any failure raises."""
    check_lock_config_id(lock, config)
    check_referenced_locks(lock, paths)
    check_runtime_matches_lock(lock, runtime, base.num_threads)


def load_locked_ranker(
    paths: ProjectPaths, base: BaseConfig, runtime: LightGBMRuntime
) -> LockedRanker:
    """Read the Ranker lock, check it and build the configuration (the only entry).

    Args:
        paths: Project paths.
        base: configs/base.yaml (its num_threads must equal the lock's).
        runtime: hmrec.ranker.runtime.read_lightgbm_runtime() of this process.

    Raises:
        ValueError: If the lock file is malformed or any check of
            check_ranker_lock fails.
        FileNotFoundError: If the lock file or a referenced lock file is missing.
    """
    lock = load_ranker_lock(paths.ranker_lock_file)
    config = locked_ranker_config(lock, load_ranker_config(paths.ranker_config_file))
    check_ranker_lock(lock, config, paths, base, runtime)
    return LockedRanker(
        lock=lock,
        config=config,
        tuned=locked_tuned_params(lock),
        num_trees=lock.model.num_trees,
        histogram_mode=lock.model.histogram_mode,
        num_threads=lock.runtime.num_threads,
        config_id=lock.selection.config_id,
    )
