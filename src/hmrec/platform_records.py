"""Platform records: which platforms are accepted, and by which thresholds (docs/mlops.md §8.3).

A result computed in a container is not bit-identical to the laptop. It is
accepted against the stage-1 record by a relative difference of at most 1.0% on
four quantities, and by two structural checks (the user's decision of
2026-11-22). A platform record, configs/platforms/{platform}.yaml, states that
for ONE platform and describes the runtime the statement was made for:

    platform, os, architecture     e.g. linux-arm64, linux, aarch64
    base_image                     the base image of the Dockerfile, with its digest
    ranker_runtime                 LightGBM version, OpenMP runtime, num_threads
    recall_runtime                 torch version (it identifies the OpenMP runtime
                                   of the recall processes) and library versions
    settings                       thread and determinism settings in force
    acceptance                     the threshold, the quantities, the structural checks
    calibration                    the measurement the acceptance rests on
    ranker_lock                    the Ranker lock file the record was made for

**Platform records are append-only, like lock files** (AGENTS.md §3.3): a file
is never modified or deleted. A changed runtime gets a new record.

This module only reads and checks a record. The tolerance-accepted entry to the
Ranker (hmrec.ranker.platform_entry, mode "accepted") decides with it whether a
process may run. It is at the bottom layer: it depends on hmrec.config and the
standard library, and loads neither torch nor LightGBM.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hmrec.config import (
    check_keys,
    get_bool,
    get_float,
    get_int,
    get_mapping,
    get_str,
    get_str_list,
    load_yaml,
)

TOP_LEVEL_KEYS = {
    "platform",
    "os",
    "architecture",
    "recorded_on",
    "decision_record",
    "base_image",
    "ranker_runtime",
    "recall_runtime",
    "settings",
    "acceptance",
    "calibration",
    "ranker_lock",
}
RANKER_RUNTIME_KEYS = {"lightgbm_version", "openmp_version", "num_threads"}
ACCEPTANCE_KEYS = {
    "max_relative_difference",
    "quantities",
    "structural_checks",
    "reported_only",
    "bit_identical_repetition_required",
}
RANKER_LOCK_KEYS = {"path", "sha256"}


@dataclass(frozen=True)
class PlatformAcceptance:
    """How a result of the platform is accepted against the stage-1 record.

    Attributes:
        max_relative_difference: |result - record| / record must be at most
            this for every quantity (0.01).
        quantities: The quantities the threshold applies to.
        structural_checks: Checks that detect bugs, not numeric differences.
        reported_only: Quantities that are reported and never decide acceptance.
        bit_identical_repetition_required: Whether two runs on the platform
            must give the same bits (False).
    """

    max_relative_difference: float
    quantities: tuple[str, ...]
    structural_checks: tuple[str, ...]
    reported_only: tuple[str, ...]
    bit_identical_repetition_required: bool


@dataclass(frozen=True)
class PlatformRecord:
    """One platform record (the fields the code uses; the file holds more, for the reader).

    Attributes:
        platform: Name of the platform and of the file, e.g. "linux-arm64".
        os: sys.platform of an accepted process, e.g. "linux".
        architecture: platform.machine() of an accepted process, e.g. "aarch64".
        decision_record: The docs/decisions.md entry that accepted the platform.
        base_image: The base image of the Dockerfile, with its digest.
        lightgbm_version: LightGBM version of the Ranker side.
        openmp_version: OpenMP runtime LightGBM runs on, as
            hmrec.ranker.runtime records it, e.g. "libgomp1 14.2.0-19".
        num_threads: Thread count of the Ranker side.
        torch_version: torch version of the recall side.
        r6_fine_tune_setting: Runtime setting of an R6 fine-tune, e.g. "deterministic".
        acceptance: The thresholds and checks.
        ranker_lock_path: The Ranker lock file the record was made for, relative
            to the repository root.
        ranker_lock_sha256: Its sha256.
    """

    platform: str
    os: str
    architecture: str
    decision_record: str
    base_image: str
    lightgbm_version: str
    openmp_version: str
    num_threads: int
    torch_version: str
    r6_fine_tune_setting: str
    acceptance: PlatformAcceptance
    ranker_lock_path: str
    ranker_lock_sha256: str


def parse_acceptance(raw: Mapping[str, Any], where: str) -> PlatformAcceptance:
    """Build the acceptance part of a record.

    Raises:
        ValueError: If a key is missing or unknown, the threshold is not in
            (0, 1), or a list is empty.
    """
    check_keys(raw, ACCEPTANCE_KEYS, where)
    acceptance = PlatformAcceptance(
        max_relative_difference=get_float(raw, "max_relative_difference", where),
        quantities=get_str_list(raw, "quantities", where),
        structural_checks=get_str_list(raw, "structural_checks", where),
        reported_only=get_str_list(raw, "reported_only", where),
        bit_identical_repetition_required=get_bool(raw, "bit_identical_repetition_required", where),
    )
    if not 0 < acceptance.max_relative_difference < 1:
        raise ValueError(
            f"{where}.max_relative_difference must be in (0, 1), "
            f"got {acceptance.max_relative_difference}"
        )
    if not acceptance.quantities or not acceptance.structural_checks:
        raise ValueError(f"{where} must name at least one quantity and one structural check")
    return acceptance


def parse_platform_record(raw: Mapping[str, Any], where: str) -> PlatformRecord:
    """Build a PlatformRecord from the parsed content of a record file.

    Args:
        raw: The parsed YAML mapping.
        where: Name of the file for error messages.

    Raises:
        ValueError: If a key is missing or unknown, or a value is malformed.
    """
    check_keys(raw, TOP_LEVEL_KEYS, where)
    ranker = get_mapping(raw, "ranker_runtime", where)
    check_keys(ranker, RANKER_RUNTIME_KEYS, f"{where}.ranker_runtime")
    recall = get_mapping(raw, "recall_runtime", where)
    settings = get_mapping(raw, "settings", where)
    lock = get_mapping(raw, "ranker_lock", where)
    check_keys(lock, RANKER_LOCK_KEYS, f"{where}.ranker_lock")
    record = PlatformRecord(
        platform=get_str(raw, "platform", where),
        os=get_str(raw, "os", where),
        architecture=get_str(raw, "architecture", where),
        decision_record=get_str(raw, "decision_record", where),
        base_image=get_str(raw, "base_image", where),
        lightgbm_version=get_str(ranker, "lightgbm_version", f"{where}.ranker_runtime"),
        openmp_version=get_str(ranker, "openmp_version", f"{where}.ranker_runtime"),
        num_threads=get_int(ranker, "num_threads", f"{where}.ranker_runtime"),
        torch_version=get_str(recall, "torch_version", f"{where}.recall_runtime"),
        r6_fine_tune_setting=get_str(settings, "r6_fine_tune", f"{where}.settings"),
        acceptance=parse_acceptance(get_mapping(raw, "acceptance", where), f"{where}.acceptance"),
        ranker_lock_path=get_str(lock, "path", f"{where}.ranker_lock"),
        ranker_lock_sha256=get_str(lock, "sha256", f"{where}.ranker_lock"),
    )
    if "sha256:" not in record.base_image:
        raise ValueError(f"{where}.base_image must be pinned by digest, got {record.base_image!r}")
    return record


def load_platform_record(path: Path) -> PlatformRecord:
    """Read and check the record of one platform.

    Args:
        path: configs/platforms/{platform}.yaml.

    Raises:
        FileNotFoundError: If the platform has no record.
        ValueError: If the file is malformed, or its `platform` is not the
            name of the file (a record copied to another name is not a record
            of that platform).
    """
    if not path.is_file():
        raise FileNotFoundError(f"platform record not found: {path}")
    record = parse_platform_record(load_yaml(path), path.name)
    if record.platform != path.stem:
        raise ValueError(
            f"{path} records platform {record.platform!r}, expected {path.stem!r} (the file name)"
        )
    return record
