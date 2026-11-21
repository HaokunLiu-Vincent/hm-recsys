"""Tests for the LightGBM-side report of hmrec.platform_report.

The file name starts with "test_ranker_" so that it runs in the LightGBM-side test
process (tests/conftest.py): the report imports lightgbm.
"""

import os
import sys

import lightgbm as lgb

from hmrec.platform_report import report_lines
from hmrec.ranker.runtime import read_lightgbm_runtime


def test_the_lightgbm_side_report_of_this_process() -> None:
    lines = report_lines("lightgbm", os.environ)
    runtime = read_lightgbm_runtime()
    assert lines[0] == "side: lightgbm"
    assert f"package lightgbm: {lgb.__version__}" in lines
    assert f"lightgbm openmp runtime: {runtime.libomp_path}" in lines
    assert f"lightgbm openmp version: {runtime.libomp_version}" in lines
    # LightGBM must be the only library that brings an OpenMP runtime into this process.
    assert "openmp runtimes loaded: 1" in lines
    assert "torch" not in sys.modules
    if sys.platform == "darwin":
        # The laptop: Homebrew's libomp, the runtime every recorded result ran on.
        assert "/Cellar/libomp/" in runtime.libomp_path
        assert any(line.startswith("reference platform: yes") for line in lines)
    else:
        # A Linux container: GNU libgomp of the image, with its package version.
        assert "libgomp.so" in runtime.libomp_path
        assert runtime.libomp_version.startswith("libgomp1 ")
        assert any(line.startswith("reference platform: NO") for line in lines)
        assert any(
            line.startswith("openmp runtime: ") and "(system package: libgomp1 " in line
            for line in lines
        )
