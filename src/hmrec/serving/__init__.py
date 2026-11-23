"""The recommendation service (docs/mlops.md §9; M14): it looks up precomputed Top-12 lists.

    hmrec.serving.store    the prediction set in memory; follows state/current.json
    hmrec.serving.app      the HTTP API (FastAPI) and one log line per request

The service computes nothing of the system: the weekly pipeline writes a
prediction set and its serving form (hmrec.serving_set), and the service reads
it from a READ-ONLY folder of the state.

Its image holds only this package, hmrec.serving_set and hmrec.structured_logs,
with numpy, FastAPI and uvicorn (Dockerfile.serving). So this package imports
nothing else of the project: no torch, no LightGBM, no polars, no config file.
tests/unit/test_serving_code_rules.py checks this.
"""
