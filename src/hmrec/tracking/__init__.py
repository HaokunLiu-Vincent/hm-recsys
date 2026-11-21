"""The local MLflow view over run manifests and result tables (docs/mlops.md §7.3).

The manifests of the weekly pipeline and the result tables are the source of
truth. MLflow is only a VIEW of them: `make mlflow-sync` copies records into a
local MLflow store (SQLite, under artifacts/mlflow/, not in git), and
`make mlflow-check` proves that every metric in the store equals the value in
its source. Nothing reads MLflow to decide anything, and the store can be
deleted and rebuilt at any time.

    records.py   which runs and metrics the view holds (no MLflow needed)
    store.py     writing to and reading from the MLflow store
    check.py     every logged metric against its source

MLflow lives in a dependency group of its own ("tracking"), which the pipeline
image does not install; store.py therefore loads it only when it is used.

This package is in the upper layer. It reads files only; it imports neither
torch nor lightgbm, and no other code imports it.
"""
