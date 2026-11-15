"""Streaming LightGBM Dataset construction (proposal §1.3).

Input: sampled feature Parquet files (read by row group).
Output: a LightGBM Dataset built from lgb.Sequence, saved as a binary file
under artifacts/lgb_datasets/{run_id}/.

Not implemented yet (milestone M7).
"""
