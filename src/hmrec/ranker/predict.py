"""Chunked scoring of all candidates and Top-12 / Top-20 output (proposal §1.3, §14.2).

Input: a trained model and artifacts/features/{fold}/ (no sampling).
Output: artifacts/predictions/{run_id}/{fold}.parquet, one row per
(query_id, article_idx) kept, with score and rank.

Not implemented yet (milestone M7).
"""
