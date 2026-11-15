"""Assemble all feature groups in query chunks and write Parquet (proposal §7.3).

Input: the fused candidate table for one fold.
Output: artifacts/features/{fold}/part-*.parquet, one row per
(query_id, article_idx) with all §8 feature columns. Frozen once written.

Not implemented yet (milestone M6).
"""
