"""FeatureGroup interface (proposal §8).

Input: an anchor date and a candidate table (one row per (query_id, article_idx)).
Output: the same rows with this group's feature columns added (float32 or
categorical). All windows are half-open and end before the anchor.

Not implemented yet (milestone M6).
"""
