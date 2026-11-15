"""RecallChannel interface and output schema validation (proposal §5).

Every channel writes one table, one row per (query_id, article_idx):
    query_id, article_idx (int32), channel (str), rank (int32, 1-based,
    consecutive), score (float32).
Rows are sorted by score DESC, article_idx ASC. article_idx is unique within a
query. A channel never pads to its quota.

The schema check here must run before any channel output is written to disk.

Not implemented yet (milestone M1).
"""
