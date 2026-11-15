"""Map raw string IDs to dense int32 indexes and back (proposal §2.1).

Inputs: customer_id (64-char hex string), article_id (10-digit string with
leading zeros).
Outputs: two mapping tables, one row per entity:
    - customer_id (str) <-> customer_idx (int32)
    - article_id (str) <-> article_idx (int32)
The article mapping also covers the row order of the precomputed image/text
embedding files (embedding_row -> article_idx).

Not implemented yet (milestone M0).
"""
