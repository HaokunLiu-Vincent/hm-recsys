"""Union + dedup of the 6 channel outputs (proposal §7.3).

Input: artifacts/candidates/{fold}/{channel}.parquet for each channel.
Output: one row per (query_id, article_idx), with each channel's rank and score
as columns (NaN when that channel did not return the article), plus
n_channels and best_rank. Processed in chunks of queries.

Not implemented yet (milestone M4/M6).
"""
