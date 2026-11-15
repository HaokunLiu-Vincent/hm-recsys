"""Two-Tower training samples and user history arrays (proposal §6.1).

Input: as-of purchase lines, customers, articles, precomputed image/text vectors.
Outputs:
    - one training sample per purchase line: (history, profile) -> article_idx.
    - user history: int32 array of shape [N, 20], the last 20 purchases before
      the purchase day (padded with a mask value).
    - item feature table: float32 [n_articles, 512 + 384] with a missing mask.

Not implemented yet (milestone M3).
"""
