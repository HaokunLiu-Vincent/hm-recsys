"""Reciprocal rank fusion (proposal §8.2, §9.5).

rrf_score = sum over channels that returned the article of 1 / (k + rank), k = 60.
Also the RRF baseline: sort by rrf_score DESC, then best_rank ASC, then
article_idx ASC, and take Top-12.

Not implemented yet (milestone M4/M7).
"""
