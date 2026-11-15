"""Ranking metrics, macro-averaged over queries (proposal §2.5).

Recall@K, AP@12 / MAP@12 (denominator min(12, |G_q|)), NDCG@K (binary, ideal
DCG from the full G_q), union recall. Short outputs count missing slots as
misses. Queries without output stay in the denominator.

Not implemented yet (milestone M1).
"""
