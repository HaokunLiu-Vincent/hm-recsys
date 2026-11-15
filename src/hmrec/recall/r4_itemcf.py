"""R4 item-based collaborative filtering (proposal §5 R4).

Input: as-of purchase lines ([a - 365d, a) to fit, [a - 90d, a) for seeds), active pool.
Similarity: s(i -> j) = C_w(i, j) / (n_w(i) ** alpha * n_w(j) ** (1 - alpha)),
with IUF user weight w_u = 1 / ln(1 + d_u). Top-K_n neighbours per seed.
Computed in chunks of seed items to stay inside the memory budget (proposal §1.3).
Output: candidates in the recall/base.py schema, channel "icf".

Not implemented yet (milestone M1).
"""
