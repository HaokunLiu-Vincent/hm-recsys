"""Article catalog C and active pool A(a) (proposal §2.4).

Inputs: the articles table and as-of transactions (from data/asof.py).
Outputs:
    - catalog C: every article in articles.csv, one row per article_idx.
    - active pool A(a): articles with at least 3 purchase lines in
      [a - 28d, a), one row per article_idx.
The thresholds are read from configs, not hard-coded.

Not implemented yet (milestone M0).
"""
