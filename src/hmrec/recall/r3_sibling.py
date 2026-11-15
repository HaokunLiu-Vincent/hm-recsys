"""R3 same product_code, other colour ("sibling", proposal §5 R3).

Input: as-of purchase lines ([a - 180d, a) for style, [a - 7d, a) for market
heat) and articles (product_code).
Score: S(u, j) = P(u, c(j)) * (1 + M(j)) ** gamma. Articles the customer already
bought in the 180-day window are excluded.
Output: candidates in the recall/base.py schema, channel "sib".

Not implemented yet (milestone M1).
"""
