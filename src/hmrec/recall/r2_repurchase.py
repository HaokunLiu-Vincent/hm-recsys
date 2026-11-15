"""R2 decayed-frequency repurchase (proposal §5 R2).

Input: as-of purchase lines in [a - 180d, a).
Score: S(u, i) = sum of exp(-age_days / tau) over each purchase line of article
i by customer u (duplicate lines count separately). Soft quota, no padding.
Output: candidates in the recall/base.py schema, channel "rep".

Not implemented yet (milestone M1).
"""
