"""R1 age-bucketed decayed popularity (proposal §5 R1).

Input: as-of purchase lines in [a - 7d, a), customers (for age bucket), active pool A(a).
Score: S(g, i) = sum of exp(-age_days / tau) over purchase lines of article i
by customers in age bucket g. Each customer gets the Top-quota of their bucket;
short lists are filled from global decayed popularity (pop_is_fallback = 1).
Output: candidates in the recall/base.py schema, channel "pop".

Not implemented yet (milestone M1).
"""
