"""The §4 selection rule (proposal §4.1-4.2, configs/selection_rule.yaml).

Input: per-fold and query-weighted scores of the default and a candidate.
Output: whether the candidate replaces the default: relative improvement of the
query-weighted score >= 1% AND better on at least 3 of 4 folds.

Not implemented yet (milestone M2).
"""
