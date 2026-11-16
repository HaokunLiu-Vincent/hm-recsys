"""The §4 selection rule (proposal §4.1-4.2, configs/selection_rule.yaml).

Input: per-fold and query-weighted scores of the default and a candidate.
Output: whether the candidate replaces the default: relative improvement of the
query-weighted score >= 1% AND better on at least 3 of 4 folds.

Only SCORE_EQUALITY_TOLERANCE is defined so far; the rule itself is not
implemented yet (milestone M2).
"""

# Two scores whose absolute difference is below this value are treated as EQUAL.
# This only removes floating-point noise (metric values are float64, so real
# differences are many orders of magnitude larger). It does NOT relax "strictly
# better" in proposal §4.2: a candidate is better on a fold only if
# candidate - default >= SCORE_EQUALITY_TOLERANCE. configs/selection_rule.yaml is
# committed and must not change, so this constant lives here instead.
# See docs/decisions.md (2026-11-16).
SCORE_EQUALITY_TOLERANCE = 1e-12
