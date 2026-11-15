"""In-query random negative sampling for ranker training (proposal §9.1).

Input: feature rows for one fold and the fold's labels.
Output: all positive rows plus up to r uniformly sampled negatives per positive,
within the same query; only queries with >= 1 positive. Fixed seed.
One of two places allowed to read labels.

Not implemented yet (milestone M7).
"""
