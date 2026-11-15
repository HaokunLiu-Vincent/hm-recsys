"""R5 time-decayed implicit ALS (WALS, proposal §5 R5).

Input: as-of purchase lines in [a - 180d, a), active pool.
Confidence c_ui = 1 + alpha_conf * r_ui, with r_ui = sum of exp(-age_days / 60).
Uses the CPU version of `implicit` with alpha=1.0 and num_threads=6.
Outputs: candidates in the recall/base.py schema (channel "wals"), and the
fitted user / item factor arrays (float32) under artifacts/models/wals/{fold}/.

Not implemented yet (milestone M1).
"""
