"""The ONLY entry point for cutting history by anchor date (AGENTS.md §3.1).

Input: a transactions table (one row per purchase line) and an anchor date.
Output: the rows with t_dat inside a half-open window [anchor - lookback, anchor).
The anchor date itself is never included: it belongs to the label week.

Every function here asserts that the returned data has max(t_dat) < anchor.

Not implemented yet (milestone M0/M1).
"""
