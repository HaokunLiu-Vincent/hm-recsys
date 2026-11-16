# Fixture for asof / catalog / queries tests

`transactions.csv` is already in the PROCESSED format (int indexes, `t_dat_day`).
Day 0 = 2020-01-01 (`DAY_ZERO` in `tests/unit/conftest.py`, not the real 2018-09-20).

Test folds:

| fold | anchor | label_end | label days |
|---|---|---|---|
| D1 | 2020-02-12 (day 42) | 2020-02-18 (day 48) | 42..48 |
| D2 | 2020-02-19 (day 49) | 2020-02-25 (day 55) | 49..55 |

Rows (line number in the CSV, header excluded):

| # | customer | article | day | why it is there |
|---|---|---|---|---|
| 1 | 0 | 0 | 14 | first day of the 28-day window before day 42 (included) |
| 2 | 1 | 0 | 30 | |
| 3 | 2 | 0 | 41 | day before the anchor (included) |
| 4 | 3 | 1 | 13 | one day before the 28-day window (excluded) |
| 5 | 1 | 1 | 30 | |
| 6 | 2 | 1 | 41 | |
| 7 | 0 | 2 | 20 | |
| 8 | 0 | 2 | 20 | exact duplicate of row 7 (a repeat purchase, counts twice) |
| 9 | 3 | 2 | 35 | first day of the 7-day window before day 42 |
| 10 | 1 | 3 | 40 | |
| 11 | 3 | 3 | 41 | |
| 12 | 0 | 3 | 42 | D1 anchor day: label week, never history |
| 13 | 0 | 4 | 45 | |
| 14 | 0 | 4 | 48 | same customer + article again in the label week (one label) |
| 15 | 2 | 0 | 48 | last day of the D1 label week |
| 16 | 1 | 1 | 49 | D2 anchor day: D2 label, not D1 |
| 17 | 0 | 2 | 55 | last day of the D2 label week |

Expected values (anchor = D1, day 42):

- History, lookback 7 → days 35..41 → rows 3, 6, 9, 10, 11 (5 rows).
- History, lookback 28 → days 14..41 → rows 1–3 and 5–11 (10 rows; row 4 is too early).
- Active pool (28 days, ≥ 3 lines): article 0 = rows 1, 2, 3 → 3 ✓; article 1 = rows 5, 6 → 2 ✗;
  article 2 = rows 7, 8, 9 → 3 ✓ (the duplicate counts); article 3 = rows 10, 11 → 2 ✗
  (row 12 is on the anchor). Pool = {0: 3, 2: 3}.
- D1 queries: customers 0, 2 → positions 0, 1 → query_id 0, 1 (D1 ordinal 0).
  Labels: (0, 3), (0, 4), (1, 0). Article 4 bought twice → one label.
- D2 queries (ordinal 1, offset O): customers 0, 1 → query_id O + 0, O + 1.
  Labels: (O + 0, 2), (O + 1, 1).
