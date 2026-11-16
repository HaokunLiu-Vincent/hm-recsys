# preprocess fixtures

Hand-made inputs for `tests/unit/test_preprocess.py`. Small enough to check by hand.

- `articles.csv`: 4 articles, IDs with leading zeros, rows NOT in ID order.
  Sorted IDs give article_idx: 0108775015 -> 0, 0108775044 -> 1, 0110065001 -> 2, 0951111001 -> 3.
  One colour_group_code is -1; one description contains a comma and quotes.
- `customers.csv`: 10 customers (cust_a..cust_j -> customer_idx 0..9), rows NOT in ID order.
  Ages cover every age bucket and both edges of AGE_18_24 / AGE_55_PLUS:
  17 and 100 -> AGE_INVALID, missing -> AGE_UNKNOWN. One "None" fashion_news_frequency.
- `transactions.csv`: 7 lines over 3 days (2018-09-20 = day 0). Line 5 is an exact
  duplicate of line 2 and must be kept. Lines are NOT sorted by (day, customer).
