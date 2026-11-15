"""Convert the raw Kaggle CSV files to compact Parquet files (proposal §1.3, §2.1).

Inputs (data/raw/): transactions_train.csv, customers.csv, articles.csv.
Outputs (data/processed/):
    - transactions: one row per purchase line (exact duplicate rows are kept,
      they mean repeated purchases). Columns: customer_idx (int32),
      article_idx (int32), t_dat (int16 days since the first date),
      price (float32), sales_channel_id (int8).
    - customers: one row per customer, keyed by customer_idx (int32).
    - articles: one row per article, keyed by article_idx (int32).

article_id is read as a 10-character string so leading zeros are kept.

Not implemented yet (milestone M0).
"""
