"""Weekly query list and label sets G_q (proposal §2.3, §14.2).

Input: transactions and a Fold.
Outputs:
    - artifacts/queries/{fold}.parquet: one row per query.
      Columns: query_id, customer_idx (int32), anchor.
      A query exists only if the customer bought at least one catalog article
      in the label week [anchor, anchor + 7d).
    - artifacts/labels/{fold}.parquet: one row per (query_id, article_idx),
      de-duplicated.
Labels are read only by eval/ and ranker/sampling.py.

Not implemented yet (milestone M0/M1).
"""
