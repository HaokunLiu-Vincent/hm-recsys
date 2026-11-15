"""Two-Tower inference: user / item vectors and Top-100 retrieval (proposal §5 R6).

Input: a checkpoint, anchor-date user histories, the active pool A(a).
Output: candidates in the recall/base.py schema (channel "tt"), computed with
chunked matrix multiply + torch.topk on "mps".

Not implemented yet (milestone M3).
"""
