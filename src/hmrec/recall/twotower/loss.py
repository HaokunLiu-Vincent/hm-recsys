"""Multi-positive in-batch softmax loss with LogQ correction (proposal §5 R6).

Input: user vectors [B, 128], item vectors [B, 128], item sampling log-probs [B],
and a positive mask [B, B] (same customer, same day = all positives).
Output: scalar float32 loss.

Not implemented yet (milestone M3).
"""
