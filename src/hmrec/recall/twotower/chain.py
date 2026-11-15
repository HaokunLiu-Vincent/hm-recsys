"""The single Two-Tower training chain: predict first, then fine-tune (proposal §6).

For each fold in time order: the current model predicts the fold (candidates
and Recall@100), then it is fine-tuned on that fold's purchases to give the
next model. The predicted week is never seen before it is scored.

Not implemented yet (milestone M3).
"""
