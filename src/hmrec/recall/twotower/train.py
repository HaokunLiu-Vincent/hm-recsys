"""Base training and weekly fine-tuning for the Two-Tower model (proposal §6.2, §6.4).

Base: label dates 2019-12-03 to 2020-06-02, constant LR, one checkpoint per epoch.
Fine-tune: one week of purchases (+ optional replay), LR = 0.1 x base LR.
Outputs: checkpoints under artifacts/models/twotower/.

Not implemented yet (milestone M3).
"""
