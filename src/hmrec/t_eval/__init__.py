"""The single evaluation of the DeepFM configuration chosen on V, on T1-T4 (proposal §16.12).

Run once, reported only. plan / gate / rehearsal / run / compare import neither
torch nor LightGBM; trial is on the torch side. LightGBM is not loaded anywhere:
LightGBM lambdarank on T is the M8 record and is only read.
"""
