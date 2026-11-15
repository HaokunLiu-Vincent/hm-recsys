"""ItemTower and UserTower modules (proposal §5 R6).

ItemTower: frozen CLIP (512) + MiniLM (384) vectors through trainable
projections, plus trainable article_id / product_code / category embeddings.
UserTower: pooled ItemTower encodings of the last 20 purchases + user profile.
Both output 128-dim L2-normalized float32 vectors. Runs on device "mps".

Not implemented yet (milestone M3).
"""
