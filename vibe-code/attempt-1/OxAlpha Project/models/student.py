"""SPEC: S7

OxAlpha Reader = WindowEncoder + TokenEmbed + StudyEncoder + LabelQueryHead. No text inputs (T-LEAK-1). Also the Stage A head when fed cached features.

API (planned): Reader(cfg, priors).forward(batch) -> dict(logits[B,12], attn)
"""

raise NotImplementedError("stub -- see specs.txt S7")
