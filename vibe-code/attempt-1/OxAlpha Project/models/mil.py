"""SPEC: S7.5

Ablation aggregators over window tokens: mean, max, gated-attention MIL (Ilse 2018) with a linear 12-way head.

API (planned): make_pool(kind, cfg) -> nn.Module; forward(tokens, mask) -> logits[B,12]
"""

raise NotImplementedError("stub -- see specs.txt S7.5")
