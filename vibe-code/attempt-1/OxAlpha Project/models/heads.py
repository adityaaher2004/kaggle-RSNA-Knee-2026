"""SPEC: S7.4

Label-query head: 12 learned queries, 1 cross-attention decoder layer (optional query self-attn), per-label linear, bias init logit(prior). Returns logits and attention.

API (planned): LabelQueryHead(cfg, priors).forward(tokens, mask) -> (logits[B,12], attn[B,12,S*T])
"""

raise NotImplementedError("stub -- see specs.txt S7.4")
