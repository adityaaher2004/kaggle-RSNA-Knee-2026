"""SPEC: S7.1

2.5D window encoder: timm backbone (EffNetV2-S default), GeM pool, Linear(C->d)+LayerNorm, gradient checkpointing.

API (planned): WindowEncoder(cfg).forward(x[N,3,H,W]) -> tokens[N,d]
"""

raise NotImplementedError("stub -- see specs.txt S7.1")
