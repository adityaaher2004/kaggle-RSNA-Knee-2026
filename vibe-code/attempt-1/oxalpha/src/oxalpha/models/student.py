"""SPEC: REQ-MODEL-5, REQ-LEAK-1

Deployed image-only model: trunk + label-embedding head. MUST NOT import text_encoder/fusion.

API (planned): Student.forward(x, s_mask, meta) -> logits[B,12], z_study, attn
"""

raise NotImplementedError("stub -- see SPEC.md (REQ-MODEL-5, REQ-LEAK-1)")
