"""SPEC: S4  (T-SHAPE-1)

Dummy CPU batch through Reader gives logits B x 12 and attn B x 12 x S*T; padded windows get zero attention.
"""
import pytest

pytest.skip("not implemented -- see TODO T-SHAPE-1", allow_module_level=True)
