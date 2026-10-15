"""SPEC: S7, R4  (T-LEAK-1)

Reader has no text parameters; inference/ never imports models.teacher/text_encoder/labels; batch keys at inference contain no report fields.
"""
import pytest

pytest.skip("not implemented -- see TODO T-LEAK-1", allow_module_level=True)
