"""SPEC: REQ-NLP-1, REQ-NLP-2

Language ID, per-language lexicon/negation extractor -> e in {-1,0,1}^12 + confidence; XLM-R-base mean-pooled z_text[768]. TRAIN ONLY.

API (planned): extract_labels(report)->(e,c) ; TextEncoder.encode(reports)->z_text
"""

raise NotImplementedError("stub -- see SPEC.md (REQ-NLP-1, REQ-NLP-2)")
