"""SPEC: S6.2

vLLM offline batch labeling with JSON-schema constrained output: per label status {present, absent, uncertain, not_mentioned}, p in [0,1], evidence quote. Writes jsonl keyed by StudyInstanceUID + prompt version + model.

API (planned): run(reports_df, cfg) -> path to jsonl
"""

raise NotImplementedError("stub -- see specs.txt S6.2")
