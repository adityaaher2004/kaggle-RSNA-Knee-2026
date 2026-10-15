"""SPEC: S3

Series -> cache v1 npz (uint8 vol <=48x256x256, z_mm) + json sidecar; per-series percentile window; multiprocessing driver. The Kaggle kernel imports preprocess_series unchanged.

API (planned): preprocess_series(series_dir, cfg) -> (vol uint8, z_mm, meta); build_cache(raw_root, out_root, cfg, workers)
"""

raise NotImplementedError("stub -- see specs.txt S3")
