"""SPEC: S3.5, S4

StudyDataset: study -> selected series -> T windows (stratified jitter train / even eval) of 3 slices; returns x[S,T,3,H,W], masks, z_pos, plane, contrast, slot, soft targets q, confidence c. collate pads to s_max.

API (planned): StudyDataset(df_study, df_series, labels_df, cfg, train); collate(batch) -> dict of tensors (shapes in S4)
"""

raise NotImplementedError("stub -- see specs.txt S3.5")
