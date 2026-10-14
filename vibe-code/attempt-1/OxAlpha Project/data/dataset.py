"""SPEC: S3.6, S4

StudyBagDataset: per study -> x[S,24,3,224,224], s_mask[S], meta[S,2], optional labels/weights/report fields (train only).

API (planned): StudyBagDataset(csv, series_csv, root, fold, train) ; collate_studies(batch)
"""

raise NotImplementedError("stub -- see specs.txt S3.6, S4")
