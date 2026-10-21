"""SPEC: SPEC.md S1. Executable data contract: each assert is a [V] fact from SPEC S1.
Run: python tests/test_data_contract.py   (needs RSNA_DATA). On the HPC full set a failing assert means
the local-sample fact does not hold there -- update SPEC S1 first, then this file, in one commit.
CSV facts must hold on any copy of the tables; DICOM-on-disk facts are checked only for what is present.
"""
import os
import pandas as pd

D = os.environ["RSNA_DATA"]
tr = pd.read_csv(f"{D}/train.csv"); ts = pd.read_csv(f"{D}/train_series.csv")
sub = pd.read_csv(f"{D}/sample_submission.csv")
labs = [c for c in tr.columns if c not in ("StudyInstanceUID", "Report")]

assert len(labs) == 12 and list(sub.columns) == ["StudyInstanceUID"] + labs, "label/submission columns drifted"
assert tr.StudyInstanceUID.is_unique and tr.Report.notna().all() and (tr.Report.str.len() > 0).all()
assert set(tr.StudyInstanceUID) == set(ts.StudyInstanceUID), "train.csv and train_series.csv disagree on studies"
assert ts.SeriesInstanceUID.is_unique
assert set(ts.Anatomical_Plane) == {"Sagittal", "Coronal", "Axial"}
assert (ts.Fluid_Sensitive == ts.Fat_Suppression).all(), "I-07: flags no longer identical -> two features again"
assert (ts.groupby("StudyInstanceUID").Anatomical_Plane.nunique() == 3).all(), "every study has all 3 planes"
# I-01: labels are all-or-nothing per study, and tiny in number
k = tr[labs].notna().sum(axis=1)
assert set(k) <= {0, 12}, "partially labelled studies exist -> weak-label masking logic must change"
print(f"gold studies: {(k == 12).sum()} / {len(tr)}  (SPEC S1 expects 58 / 4407 on the local copy)")
gold = tr[k == 12]
print("all-negative gold studies:", int((gold[labs].sum(axis=1) == 0).sum()), "(0 on local copy: gold set is enriched, I-02)")

# on-disk series must be known to the csv, with the same study
for split, df in (("train_series", ts), ("test_series", pd.read_csv(f"{D}/test_series.csv"))):
    root = f"{D}/series_data/{split}"
    if not os.path.isdir(root): continue
    known = dict(zip(df.SeriesInstanceUID, df.StudyInstanceUID))
    for st in os.listdir(root):
        for se in os.listdir(f"{root}/{st}"):
            assert known.get(se) == st, f"{split}: {st}/{se} not in csv"
print("data contract OK")
