"""SPEC: SPEC.md S1 (verified facts). Re-runnable EDA; prints markdown to stdout.

  python scripts/eda_local.py --data <DATA dir> [--dicom-limit N] > DATA_REPORT.md

CSV sections run on the full tables. DICOM section reads headers only (stop_before_pixels) for up to
N series that exist on disk -- locally that is every series, on HPC use e.g. --dicom-limit 300.
"""
import argparse, glob, os, re
import numpy as np, pandas as pd, pydicom

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=os.environ.get("RSNA_DATA"))
ap.add_argument("--dicom-limit", type=int, default=10**9)
args = ap.parse_args(); D = args.data
assert D, "pass --data or set RSNA_DATA"

tr = pd.read_csv(f"{D}/train.csv"); ts = pd.read_csv(f"{D}/train_series.csv")
te = pd.read_csv(f"{D}/test.csv"); tes = pd.read_csv(f"{D}/test_series.csv")
labs = [c for c in tr.columns if c not in ("StudyInstanceUID", "Report")]
gold = tr.dropna(subset=labs)
out = lambda s="": print(s)

out("## Tables")
out(f"- train.csv {tr.shape}, unique studies {tr.StudyInstanceUID.nunique()}; train_series.csv {ts.shape}, "
    f"unique series {ts.SeriesInstanceUID.nunique()}, studies {ts.StudyInstanceUID.nunique()}")
out(f"- train/series study sets equal: {set(tr.StudyInstanceUID) == set(ts.StudyInstanceUID)}")
out(f"- test.csv {te.shape}, test_series.csv {tes.shape} (sample only)")
out(f"- labels: {labs}")
out(f"- studies with all 12 labels: {len(gold)}; with some-but-not-all: {int((tr[labs].notna().any(axis=1)).sum() - len(gold))}; with none: {int(tr[labs].isna().all(axis=1).sum())}")

out("\n## Series")
n = ts.groupby("StudyInstanceUID").size()
out(f"- series/study: mean {n.mean():.2f}, median {n.median():.0f}, min {n.min()}, max {n.max()}; dist {n.value_counts().sort_index().to_dict()}")
out(f"- planes: {ts.Anatomical_Plane.value_counts().to_dict()}")
out(f"- (Fluid_Sensitive, Fat_Suppression) counts: {ts.groupby(['Fluid_Sensitive','Fat_Suppression']).size().to_dict()}")
out(f"- plane x fluid: {ts.groupby(['Anatomical_Plane','Fluid_Sensitive']).size().to_dict()}")
cnt = ts.groupby(["StudyInstanceUID", "Anatomical_Plane"]).size().unstack(fill_value=0)
out(f"- studies having all 3 planes: {int((cnt > 0).all(axis=1).sum())}/{len(cnt)}; series per plane per study mean {cnt.mean().round(2).to_dict()} max {cnt.max().to_dict()}")
out(f"- studies with >=1 fluid-sensitive series: {(ts.groupby('StudyInstanceUID').Fluid_Sensitive.max() == 1).sum()}")

out("\n## Gold labels (studies with provided labels)")
out(f"- prevalence: {gold[labs].mean().round(2).to_dict()}")
out(f"- positives per gold study: {gold[labs].sum(axis=1).astype(int).value_counts().sort_index().to_dict()}  (0 means an all-negative study)")
out(f"- positive counts: {gold[labs].sum().astype(int).to_dict()}")
c = gold[labs].corr()
out("- label pairs with |phi|>0.4: " + ", ".join(f"{a}~{b} {c.loc[a, b]:.2f}" for i, a in enumerate(labs) for b in labs[i + 1:] if abs(c.loc[a, b]) > 0.4))

out("\n## Reports")
r = tr.Report.fillna("")
out(f"- chars: {r.str.len().describe().round(0).to_dict()}")
out(f"- exact-duplicate report texts: {int(r.duplicated(keep=False).sum())} rows ({r.nunique()} unique)")
try:
    from langdetect import DetectorFactory, detect
    DetectorFactory.seed = 0
    def lid(t):
        try: return detect(t)
        except Exception: return "unk"
    lg = r.map(lid)
    out(f"- langdetect (seed 0) all: {lg.value_counts().head(15).to_dict()}")
    out(f"- langdetect gold: {lg[gold.index].value_counts().to_dict()}")
except ImportError:
    out("- langdetect not installed; language section skipped")
kw = {"ACL": r"\bACL\b|anterior cruciate|cruzado anterior|\bLCA\b", "MCL": r"\bMCL\b|medial collateral|colateral medial",
      "Meniscus": r"menisc|menisk", "Effusion": r"effusion|derrame|erguss", "Baker": r"baker",
      "Fracture": r"fractur|fraktur", "Synovitis": r"synov|sinov"}
out(f"- keyword hit fraction (EN/ES/DE stems only, crude): {{{', '.join(f'{k}: {r.str.contains(v, case=False, regex=True).mean():.2f}' for k, v in kw.items())}}}")

out("\n## Local DICOM (headers only)")
rows = []
for split, root in (("train", f"{D}/series_data/train_series"), ("test", f"{D}/series_data/test_series")):
    if not os.path.isdir(root): continue
    for st in sorted(os.listdir(root)):
        for se in sorted(os.listdir(f"{root}/{st}")):
            fs = glob.glob(f"{root}/{st}/{se}/*.dcm")
            if fs: rows.append((split, st, se, fs))
rows = rows[: args.dicom_limit]
meta = pd.concat([ts, tes]).set_index("SeriesInstanceUID")
recs = []
for split, st, se, fs in rows:
    ds = [pydicom.dcmread(f, stop_before_pixels=True) for f in fs]
    iop = np.array(ds[0].ImageOrientationPatient, float); nrm = np.cross(iop[:3], iop[3:])
    proj = np.array([d.ImagePositionPatient for d in ds], float) @ nrm
    inst = np.array([int(d.InstanceNumber) for d in ds]); g = np.diff(np.sort(proj))
    d0 = ds[0]; ps = float(d0.PixelSpacing[0])
    recs.append(dict(split=split, study=st[-6:], series=se[-6:], plane=meta.Anatomical_Plane.get(se), fluid=meta.Fluid_Sensitive.get(se),
                     n=len(ds), mfr=str(d0.get("Manufacturer"))[:8], tesla=d0.get("MagneticFieldStrength"), rows=d0.Rows, cols=d0.Columns,
                     px_mm=round(ps, 3), fov_mm=round(ps * d0.Rows), thick=d0.get("SliceThickness"), gap_mm=round(float(g.mean()), 2),
                     fname_order_is_anat=bool(np.all(np.diff(proj) > 0) or np.all(np.diff(proj) < 0)),
                     inst_vs_proj=int(np.sign(np.corrcoef(inst, proj)[0, 1])) if len(set(inst)) > 1 else 0,
                     bits=d0.BitsStored, signed=int(d0.PixelRepresentation), slope=d0.get("RescaleSlope"), pid=str(d0.get("PatientID"))[:8]))
if recs:
    df = pd.DataFrame(recs)
    out(df.to_string(index=False))
    out(f"\n- series on disk {len(df)}; studies {df.study.nunique()}; patients {df.pid.nunique()}; slices {int(df.n.sum())}")
    out(f"- filename order == anatomical order in {int(df.fname_order_is_anat.sum())}/{len(df)} series (0 expected: never sort by filename)")
    out(f"- InstanceNumber direction vs IPP-projection: +1 in {(df.inst_vs_proj == 1).sum()}, -1 in {(df.inst_vs_proj == -1).sum()} series (sign is not constant)")
    out(f"- FOV mm range {df.fov_mm.min()}-{df.fov_mm.max()}, pixel spacing {df.px_mm.min()}-{df.px_mm.max()}, matrix {df.rows.min()}-{df.rows.max()}, slices/series {df.n.min()}-{df.n.max()}")
    out(f"- manufacturers {df.mfr.value_counts().to_dict()}, field strength {df.tesla.value_counts().to_dict()}, PixelRepresentation {df.signed.value_counts().to_dict()}")
else:
    out("- no DICOM on disk")
