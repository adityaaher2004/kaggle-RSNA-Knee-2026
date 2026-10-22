import os
import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path(os.environ["RSNA_DATA"])
WEAK = Path("work/llm_labels_v4_train_only.csv")
GOLD = Path("work/gold_58_uids.txt")
OUTPUT = Path("work/folds.csv")

SEED = 42
VAL_FRACTION = 0.15

labels = pd.read_csv(WEAK, dtype={"StudyInstanceUID": str})
series = pd.read_csv(
    DATA / "train_series.csv",
    dtype={"StudyInstanceUID": str},
)

gold_ids = {
    uid.strip()
    for uid in GOLD.read_text().splitlines()
    if uid.strip()
}

label_ids = set(labels["StudyInstanceUID"].str.strip())
series_ids = set(series["StudyInstanceUID"].str.strip())

eligible = sorted((label_ids & series_ids) - gold_ids)

if len(eligible) < 2:
    raise ValueError("Insufficient eligible studies for train/val split")

# Prefer grouping by a verified patient/duplicate identifier.
group_col = next(
    (
        name for name in
        ["PatientGroup", "patient_group", "patient_hash",
         "near_duplicate_cluster", "group_id"]
        if name in series.columns
    ),
    None,
)

if group_col:
    groups = series[
        series["StudyInstanceUID"].isin(eligible)
    ][["StudyInstanceUID", group_col]].drop_duplicates()

    counts = groups.groupby("StudyInstanceUID")[group_col].nunique()
    if (counts > 1).any():
        raise ValueError("Conflicting group IDs for a study")

    groups = groups.drop_duplicates("StudyInstanceUID")
    groups[group_col] = groups[group_col].fillna(
        groups["StudyInstanceUID"]
    )

    rng = np.random.default_rng(SEED)
    unique_groups = sorted(groups[group_col].unique())
    rng.shuffle(unique_groups)

    n_val = max(1, min(
        len(unique_groups) - 1,
        round(len(unique_groups) * VAL_FRACTION)
    ))
    val_groups = set(unique_groups[:n_val])

    groups["split"] = np.where(
        groups[group_col].isin(val_groups), "val", "train"
    )

    folds = groups.rename(columns={group_col: "group_id"})

else:
    print(
        "WARNING: No verified patient/duplicate grouping column. "
        "Creating a study-level split only."
    )

    rng = np.random.default_rng(SEED)
    shuffled = np.array(eligible, dtype=object)
    rng.shuffle(shuffled)

    n_val = max(1, min(
        len(shuffled) - 1,
        round(len(shuffled) * VAL_FRACTION)
    ))
    val_ids = set(shuffled[:n_val])

    folds = pd.DataFrame({
        "StudyInstanceUID": eligible,
        "split": [
            "val" if uid in val_ids else "train"
            for uid in eligible
        ],
    })

assert not set(folds["StudyInstanceUID"]) & gold_ids
assert folds["StudyInstanceUID"].is_unique
assert set(folds["split"]) == {"train", "val"}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
folds.to_csv(OUTPUT, index=False)

print("Saved:", OUTPUT)
print("Train:", (folds["split"] == "train").sum())
print("Validation:", (folds["split"] == "val").sum())
print("Protected gold excluded:", len(gold_ids))
