# Q-031 hotfix — probabilistic Glimmer soft targets

This is a **small replacement patch**, applied **after** the Q-031 patch. The model (Q-030) and its 12 independent logits do **not** need modification. This patch changes **training-only** CSV parsing, loss masking, reporting, and checkpoint provenance.

## Why

Blended labels such as `0.25`, `0.5`, and `0.75` are valid **soft Bernoulli targets**, not invalid binary annotations. For logits `z` and soft target `y ∈ [0,1]`, the loss is:

`L = sum(observed * confidence * BCEWithLogits(z, y)) / sum(observed * confidence)`

where `BCEWithLogits(z,y) = -[y log sigmoid(z) + (1-y) log(1-sigmoid(z))]`.

The gradient for each active cell, absent class reweighting, is proportional to `sigmoid(z) - y`. No thresholding, rounding, or pseudo-label binarization is performed. `0.25` is a target probability **not** automatically a confidence weight. Optional `<label>__confidence` remains the separate reliability weight and defaults to 1 when absent. There is no automatic certainty-from-distance-to-0.5 penalty; introduce one only after auditing the extractor calibration.

## Patch contents

- `src/oxalpha/training/records.py`: parse finite [0,1] targets (incl. fractional) or null/unknown. Out-of-range/invalid values fail closed. Explicit `__status=uncertain/not_processed/not_mentioned` continues to mask. Hard contradictory positive/negative statuses still reject. Soft targets with positive/negative extractor status are allowed and retain their literal fractional value.
- `src/oxalpha/losses/masked.py`: active cells are finite targets in [0,1], not just 0/1. NaN and legacy -1 are missing, other out-of-range values are errors. Zero-confidence and unknown labels give exactly zero gradient. Exact denominator, including sums below 1.
- `src/oxalpha/training/engine.py`: soft targets train normally, but ROC-AUC uses **hard-labeled observed cells only**, never labels rounded from probabilities. Reports `soft_targets_excluded_from_auc` and `hard_macro_auc`. When any observed validation label is soft, `official_macro_auc` is `null`, and model selection falls back to negative validation soft BCE. AUC on weak-label hard subsets is not a genuine gold-label benchmark.
- `scripts/train_student.py`: label coverage counts `pos`, `neg`, and `soft` separately, plus the sum of expected positive probabilities. Soft-only train/val folds are accepted. `target_semantics=soft_bernoulli_bce_v2` in provenance intentionally prevents resume from a pre-hotfix checkpoint without explicit migration.
- `tests/test_q031_soft_labels.py`: CSV parsing, gradients, confidence, masking, invalid values, AUC exclusion, and an end-to-end all-soft trainer+checkpoint run.

## Apply on Quartz

From the repository root, with the **original Q-031 patch already applied**:

```bash
module load python/gpu/3.12.5
cd /N/scratch/aaher/RSNA/repo/quartz/vibe-code/project
unzip -o quartz_q031_soft_labels_patch.zip
export PYTHONPATH="$PWD/src:$PYTHONPATH"
python -m pytest -q tests/test_masked_loss.py tests/test_q031_*.py
```

Then use the existing trainer; the source CSV no longer needs binarization:

```bash
python scripts/train_student.py \
  --data-root "$RSNA_DATA" \
  --weak-labels "$RSNA_DATA/llm_labels_v4_blend.csv" \
  --folds work/folds.csv \
  --gold-ids work/gold_58_uids.txt \
  --backbone-weights work/convnext_tiny_backbone.pth \
  --out-dir work/q031_soft_smoke \
  --epochs 1 --warmup-epochs 0 \
  --limit-train 8 --limit-val 8 \
  --size 224 --slices 16 --max-series 6 \
  --batch-size 1 --accumulate 8 --chunk-size 4
```

Alternatively use `--pretrained` in place of `--backbone-weights` if weights are cached or available for downloading. Paths for `--folds`, `--gold-ids`, and local backbone weights are examples; retain your actual validated paths. Gold exclusion and grouped fold requirements remain unchanged. **Use a fresh `--out-dir` and do not resume old Q-031 checkpoints with changed target semantics.**

## Verification

Local integrated suite (T-DATA-1/2/3, Q-017, Q-030, Q-031, and this patch): **98 passed, 2 skipped**. The skipped tests require real DICOM integration dependencies in this local container. No Slate dataset, V100 job, real blended CSV, or full training run was available in this environment.

Important assumption: numeric values in `llm_labels_v4_blend.csv` represent **probabilities** between 0 and 1. If the values instead encode ordinal categories, vote counts, or another scoring scale, their semantics must be checked before treating them as Bernoulli targets.
