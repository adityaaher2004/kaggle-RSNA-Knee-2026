# Q-031 — masked weak-label objective + starter V100 training harness

**Patch type:** additive: apply to the root of your existing Quartz repo after T-DATA-1/2/3, Q-017, and Q-030. No existing source files are replaced.

**Scope:** The Q-031 SPEC item is the masked, confidence-weighted BCE. To unblock Q-032, the patch also includes a conservative train/validation CLI and atomic, resumable checkpoints. It is **not** a claim that Q-020 Glimmer ingestion, Q-023 grouped folds, Q-024 validation analysis, Q-032 feature caching/full training, or Kaggle submission have been finished.

## Files

| Relative path | Purpose |
|---|---|
| `src/oxalpha/losses/masked.py` | Q-LABEL-03 masked weighted BCE with strict null and zero-confidence semantics, optional train-fold `pos_weight`, exact numerator and denominator |
| `src/oxalpha/losses/__init__.py` | Exports |
| `src/oxalpha/training/records.py` | Normalized weak-label CSV/Parquet reader; gold exclusion; fixed fold + patient/duplicate group checks |
| `src/oxalpha/training/engine.py` | V100 fp16 GradScaler, exact-weight gradient accumulation, gradient clipping, validation and explicitly partial macro AUC |
| `src/oxalpha/training/checkpoints.py` | Atomic `last.pt` / `best_val.pt`, optimizer/scheduler/scaler/RNG resume and immutable data/config fingerprint |
| `src/oxalpha/training/__init__.py` | Exports |
| `scripts/train_student.py` | End-to-end online DICOM train loop using existing study loader and MRI-only student |
| `tests/test_masked_loss.py` | Perfect/no-label/rare-positive/zero-gradient tests |
| `tests/test_q031_records.py` | Gold, uncertain status, duplicate UID and split/group leakage tests |
| `tests/test_q031_engine.py` | Exact accumulation, empty-supervision skip and correctly flagged partial macro AUC |
| `tests/test_q031_checkpoint.py` | Safe resume, RNG and config mismatch tests |
| `tests/test_q031_cli.py` | End-to-end CPU synthetic MRI training CLI, saving and resuming checkpoints |

## Installation and tests

```bash
module load python/gpu/3.12.5
cd /N/scratch/aaher/RSNA/repo/quartz/vibe-code/project
# Copy/download ZIP here then:
unzip -o quartz_q031_patch.zip
export PYTHONPATH="$PWD/src:$PYTHONPATH"
python -m pytest -q tests/test_masked_loss.py tests/test_q031_*.py
```

Integration-tested locally with T-DATA-1/2/3 + Q-017 + Q-030: **82 passed, 2 skipped** (pydicom-dependent tests skipped in local environment). Full native V100 training with your real DICOMs and Glimmer labels has not been executed here.

Runtime requirements: PyTorch, NumPy, pydicom, scipy, timm, and normal DICOM codec plugins where needed. CSV labels require only Python's standard library. Parquet labels require `pandas` + `pyarrow`.

## Input contracts (fail closed)

You must supply the following user-controlled private paths. Neither `train.csv` report labels nor the gold-58 may be used as training targets.

1. `--weak-labels work/weak_labels.csv` (or `.parquet`): exactly `StudyInstanceUID` plus the exact 12 columns exported by `oxalpha.models.student.LABELS`; each value is `0`, `1`, or blank/null. Optional `<label>__confidence` and `<label>__status` columns per label. For the status field, `positive` must correspond to 1, `negative` to 0, and `uncertain`, `not_mentioned`, `not_processed`, or blank **abstain**. If status is missing, an explicit 0 or 1 is supervised with confidence 1, **uncalibrated**. Confidence is capped at 1, zero not floored. Additional fields are ignored and never passed to the student.
2. `--folds work/folds.csv`: `StudyInstanceUID,split` (`train` or `val`). Preferred grouping columns: `PatientGroup`, `ClusterGroup`, `group_id`, `patient_hash`, or `near_duplicate_cluster`. A shared value in **any** provided grouping column cannot cross train/val. Missing grouping metadata produces a warning, *not* a claim of patient-safe grouping. Freeze these fold assignments before training.
3. `--gold-ids work/gold_58_uids.txt`: one UID per line (or CSV with `StudyInstanceUID`), containing the **real predesignated 58 gold study IDs**. Never use dummy IDs with real training. Their appearance in either weak labels or folds is a **hard error**. This file is required.
4. `--data-root "$RSNA_DATA"`: directory with `train_series.csv`, `train_series/<StudyInstanceUID>/<SeriesInstanceUID>/*.dcm`.
5. A valid pretrained ConvNeXt-Tiny `timm` state dict via `--backbone-weights`, OR `--pretrained` (will rely on timm cache/network). Do not use randomly initialized ConvNeXt for the baseline.

### Prepare the normalized labels (example)

`StudyInstanceUID`, `ACL`, `MCL`, `Medial Meniscus`, `Lateral Meniscus`, `Medial OA`, `Lateral OA`, `PF OA`, `Effusion`, `Synovitis`, `Baker's`, `Contusion`, `Fracture` are **literal CSV header values**. Extra columns such as `ACL__confidence`, `ACL__status`, etc. may be present. Write normalized Glimmer outputs with Q-020; do *not* synthesize labels from reports merely to run the training script.

### Q-031 first GPU smoke training run

```bash
python scripts/train_student.py \
  --data-root "$RSNA_DATA" \
  --weak-labels work/weak_labels.csv \
  --folds work/folds.csv \
  --gold-ids work/gold_58_uids.txt \
  --backbone-weights work/convnext_tiny_backbone.pth \
  --out-dir work/q031_smoke \
  --epochs 1 --warmup-epochs 0 \
  --limit-train 8 --limit-val 8 \
  --size 224 --slices 16 --max-series 6 \
  --batch-size 1 --accumulate 8 --chunk-size 4
```

Use `--pretrained` in place of `--backbone-weights` only if timm's pretrained weights are already cached or can be downloaded. First smoke training performs real preprocessing and may be I/O-bound. `--limit-train/--limit-val` explicitly changes experiment identity and are **not** for final results.

### Full conservative V100 configuration

```bash
python scripts/train_student.py \
  --data-root "$RSNA_DATA" \
  --weak-labels work/weak_labels.csv \
  --folds work/folds.csv \
  --gold-ids work/gold_58_uids.txt \
  --backbone-weights work/convnext_tiny_backbone.pth \
  --out-dir work/q031_baseline \
  --epochs 12 --warmup-epochs 2 \
  --size 224 --slices 16 --max-series 6 \
  --batch-size 1 --accumulate 8 --chunk-size 4
```

**Important:** The online trainer rereads and preprocesses DICOM every epoch. At the reported ~7.4 seconds/study preprocessing, this would be very expensive for the entire set. Q-032 should add a versioned **frozen ConvNeXt feature cache** before committing to many epochs. Do not blindly run 12 online epochs without measuring throughput.

**Resume only when exactly the same config + input hashes are used:**

```bash
# Repeat the entire original command with:
  --resume work/q031_baseline/last.pt
```

Outputs remain under private `work/`: `last.pt`, `best_val.pt`, `provenance.json`, `coverage.json`, `metrics.json`, `history.jsonl`, and `train_seen_ids.txt`. Do not publish private study identifiers, DICOM, label rows, or checkpoints without reviewing competition data terms.

## Training semantics

- Never pass targets/weights/reports into `QuartzKneeStudent.forward`. Only Q-030's inference-safe batch reaches the model.
- Mathematically `sum(mask * confidence * BCEWithLogits) / sum(mask * confidence)`. A zero-denominator microbatch contributes no gradients; an entirely unsupervised accumulation window skips optimizer step entirely.
- Gradient accumulation uses **unnormalized numerator backwards per microbatch**, then divides unscaled gradients once by the **total valid weight in the window**. Correct with uneven supervision/confidence.
- V100 uses CUDA FP16 autocast + `torch.amp.GradScaler('cuda')`; CPU tests use float32. The ConvNeXt backbone stays frozen unless `--unfreeze-backbone` is explicitly passed.
- Validation prints AUC per label with observed pos/neg counts. Official 12-label macro AUC is **null unless all 12 labels have positive and negative validation studies**; otherwise reports a distinctly named partial macro and selects best by validation loss.
- A checkpoint bundles model, optimizer, scheduler, scaler, Python/Torch RNG, provenance and best metric; resume verifies label, gold, fold, series metadata and backbone weight hashes.

## Pending work

- Q-020 normalize and audit your actual Muse Glimmer labels.
- Q-023 build **truly patient/near-duplicate grouped frozen folds**; the trainer validates provided grouping values but cannot infer unidentified patient relationships.
- Q-032 implement cached frozen ConvNeXt slice features, verify measurable loss decrease/AUC on a non-gold fold, profile V100 training memory, and save complete checkpoint.
- Q-040/Q-042 offline Kaggle notebook and submission tests.
