# OxAlpha TODO

Source of truth: `specs.txt` v0.2.0 (section refs in brackets). Order = build order; dates are targets (spec S16).
`[ ]` open, `[x]` done. **GPU** = Slurm h100 job. **USER** = needs you (accounts, rules, Kaggle UI).
Deadline rule R5: anything that cannot produce a measured result by **2026-10-20** moves to "Deferred".

## P0 — Unblock (today, 10-14 → 10-15)
- [ ] **USER** P0.1 Team/merger deadline **2026-10-15**: confirm you are entered (rules accepted) and decide on any team merge [S0.3]
- [ ] **USER** P0.2 T-RULES-1 Read Overview/Evaluation/Rules on Kaggle; confirm macro-AUC, 9 h, internet off, submission format, efficiency formula [S0.3, S11, S13]
- [ ] **USER** P0.3 T-RULES-2 Rules on external data/models: is a locally-run open LLM allowed for labeling? Are other teams' public label datasets / public notebooks allowed in a final submission? [S6.2, S6.4, F-21]
- [x] P0.4 Inspect repo, HPC, data; spec 0.2.0, layout, TODO (2026-10-14)
- [ ] P0.5 `scripts/setup_env.sh` → venv on python/3.14.5; check timm/pydicom/gdcm wheels install on 3.14; write `requirements.lock` [S0.1]
- [ ] P0.6 Fix Kaggle CLI (old ~/.local one points to a removed python3.13); `kaggle competitions files rsna-knee-abnormality-detection` works [S0.1]
- [ ] P0.7 `scripts/download_data.sh csv` → 5 CSVs in `full_data/...` [S0.2]
- [ ] P0.8 `scripts/download_data.sh dicom` in a long CPU job/tmux (~570 GB); verify ~819,640 .dcm and 24,371 + 15 series dirs non-empty [S0.2]
- [ ] P0.9 Purge protection: plan SDA (or home) copies for labels, folds, cache, checkpoints [F-18]
- [ ] P0.10 Push repo to GitHub after each work session (repo itself lives on purgeable scratch)

## P1 — EDA on CSVs (10-15, CPU, needs P0.7)
- [ ] Re-verify S0.3 [V-old] facts: 58 gold rows, label counts, series/study, plane counts, Fluid == FatSat
- [ ] 03_report_languages: language ID over 4,407 reports; length; templates; exact-duplicate reports
- [ ] 02_label_cooccurrence: phi + conditional probs on gold 58 [S7.6, S11]
- [ ] Read ~30 reports across languages + all 58 gold rows → notes for the prompt (no report text in committed outputs)

## P2 — Labels: the teacher (10-15 → 10-16; gates training)
- [ ] P2.1 `labels/gold_eval.py` + `validation/metrics.py` (macro/per-label AUC, bootstrap CI) [S6.5, S11]
- [ ] P2.2 `labels/llm_labeler.py` with vLLM JSON-constrained decoding; smoke test on 20 reports on h100-debug **GPU** [S6.2]
- [ ] P2.3 Write `labels/prompts/labeler_v1.txt` (12 definitions, compartment rules, negation/hedging, schema)
- [ ] P2.4 Run 2 candidate models on all 4,407 **GPU** (`scripts/slurm/label_llm.sbatch`); compare on gold → table rows L1 [S12]
- [ ] P2.5 (if allowed by P0.3) download public label sets; `labels/fuse.py` stacking on gold → L2 [S6.4]
- [ ] P2.6 Measure "not_mentioned" vs gold per label → silence prior [S5]
- [ ] P2.7 Freeze label set `labels_vX.csv` (q, c) + record version; no edits after image comparisons begin [S6.6]
- [ ] P2.8 (low priority) `labels/rules_extractor.py` en/es/de/nl/tr as disagreement flagger [S6.3]

## P3 — Data pipeline (10-15 → 10-16; needs P0.8 for full run, sample series for dev)
- [ ] P3.1 `data/dicom.py` headers + pixel decode with fallback chain; log decoder per series [S3.1]
- [ ] P3.2 `data/geometry.py` IPP/IOP sort, dedup/localiser drop, canonical orientation, 0.6 mm resample, fg-centred 256 crop, side flag [S3.2]
- [ ] P3.3 `data/preprocessing.py` cache v1 writer + sharded driver; `scripts/slurm/preprocess.sbatch` over 24,371 series **CPU** [S3.4]
- [ ] P3.4 `data/metadata.py` header table parquet (scanner, field strength, spacing, transfer syntax) [S10]
- [ ] P3.5 Visual check: 10 series per plane × contrast after canonicalisation (orientation consistent? knee centred?)
- [ ] P3.6 `data/series_select.py`, `data/augment.py`, `data/dataset.py` + collate [S3.5–S3.7, S4]
- [ ] T-ORD-1 `tests/test_order.py`; T-SHAPE-1 dataset part

## P4 — Validation framework (10-16)
- [ ] `validation/sites.py` site proxy clusters (header table + language) [S10]
- [ ] `validation/dedup.py` report MinHash (image cosine added after Stage A features exist)
- [ ] `validation/folds.py` → `folds.csv` + checksum, frozen; gold ~11-12 per fold
- [ ] `validation/oof.py` gold-OOF / label-OOF tables + experiment-table row writer [S12]
- [ ] T-SPLIT-1 `tests/test_split.py`

## P5 — Stage A: frozen features (10-16 → 10-17)
- [ ] `training/cache_features.py` EffNetV2-S features, t_eval=16 windows, all series **GPU** [S9]
- [ ] `models/mil.py` (mean/max/gated), `models/embeddings.py`, `models/series_encoder.py`, `models/heads.py`, `models/student.py` [S7]
- [ ] `losses/soft_bce.py`, `losses/composite.py` [S8]
- [ ] `training/train_stage_a.py`: rows A0, A1, A2, A4, A10 and label rows L0–L2 [S12]
- [ ] T-SHAPE-1, T-LEAK-1

## P6 — First submission (target 10-17)
- [ ] `inference/predict.py` (producer/consumer decode, time guard, prior fallback), `inference/make_submission.py` [S13]
- [ ] `kaggle/build_wheels.sh` run once in a Kaggle notebook with internet → dataset `oxalpha-wheels` **USER**
- [ ] Upload code + weights datasets (`kaggle/dataset-metadata.json`) **USER**
- [ ] `kaggle/infer_notebook.py`; dry run on the 3 local test studies; T-PARITY-1, T-SUB-1
- [ ] Submit Stage A model (or 1-fold Reader) → first public LB number

## P7 — Stage B: the Reader (10-17 → 10-19) **GPU**
- [ ] `training/loops.py`, `amp.py`, `schedulers.py`, `ema.py`, `train_stage_b.py` [S9]
- [ ] h100-debug profiling run: peak memory + img/s at micro-batch 4; adjust micro_batch/accum
- [ ] B1 EffNetV2-S 5 folds (`sbatch scripts/slurm/train_fold.sbatch`) → gold-OOF, label-OOF
- [ ] B2 ConvNeXt-T 5 folds in parallel (ensemble diversity)
- [ ] B3 hard targets/ASL; A14 t_eval 10/16/24; A13 left→right mirroring (only if time)
- [ ] Submit B1 fold ensemble

## P8 — Refine + ensemble (10-20 → 10-21)
- [ ] B5 noisy-student targets from B1 OOF (`configs/stage_b2_refine.yaml`) [S8 L_KD]
- [ ] Extra seeds for promoted configs
- [ ] `inference/ensemble.py` logit-avg within family, rank-avg across; OOF weights [S13]
- [ ] (if allowed by P0.3) blend with best public notebook outputs; decide weight on public LB + OOF
- [ ] Full-length timing run of the final kernel on Kaggle (< 7.5 h projected)

## P9 — Final (10-22)
- [ ] Pick 2 final submissions: best OOF-justified ensemble + best public-LB ensemble
- [ ] Freeze spec version, tag git commit, archive runs to SDA

## Cross-cutting tests (`tests/`)
- [ ] T-LEAK-1 `test_leak.py`  - [ ] T-ORD-1 `test_order.py`  - [ ] T-SPLIT-1 `test_split.py`
- [ ] T-SHAPE-1 `test_shapes.py`  - [ ] T-SUB-1 `test_submission.py`  - [ ] T-PARITY-1 `test_parity.py`
- [ ] T-SPEC-1 `test_spec_sync.py` (SPEC tags ↔ sections, [cfg] keys ↔ base.yaml)

## Deferred (post-deadline unless the plan runs ahead) [S16]
- [ ] Multimodal teacher: XLM-R `text_encoder.py`, `fusion.py`, `teacher.py`, ITC, feature/attention KD, `train_stage_c.py` [S7.7]
- [ ] Label-graph / GNN head [S7.6]; 3D / SSL encoders; segmentation-assisted ROI features; plane-specific models
- [ ] 320 px cache v2 (B4); Efficiency-track entry
