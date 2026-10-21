# OxAlpha TODO (v0.2.0)

Source of truth: [SPEC.md](SPEC.md). `[ ]` open · `[x]` done. **L** = local CPU agent, **H** = HPC agent (GPU). Order = build order.
Deadline per RSNA page: contest runs through **2026-10-22** (today 10-20) → do Tier 0 completely before touching Tier 1.
Setup everywhere: `pip install -e .` from `vibe-code/attempt-1/oxalpha`, `export RSNA_DATA=<dir with train.csv and series_data/>`.
Handoff local↔HPC is git only: land code on the `local` branch, HPC agent pulls; HPC returns results as small text files (`work/results/*.md|csv`, no data, no checkpoints in git).

## Done this cycle
- [x] Data inspected, `DATA_REPORT.md` generated (`scripts/eda_local.py`), contract test `tests/test_data_contract.py` passes
- [x] Scaffold → `src/oxalpha/` package + `pyproject.toml`; `base.yaml` fixed (I-05, I-08); DICOM/checkpoints untracked + `.gitignore` (I-10)
- [x] SPEC v0.2.0 + this TODO

## Phase 0 — Unknowns that gate everything
- [ ] **T-RULES-1 (human/H)** Read Kaggle Overview/Evaluation/Rules: metric, runtime, external-data/pretrained-weight rules, efficiency-award criteria, final-submission time. Write answers into SPEC F11/F10/D-2. *Verify:* `Manual:` SPEC F11 retagged `[S]`→`[P]` with URL.
- [ ] **T-HPC-1 (H)** Record Quartz GPU type/count, partition, wall-time, storage path for full DATA; fill SPEC §9. Confirm full DATA passes `python tests/test_data_contract.py` and run `python scripts/eda_local.py --data $RSNA_DATA --dicom-limit 300 > work/results/eda_hpc.md`; diff against local DATA_REPORT, update SPEC F1–F9 if any differ (I-09: count of studies).
- [ ] **T-ENV-1 (H)** Pin versions for the Kaggle/HPC image into `requirements.txt` (local Py 3.14 is not the target). Check that timm weights can load offline.

## Phase 1 — Data pipeline (REQ-DATA-*) — L builds, H runs at scale
- [ ] **T-DATA-1 (L)** `data/dicom.py`: load, signed/unsigned pixels, geometry dict. *Verify:* `python tests/test_dicom_load.py` (write it).
- [ ] **T-DATA-2 (L)** `data/dicom.py`: IPP-normal ordering with canonical direction (F8). *Verify:* `python tests/test_slice_order.py` (write it: shuffle, same order; one direction per plane).
- [ ] **T-DATA-3 (L)** `data/preprocessing.py`: FOV resample, per-series percentile normalisation. *Verify:* `python tests/test_geometry.py` (write it).
- [ ] **T-DATA-4 (L→H)** `scripts/scan_dicom_manifest.py` (headers only, resumable) → `work/manifest.csv` for all 24,371 series + test. *Verify:* `python tests/test_manifest.py work/manifest.csv`. HPC runs full set; output also answers F9 (patients with >1 study?).
- [ ] **T-DATA-5 (L)** `data/dataset.py` `StudyBagDataset` (≤ 8 series, masks, meta ids, no text fields in the image batch). *Verify:* `python tests/test_shapes.py`.
- [ ] **T-DATA-6 (H, optional)** npz parity check (REQ-DATA-6) only if the volumes exist on HPC.

## Phase 2 — Weak labels (REQ-NLP-*) — start in parallel with Phase 1 (CPU)
- [ ] **T-NLP-1 (L)** Language ID log for all reports (`langdetect` seed 0 + manual check of 20 per language; F5 accuracy is unverified).
- [ ] **T-NLP-2 (L)** Extractor v1, built blind to gold: lexicon+negation for languages with ≥ 100 reports, label slot masked otherwise. Write down the written label definitions first (`docs/label_definitions.md`).
- [ ] **T-NLP-3 (H)** Extractor v2 candidate: offline multilingual LLM → JSON schema (needs D-2). Run on all 4,407 reports; store raw + parsed.
- [ ] **T-NLP-4 (L)** `validation/score_extractor.py`: freeze (git hash), score once on gold-58 with Wilson CIs, write `work/results/extractor_gold.md`, emit `weak_labels.csv` (+1/0/−1, confidence, usable-slot mask by precision rule). *Verify:* `python -m oxalpha.validation.score_extractor --gold`.
- [ ] **T-NLP-5 (L)** Cache frozen XLM-R report embeddings (only needed for Tier 2).

## Phase 3 — Validation (REQ-VAL-*) — L
- [ ] **T-VAL-1** `validation/dedup.py` report MinHash clusters; patient groups from manifest (T-DATA-4).
- [ ] **T-VAL-2** `validation/folds.py` → frozen `folds.csv` (5 folds, groups, gold spread). *Verify:* `python tests/test_split_leakage.py work/folds.csv`.
- [ ] **T-VAL-3** `validation/metrics.py` macro/per-label AUC (skip single-class labels explicitly), bootstrap CI, ECE; `validation/oof.py` writes weak-OOF + gold-OOF table. *Verify:* `python tests/test_metrics.py` (synthetic: perfect→1, random≈0.5, constant label skipped).

## Phase 4 — Tier 0 submission (do this before anything in Phase 5)
- [ ] **T-T0-1 (H)** Extract frozen EfficientNet-B0 slice features for all series → `work/feats/*.npy` (fp16). Log seconds/series.
- [ ] **T-T0-2 (L or H)** A0: mean/max pool over slices and series + plane/contrast one-hot → logistic/ridge per label on gold+weak, 5-fold OOF. Write the experiment table row (weak-OOF, gold-OOF, CI).
- [ ] **T-T0-3 (H)** `inference/predict.py` + `inference/make_submission.py` for the Tier-0 model, DICOM→submission end to end on the 1 local test study. *Verify:* `python tests/test_submission.py submission.csv` (3 rows × 13 cols per sample file; [0,1]).
- [ ] **T-T0-4 (H)** Kaggle notebook dry run offline (internet off) with timing; submit Tier 0.

## Phase 5 — Tier 1/2 (only after Phase 4 is on the leaderboard)
- [ ] **T-M-1 (L)** `models/{slice_encoder,mil,series_encoder,student}.py` (REQ-MODEL-1…5) with CPU dummy-batch test and `tests/test_leak.py` (REQ-LEAK-1).
- [ ] **T-M-2 (L)** `losses/asymmetric.py` (+ BCE control), masking and weights; `tests/test_loss_mask.py` (masked slot ⇒ zero grad).
- [ ] **T-M-3 (H)** Training loop (AMP, accumulation, EMA, checkpoint with config hash); profile real GPU memory before fixing micro-batch.
- [ ] **T-M-4 (H)** Ablations A1→A5 on frozen `folds.csv`, 2 seeds; fill SPEC §10 table.
- [ ] **T-M-5 (H)** A6 / A6b (weak vs gold-only, richer report targets).
- [ ] **T-M-6 (H)** Tier 2 teacher + logit-KD (A7) only if A6 shows weak labels help.
- [ ] **T-M-7 (H)** `scripts/profile_student.py` params/FLOPs/seconds-per-study for the efficiency award.

## Phase 6 — Literature and interpretability
- [ ] **T-LIT-1 (L)** Read the four 2024–2026 papers in `Readings.txt` and the PDFs in `RSNA/papers/`; for each: what is adopted, what is adapted for study→series→slice with missing text at test, ablation needed. Add to SPEC §11 only after reading.
- [ ] **T-INTERP-1 (L/H)** Slice-occlusion check on 20 val studies before claiming any attention-based localisation (I-12).

## Decisions requiring a human
- [ ] **D-1** Accept Tier 0 first given the 10-22 deadline (I-04)?
- [ ] **D-2** Are external pretrained weights (timm, LLM, XLM-R) allowed and available offline on Kaggle? Which LLM is licensed/available on Quartz?
- [ ] **D-3** The 79 committed DICOMs remain in git history of `origin` (GitHub, visibility unknown). Make private / rewrite history? (Not done by agent: destructive.)
- [ ] **D-4** 4,407 train studies vs ">5,000" in RSNA text — expected?
- [ ] **D-5** Merge/rename strategy: `OxAlpha Project` → `oxalpha` and `src/oxalpha` layout is on branch `local`; `hpc` worktree and `master` still have the old path. Approve before merging.
- [ ] **D-6** Untracked-but-present 5 MB tokenizer files (×4 in `self-code/learn/xlm-r`) and `.mhtml` chat are still tracked; remove from git?

## Deferred until a strong baseline
ConvNeXt-T · 288 px · label-graph head · image-text InfoNCE + memory queue · feature/attention KD · plane-specific models · 3D/SSL encoders · test-time sliding windows.
