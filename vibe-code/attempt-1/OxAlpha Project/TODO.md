# OxAlpha TODO

Source of truth: `specs.txt` (section refs in brackets). Order = build order. `[ ]` open, `[x]` done.
Hardware note: local box is CPU-only. Items marked **GPU** need Kaggle/rented GPU.

## Phase 0 — Verify unknowns (blocks everything)
- [ ] T-RULES-1 Read the Kaggle competition page: metric (assumed mean per-label AUC), submission limits, runtime limit, external-data rules, whether test DICOM or npz is given [S11, S13]
- [ ] T-DATA-1 Finish/obtain full dataset; delete the stale `rsna-knee-abnormality-detection.zip.kaggle-partial` after confirming data is complete [S0]
- [ ] T-DATA-2 Reverse-engineer how the provided `(24,224,224)` npz were made (slice selection, crop, normalisation) using the 5 local train series [S3.3]
- [ ] T-ENV-1 Pin env for Kaggle image (`timm`, `monai`, `SimpleITK`); local Python 3.14 is not the target [S14 F-21]
- [ ] T-LIT-1 Dated 2024–2026 literature scan (MIL, report-guided MRI, LUPI/KD); add citations to S15 before adopting any frontier idea

## Phase 1 — EDA (notebooks, CPU)
- [ ] 01_eda: series/study counts, plane × contrast, slices-per-series, missing planes per study
- [ ] 03_report_languages: language ID over 4,407 reports; length, templates, duplicates
- [ ] 02_label_cooccurrence: phi / conditional prob on the 58 gold rows (small!) [S11]
- [ ] Per-finding keyword hit-rate in reports (how many studies mention each of 12 findings)
- [ ] Visual spot check: 20 random npz per plane (orientation, contrast, blank slices)

## Phase 2 — Validation framework
- [ ] `validation/dedup.py` near-duplicate studies (report MinHash, then feature cosine) [S10]
- [ ] `validation/folds.py` 5-fold, dup-cluster groups, iterative stratification, gold-58 spread; write `folds.csv` once and freeze
- [ ] `validation/metrics.py` AUC (per-label/macro/micro), bootstrap CI, ECE
- [ ] `validation/oof.py` weak-OOF and gold-OOF tables
- [ ] Test: no StudyInstanceUID (or dup cluster) appears in two folds

## Phase 3 — Weak labels from reports (gates all training; CPU)
- [ ] `models/text_encoder.py` extractor: language ID → per-language lexicon → negation/uncertainty → `{-1,0,+1}^12` + confidence [S6]
- [ ] Lexicons for es, de, nl, en first; add languages found in 03_report_languages
- [ ] Validate on gold 58: per-label precision/recall; mask slots below `weak.min_precision` [S5]
- [ ] Freeze lexicon before any image-model results are viewed
- [ ] Cache XLM-R `z_text` (frozen) for all 4,407 reports
- [ ] Decide whether LLM-assisted labelling is worth adding (only if rules plateau)

## Phase 4 — Data pipeline
- [ ] `data/dicom.py` pydicom loader, IPP/IOP sort, transfer-syntax fallback, geometry dict [S3.1]
- [ ] `data/preprocessing.py` DICOM → 24×224×224 matching npz (SSIM > 0.99 on 5 local series) [S3.3]
- [ ] `data/dataset.py` `StudyBagDataset` + collate (pad to S_max=8, masks, meta ids) [S3.6, S4]
- [ ] Augmentation pipeline (consistent per series, no flips) [S3.7]
- [ ] Test: shapes match S4 table; padded series fully masked

## Phase 5 — Stage A baseline (first number to beat)
- [ ] `requirements.txt` install on GPU box; smoke test on CPU with tiny batch
- [ ] Cache frozen EffNet-B0 slice features (24×512 per series) **GPU**
- [ ] A0 mean-pool + logistic; A1 max-pool / gated MIL [S12]
- [ ] A2 six pathology-group heads; A3 series transformer; A4 metadata token
- [ ] Record weak-OOF + gold-OOF in experiment table

## Phase 6 — Stage B student (end-to-end) **GPU**
- [ ] `models/slice_encoder.py`, `mil.py`, `series_encoder.py`, `student.py` [S7]
- [ ] `losses/asymmetric.py` (+ BCE control), masking and sample weights
- [ ] Training loop: AMP, accumulation, EMA, layer-wise LR, checkpoint with config hash [S9]
- [ ] Profile real GPU memory; set micro-batch (spec estimate is unmeasured)
- [ ] A5 end-to-end vs frozen; A6 weak+gold vs gold-only
- [ ] Aux losses: plane/contrast from train_series.csv (`losses/auxiliary.py`)

## Phase 7 — Teacher + distillation **GPU**
- [ ] `models/fusion.py`, `teacher.py` with modality dropout [S7.7]
- [ ] `losses/kd.py` (logit, feature, attention), `losses/itc.py`, `losses/composite.py` with warm-up
- [ ] Check teacher leakage: does teacher beat student on images only? (F-2)
- [ ] Offline KD: freeze teacher, train student (C2)
- [ ] Ablations A7 (logit KD), A8 (ITC), A9 (feature/attention KD); λ search over {λ_KD, λ_ITC} [S8]
- [ ] Decision: keep teacher path only if it wins on weak-OOF and gold-OOF

## Phase 8 — Inference, ensemble, submission
- [ ] `inference/predict.py` series selection + TTA (no flips) [S13]
- [ ] `inference/ensemble.py` logit averaging across folds/seeds/archs
- [ ] `inference/make_submission.py` columns identical to `sample_submission.csv`
- [ ] Dry run on the 3 local test studies end to end
- [ ] Time check against Kaggle runtime limit

## Cross-cutting tests (one small test each, no frameworks)
- [ ] T-LEAK-1 `Student` has no text parameters and never receives report fields
- [ ] T-ORD-1 Slice sort uses IPP projection, shuffled input gives same order
- [ ] T-SPLIT-1 No fold leakage (see Phase 2)
- [ ] T-SHAPE-1 Forward pass shapes match S4 on a CPU dummy batch
- [ ] T-SUB-1 Submission has 3 rows × 13 cols, probabilities in [0,1]

## Deferred until a strong baseline exists
- [ ] ConvNeXt-T backbone (A11); 288 px (A12); label-graph head (A10); ITC memory queue; plane-specific models; LLM-assisted labels; 3D/SSL encoders
