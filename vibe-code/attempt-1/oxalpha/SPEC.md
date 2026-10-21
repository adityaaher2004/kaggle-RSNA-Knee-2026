# OxAlpha SPEC v0.2.0 — RSNA Knee Abnormality Detection

Status: DRAFT for human review · Date: 2026-10-20 · Supersedes `specs.txt` v0.1.0 (still in git history at `7ad205f`)
Data facts regenerate with `python scripts/eda_local.py --data $RSNA_DATA` → [DATA_REPORT.md](DATA_REPORT.md).
Task list: [TODO.md](TODO.md). Every task there cites a requirement ID below.

## 0. Rules of this document (spec-anchored, spec-as-source)

1. **SPEC.md is the source of truth.** Numbers code consumes live in `configs/*.yaml` and are tagged `[cfg: key]`; change the YAML and this file in the same commit.
2. **Structure changes (module, tensor shape, loss term) are made here first**, then code. Each module docstring starts `SPEC: <req id>`.
3. **Every claim carries a tag:** `[V]` verified on local files (command given), `[P]` verified from a primary web source (URL given), `[S]` secondary/unconfirmed source, `[E]` established method from literature, `[H]` hypothesis needing an ablation, `[U]` unknown — do not assume.
4. **Requirement IDs** (`REQ-*`) each have a *Verify* command whose exit code decides. `Manual:` marks human checks. A command that does not exist yet says so; building it is part of the task.
5. **Nothing unavailable at test time (report text, labels) may enter the student inference path** (REQ-LEAK-1).
6. `[H]` items ship only after winning an ablation (§10). Simpler wins ties.

## 1. Verified facts (re-run to refresh)

| # | Fact | Tag / how |
|---|---|---|
| F1 | `train.csv`: 4,407 unique studies; `train_series.csv`: 24,371 series over the same 4,407 studies; 3–14 series/study (median 5). | [V] `python tests/test_data_contract.py` |
| F2 | **Only 58 studies (1.3%) have any of the 12 labels, and those have all 12. 4,349 have none.** No partial labels. Reports exist for all 4,407. | [V] same |
| F3 | Gold positives (n=58): ACL 24, MCL 9, MedMen 26, LatMen 23, MedOA 15, LatOA 11, PF-OA 21, Effusion 35, Synovitis 27, Baker's 12, Contusion 19, Fracture 18. **No gold study is all-negative** (every one has ≥1 positive; 1–9 positives each) → the gold set is enriched, not a population sample. | [V] DATA_REPORT |
| F4 | Every study has all 3 planes (Sag 9,864 / Cor 8,609 / Ax 5,898 series) and ≥1 fluid-sensitive series. `Fluid_Sensitive == Fat_Suppression` in all 24,371 series → **one bit, not two**. | [V] test_data_contract |
| F5 | Reports: median 977 chars (max 4,743), 177 rows are exact-duplicate texts. `langdetect` (seed 0) finds en 1736, es 682, tr 546, hr 406, el 321, de 262, bg 220, nl 153, fr 81 — accuracy of this ID is unverified; RSNA states "about a dozen languages". | [V]+[P] |
| F6 | **Local images are a tiny sample:** 3 train studies (15 series, 460 slices, *none labeled*) and 1 test study (5 series, 138 slices). Local `test.csv`=3 studies / `test_series.csv`=15 series are a stub; real test is hidden. → **No image model can be evaluated locally.** | [V] DATA_REPORT |
| F7 | Local DICOM (20 series): 3 vendors (Philips, Siemens, GE), 1.5 T and 3 T, matrix 416–768, pixel spacing 0.221–0.433 mm, **FOV 150–180 mm**, 16–48 slices/series, slice gap ≠ thickness (e.g. 4.63 vs 3.4 mm). `PixelRepresentation` is 1 (signed) in the test study, 0 in train; `RescaleSlope` 1–3.3 or absent; intensity p99 ranges 181–1816 between series. | [V] DATA_REPORT |
| F8 | **Filename order ≠ anatomical order in 20/20 series.** InstanceNumber vs slice-normal projection has sign +1 in 11 and −1 in 9 series → the sort *direction* is not constant either. | [V] DATA_REPORT |
| F9 | `PatientID` (UUID-like) is present in DICOM headers; 4 local studies = 4 distinct patients. Whether one patient has several studies in the full set is `[U]`. | [V] / [U] |
| F10 | Competition runs "through October 22, 2026"; training set ">5,000 knee MRI exams … about a dozen languages … 16 sites"; test annotated by expert radiologists; new **"most efficient models"** award. | [P] <https://www.rsna.org/news/2026/august/ai-challenge-knee-mri> |
| F11 | Metric = macro ROC-AUC over the 12 labels; code competition ≤ 9 h, internet off. **Single secondary source, page now 404.** | [S] — confirm on Kaggle (REQ-RULES-1) |
| F12 | `sample_submission.csv`: one row per `StudyInstanceUID`, 12 float columns (0.5), same names/order as `train.csv` labels. | [V] test_data_contract |
| F13 | Local env: Python 3.14.8, torch 2.14.1 CPU, timm 1.0.30, monai, SimpleITK, transformers 4.57, sklearn, iterstrat, langdetect, datasketch, pylibjpeg installed. No CUDA. | [V] `pip list` |

## 2. Issues register (found in this review; each has a TODO)

| ID | Sev | Issue | Resolution |
|---|---|---|---|
| I-01 | **Critical** | 98.7% of studies have no label; reports are the *label source*, not only privileged info. Report→label quality caps the whole system. | Weak-label pipeline first (REQ-NLP-*); trust gated by gold precision. |
| I-02 | High | Gold-58 is enriched (no negatives) and tiny (MCL has 9 positives). AUC on it has huge CI and prevalence ≠ population. | Report pooled gold AUC with bootstrap CI; never select on gold alone (REQ-VAL-3). |
| I-03 | High | **Gold reuse.** Old spec tuned the extractor on gold-58 *and* evaluated the student on gold-58. Extractor errors then correlate with the evaluation set. | Extractor developed blind to gold; gold used once to score the frozen extractor, then as eval (REQ-NLP-3). |
| I-04 | High | **Schedule.** Contest ends 2026-10-22 [P]; today is 2026-10-20. The full teacher/KD plan cannot finish. | §4 Tier 0 = minimum viable submission, defined first. Human decision D-1. |
| I-05 | High | `base.yaml` used `${...}` interpolation (PyYAML can't) and a relative `../../../..` data path. | Fixed: `data_root` from env `RSNA_DATA`. |
| I-06 | High | Previous S0 facts are stale: it cited 5 train / 3 test studies, 1038/557 slices and 24,371 `rsna-knee-mri-processed-3d-volumes/*.npz` (24×224×224 uint8). **None of that is in `DATA/` now** (only 3+1 studies). The npz has no verified provenance; test arrives as DICOM. | DICOM→tensor pipeline is canonical (REQ-DATA-*). npz only an optional accelerator on HPC if parity-tested (REQ-DATA-6). |
| I-07 | Med | `Fluid_Sensitive` and `Fat_Suppression` are identical (F4). Old design implied two metadata inputs. | One `contrast` bit + plane (REQ-MODEL-4). |
| I-08 | Med | Scaffold had top-level packages `data/ models/ losses/…` with no package root (import-name collisions, no install). Folder name had a space. | Done: `oxalpha/src/oxalpha/{data,models,…}`, `pyproject.toml`. |
| I-09 | Med | Training sample says ">5,000" exams [P] but `train.csv` has 4,407. | `[U]` — maybe test is included in the 5,000, or the CSV is a subset. Human D-4. |
| I-10 | Med | **Repo hygiene:** 79 `.dcm` files (competition data) were tracked in git with a GitHub remote (visibility unknown); an HF token file exists on disk (gitignored in `master`, not in history: 0 commits). 5 MB tokenizer files duplicated ×4 tracked. | Done: `.dcm`/`.ipynb_checkpoints` untracked + `.gitignore`. History still contains the DICOMs → D-3. |
| I-11 | Low | Old plan carried many unproven pieces up front (GNN head, ITC memory queue, attention/feature KD). | Moved to Deferred (§10). |
| I-12 | Low | Attention weights are not explanations (F-8 in old spec stands). | Use for slice triage only; localisation claims need occlusion check (REQ-INTERP-1). |
| I-13 | Low | Local machine cannot test models on real data (F6). | CPU tests use synthetic tensors; all learning on HPC. |

## 3. Problem formulation

Study *i* → series *j=1..m_i* (3–14) → slices *k=1..N_j* (16–48 observed). Train: `(X_i, T_i, M_i, y_i)`, `y_i ∈ {0,1,NA}^12` (NA for 98.7%). Test: `(X_i, M_i)`; `M_j = (plane_j, contrast_j)` comes from `*_series.csv` and **is** available at test time. Output: 12 probabilities per study.

Not plain MIL: bag-of-bags, weak/missing top-level labels, and a text view that exists only in training (LUPI [Vapnik & Vashist 2009] / distillation [Hinton 2015]).

**Metric:** treat as per-label macro ROC-AUC `[S]`. Consequences if true: rank-based, threshold-free; per-label ranks matter, calibration doesn't; label with fewer than 2 classes in a split must be skipped explicitly, not NaN-averaged. If Kaggle says otherwise (log-loss, weighted), only §9 ensembling and loss calibration change.

## 4. Strategy and tiers (what to build first)

The label bottleneck means **extractor quality and a sound validation set dominate architecture choices**. Therefore:

- **Tier 0 — minimum viable submission (target: working in ≤ 1 day, runnable offline on Kaggle).**
  Weak labels (REQ-NLP) → frozen ImageNet backbone features per slice (REQ-DATA, REQ-MODEL-1) → mean/max pooled series features + plane/contrast one-hot → per-label logistic/ridge or small MLP → `submission.csv`. No fine-tuning, no teacher. Also acts as the ruler for everything else.
- **Tier 1 — end-to-end MRI-only student** (EffNet-B0 2.5D + gated-attention MIL + series transformer), trained on gold+weak labels with ASL. Promote only if it beats Tier 0 on weak-OOF *and* gold-OOF (REQ-VAL-3).
- **Tier 2 — report-guided training** (teacher logit-KD, report-derived auxiliary targets). Promote only after an ablation win.
- **Deferred:** §10.

Why not bigger: ~4.4k studies, ≤58 trustworthy labels, noisy weak labels, 2 days of schedule → small backbone, frozen-first, no 3D transformers. `[H]` — never benchmarked here; the ablation table will decide.

## 5. Data pipeline — DICOM is part of the model

**REQ-DATA-1 Load.** `pydicom.dcmread`, decode via pylibjpeg/gdcm for compressed syntaxes, reject files without PixelData. Handle `PixelRepresentation` 0 and 1 (F7). *Verify:* `python tests/test_dicom_load.py` (to build: loads all 20 local series, asserts dtype/shape, signed pixels are not wrapped).

**REQ-DATA-2 Order.** Sort by `dot(ImagePositionPatient, normal)`, `normal = cross(row, col)` from IOP; fallback InstanceNumber → SliceLocation, logging which. **Then fix direction canonically** (e.g. flip so the projection increases along a fixed patient axis per plane) because the raw sign varies (F8). Never sort by filename. *Verify:* `python tests/test_slice_order.py` (to build: shuffled slice list ⇒ identical output order; direction identical across series of one plane).

**REQ-DATA-3 Geometry.** Resample in-plane to a **fixed physical FOV** (default 160 mm → 224 px; `[cfg: data.fov_mm]`, `[H]` vs plain resize — FOV is 150–180 mm so the difference is ≤ 12% scale) with centre handling; do not interpolate along z beyond what MIL needs. Variable slice count is supported by MIL with a mask; optionally resample to K slices `[cfg: data.slices]` (old default 24) for batching. Record spacing/thickness/gap per series. *Verify:* `python tests/test_geometry.py` (to build).

**REQ-DATA-4 Intensity.** Per-series percentile clip [0.5, 99.5] → [0,1] → replicate to 3 channels with ImageNet mean/std. No cross-series normalisation (p99 differs 10× between series, F7).

**REQ-DATA-5 Manifest.** One CSV row per series: study, series, n_slices, plane, contrast, vendor, tesla, rows, cols, pixel spacing, FOV, thickness, gap, PixelRepresentation, RescaleSlope, `PatientID` hash, IOP normal sign. Produced by `scripts/scan_dicom_manifest.py` (to build, headers only, HPC, resumable). Feeds site/patient grouping (REQ-VAL-1) and the site-shift analysis. *Verify:* `python scripts/scan_dicom_manifest.py --data $RSNA_DATA --limit 20 --out /tmp/m.csv && python tests/test_manifest.py /tmp/m.csv`.

**REQ-DATA-6 (optional) npz parity.** If the 24×224×224 uint8 volumes exist on HPC, accept them only if `preprocess(dicom)` reproduces them (SSIM > 0.99 on ≥ 20 series). Otherwise ignore them. Test-time path is always DICOM→REQ-DATA-1…4.

**REQ-DATA-7 Series selection.** ≤ `s_max=8` series per study (≥1 per plane, fluid-sensitive first, then longest); deterministic at inference, random at train. Padded series masked everywhere.

**REQ-DATA-8 Augmentation (train).** Same geometric transform for all slices of a series; scale 0.85–1, rot ±10°, brightness/contrast ±0.15, noise σ ≤ 0.02, slice-drop 0.1, series-drop 0.15 (never all). **No flips** (medial/lateral meaning). `[cfg: data.aug]`

## 6. Weak labels from reports (gates all training)

**REQ-NLP-1 Extractor.** Output per study `e ∈ {+1, 0, −1}^12` (present / absent / not-mentioned-or-uncertain→masked) plus confidence. Candidates, to be compared on the same protocol:
 (a) per-language lexicon + negation scope; (b) **instruction-tuned multilingual LLM run offline on HPC with a fixed JSON schema** `[H]` — raised from "later" in v0.1 because ~a dozen languages including non-Latin scripts (F5) makes per-language lexicons expensive. Reports are training-only, so an LLM at train time does not violate REQ-LEAK-1; whether external pretrained weights are allowed is `[U]` (REQ-RULES-1; human D-2); (c) XLM-R fine-tuned on (b)'s outputs. Machine translation to English is not used (extra noise stage).

**REQ-NLP-2 Language ID.** Log language per study; unsupported language ⇒ all slots masked, never guessed.

**REQ-NLP-3 Blind-then-score (resolves I-03).** Build/tune the extractor using only non-gold reports and written definitions. Freeze (commit hash). Score once on gold-58: per-label precision/recall with Wilson CIs. Gold is then eval-only for the student. A label slot is used as weak supervision only if gold precision ≥ `[cfg: weak.min_precision]=0.9` with ≥ 5 positives; otherwise masked. Any later re-tune is logged and invalidates gold as clean eval.

**REQ-NLP-4 Richer report targets `[H]`.** The report contains laterality (medial/lateral), location, severity, chronicity. Extract these as auxiliary targets (e.g. "meniscus tear: horn/segment", "OA grade") — more bits per study than 12 binaries. Ablate (A6b). Not in Tier 0.

*Verify (all NLP):* `python -m oxalpha.validation.score_extractor --gold` exits 0 only if frozen-hash matches and each used slot meets the precision rule (to build).

## 7. Validation (leakage-free)

**REQ-VAL-1 Unit of split = patient/study.** Group by `PatientID` hash from REQ-DATA-5 `[U whether duplicates exist]`; additionally merge studies with near-identical reports (MinHash, `[cfg: dedup]`). Series/slices never split. Fold assignment is computed once, written to `folds.csv`, frozen and reused by every experiment.

**REQ-VAL-2 Stratification.** 5 folds; iterative stratification on weak labels where trusted; gold-58 spread evenly (~11–12 per fold) as a separate stratum.

**REQ-VAL-3 Reporting.** Always two numbers per run: *weak-OOF* macro AUC and *gold-OOF* macro AUC (pooled over folds, bootstrap 95% CI, I-02). Weak-OOF is optimistic when extractor and student agree on report style. Promotion rule: keep a component only if it improves both (or one improves and the other is inside CI) at ≥ 2 seeds.

*Verify:* `python tests/test_split_leakage.py folds.csv` — no study, patient or dup-cluster appears in two folds (to build).

## 8. Model (student is the deliverable)

Backbone choice is a hypothesis; the pipeline must work with each below.

- **REQ-MODEL-1 Slice encoder.** 2.5D (slices k−1,k,k+1 as 3 channels), `tf_efficientnet_b0` (timm) 5.3 M params, frozen in Tier 0, partially frozen in Tier 1. ConvNeXt-T deferred. `[H]` 2.5D beats 2D on this data.
- **REQ-MODEL-2 Slice aggregation.** Gated-attention MIL [Ilse 2018] `[E]`, baselines mean/max pooling kept as controls. Per-group heads (ligament, meniscus, cartilage, joint, popliteal, bone) `[H]`, ablated.
- **REQ-MODEL-3 Series fusion.** 2-layer transformer over series tokens, CLS + attention pooling `[H]`; controls: mean over series, max over series.
- **REQ-MODEL-4 Metadata.** Plane embedding (3) + contrast embedding (2) added to series token; `[H]` ablation A4.
- **REQ-MODEL-5 Head.** 12 sigmoid outputs, per-label bias initialised to logit(prior), 64-d bottleneck `[H]`.
- **REQ-LEAK-1.** `Student.forward` accepts only images and metadata. *Verify:* `python tests/test_leak.py` — inspects the signature and parameter names for no text path (to build).
- **REQ-TEACH-1 (Tier 2).** Teacher = student trunk + report embedding with modality dropout (p=0.3); offline logit KD, τ=3 [Hinton 2015] `[E]`. Caution: weak labels are `f(report)`, so a teacher that reads the report reproduces label noise; KD only helps if it transfers *soft* information the hard labels lack. Ablation A7 vs A6 decides; if flat, drop the teacher.
- **REQ-SHAPE-1.** Forward shapes on a CPU dummy batch `B×S×N×3×H×W → B×12`, padded series fully masked. *Verify:* `python tests/test_shapes.py` (to build).

### Loss
`L = L_asl(masked, weighted) [+ λ_kd L_kd] [+ λ_aux L_aux]`.
- ASL γ−=4, γ+=0, clip 0.05 [Ridnik 2021] `[E]`; control = BCE with clipped pos_weight. Masked slots (`−1`) contribute 0; sample weight gold 1.0, weak `0.5×confidence` `[H]`.
- `L_aux`: plane and contrast classification from slice features (labels exist in `train_series.csv`) `[H]`.
- Deferred losses: InfoNCE image–text, feature-KD, attention-KD (§10).

## 9. Training, inference, efficiency

- AdamW wd 0.05, lr backbone 1e-4 / heads 5e-4, cosine with 2-epoch warm-up, fp16 AMP, grad accumulation to ~16 studies, EMA 0.999, grad-clip 1.0. `[cfg: train.*]` — starting values, not tuned.
- Tier 0 caches features once; training then takes minutes on CPU-class hardware, which is why it is first.
- Config hash + git commit stored with every checkpoint.
- **Inference** is student-only; ensemble by averaging logits (rank-average if the metric turns out to be pooled AUC). TTA without flips.
- **Efficiency award [P]:** criteria are unpublished (`[U]`). Log params, FLOPs/study, and seconds/study on a fixed GPU for every submitted model so the choice is available later (REQ-EFF-1: `scripts/profile_student.py`, to build).
- **REQ-SUB-1.** `submission.csv` has exactly the columns of `sample_submission.csv`, probabilities in [0,1], one row per test study. *Verify:* `python tests/test_submission.py submission.csv` (to build).

## 10. Experiments and deferred items

| ID | Change | Question |
|---|---|---|
| A0 | frozen B0 + mean pool + logistic | floor (= Tier 0) |
| A1 | max pool / gated MIL | MIL value |
| A2 | + group heads | group-head value |
| A3 | + series transformer | series fusion value |
| A4 | + plane/contrast tokens | metadata value |
| A5 | end-to-end | fine-tune value |
| A6 | weak+gold vs gold-only | weak supervision value |
| A6b | + richer report targets (REQ-NLP-4) | more bits per report |
| A7 | + teacher logit-KD | LUPI value |
| A8 | FOV-resample vs plain resize | REQ-DATA-3 |

Deferred until a strong baseline exists: ConvNeXt-T, 288 px, label-graph head, image-text InfoNCE (+ memory queue), feature/attention KD, plane-specific models, 3D/SSL encoders, test-time sliding windows.

## 11. Literature

Established `[E]` (primary papers; confirm exact bibliographic data before citing): Hinton, Vinyals, Dean 2015 *Distilling the knowledge in a neural network* (arXiv 1503.02531); Ilse, Tomczak, Welling 2018 *Attention-based deep MIL* (ICML); Vapnik & Vashist 2009 *LUPI*; Lopez-Paz et al. 2016 *Unifying distillation and privileged information*; Ridnik et al. 2021 *Asymmetric loss*; Radford et al. 2021 *CLIP*; Conneau et al. 2020 *XLM-R*; Sechidis et al. 2011 *iterative stratification*; Tan & Le 2019 *EfficientNet*; Bien et al. 2018 *MRNet*.

2024–2026 items already in `Readings.txt` (titles only, **not yet read; no claim in this spec depends on them** until T-LIT-1): Ahmad et al. 2024 multi-teacher cross-modal distillation; MICCAI 2025 multi-modal knowledge-decomposition online distillation; 2026 collaborative encoding + hybrid-attention KD; Harvey et al. 2026 multi-dataset MIL benchmark for 3D neuroimaging. Adaptation notes will be added per paper when read (what is changed for study→series→slice bags with a missing text view at test).

## 12. Open items needing a human (also in TODO "Decisions")

D-1 Accept Tier 0 as primary given the 10/22 deadline? D-2 Is external pretrained weights (LLM, timm) allowed, and available on Kaggle offline? D-3 Rewrite git history / make repo private for the 79 committed DICOMs? D-4 Why 4,407 vs ">5,000"? D-5 Confirm metric on Kaggle.
