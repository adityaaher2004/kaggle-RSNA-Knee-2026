# QuartzKnee-HMIL — Authoritative implementation SPEC

**Version:** 1.0.0 | **Date:** 2026-10-21 | **Target branch/worktree:** `quartz` | **Status:** design complete; implementation NOT claimed | **Companion:** `TODO.md`, `STRUCTURE.md`, `RULES_AUDIT.md`

> This is the proposed source of truth for new work on `quartz`, not a claim that the local repository has been modified. Preserve upstream source history; migrate and verify existing working modules rather than overwriting them blindly.

## 0. Authority, provenance, and development discipline

- `[P]` primary published source, `[A]` earlier agent's local audit (NOT independently reproduced on Quartz), `[E]` established published method, `[H]` hypothesis awaiting ablation, `[U]` unknown.
- **Source-of-truth precedence:** (1) official competition rules/data; (2) actual Quartz data and machine inspection; (3) this spec's decisions; (4) legacy `SPEC(2).md`, `TODO(2).md`, `tree.txt`. In a conflict, document the change before editing implementation.
- **Spec-anchored:** requirement IDs `Q-*` are cited from implementations/tests. Shape, algorithm, leakage protocol and result-contract changes begin as SPEC + `specs/contracts.yaml` revisions.
- **Spec-as-source (enforceable subset):** `specs/contracts.yaml` is machine-readable authoritative input to `scripts/spec_codegen.py`; it emits label constants/order, legal metadata enum, input/output TypedDict or dataclasses, JSON schemas and required-config defaults into `src/oxalpha/generated/`. Running `python scripts/spec_codegen.py --check` must fail if generated code is stale. `python scripts/spec_lint.py` cross-checks every `Q-*` against `TODO.md` and tests; CI runs both. Changing prose alone cannot magically implement an algorithm: structural requirements deliberately fail tests until code is updated. **Never claim autonomous code updates not supported by the generator.**
- Read/write configs through the verified pre-existing `oxalpha.config.load_config` and validated `extends`. Numeric hyperparameters are in `configs/*.yaml`, not embedded in logic. Checkpoint provenance stores spec, config, data, labels, folds and git hashes.
- [A] Prior draft says only `config.py`, EDA and two tests function; other package files may be import-time `NotImplementedError`. Verify on worktree first, don't assume implemented modules.
- [A] Prior hard rule **G-1 remains active**: 58 originally labeled studies are excluded from **ALL** student/teacher training, extractor tuning, pseudo-label generation for student, prior initialization and train folds. They may be scored as a **selection-contaminated development audit**, not as an untouched estimate. Changing G-1 requires an explicit owner-approved protocol revision.

## 1. Rules, data and objective

### Q-RULE-01 — competition contract [P]
- Official metric is the arithmetic mean of ROC-AUC over all **12** targets. Every competition metric report must return all twelve per-label AUCs and coverage/class counts; local single-class labels are `undefined`, not replaced with `0.5` and not silently averaged.
- Notebook-based submission; **GPU or CPU <= 9 hours**, internet off; freely/publicly available external pretrained models allowed (licence and access still require verification); output **`submission.csv`**. Final submission deadline **2026-10-22 23:59 UTC** (2026-10-22 19:59 EDT). Entrant must already have accepted rules by 2026-10-15.
- Official **efficiency score**: `AUC/(Benchmark - maxAUC) + RuntimeSeconds/32400`, minimized; private-board benchmark and maxAUC are unknown. Optimize performance and latency as a Pareto decision, not a guessed scalar.
- Winning team obligations and definitive interpretation of restricted-license external datasets require the signed-in Rules/host posts. Public host posts have rejected some agreement-gated datasets; no unverified external image corpus is included by default. Do not assume general research-use datasets automatically qualify.
- Published official input: `test.csv` contains `StudyInstanceUID` only; `test_series.csv` has acquisition descriptors; test DICOM series replace the three example studies under `test_series/`. There is **no test Report**. Test fluid-sensitive and fat-suppression flags are **independent, even if training audit found equality**.

**Official sources:**
- https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/evaluation
- https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data
- https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules
- https://www.rsna.org/news/2026/august/ai-challenge-knee-mri

### Q-DATA-01 — independently verify dataset [A][U]
- Prior audit: **4,407** train studies, **24,371** train series, **58** fully labeled, **4,349** with missing numeric labels; reports on all studies. 3–14 series/study (median 5), all three planes observed. The audit had **no all-negative** labeled study and an enriched gold subset; do not infer population prevalence from it.
- Local mounted image sample according to prior audit: 3 train studies/1 test study, not a representative training dataset. Real data ~569.76 GB/819,640 files, ~1,300 held-out studies per Kaggle public data page. Full counts MUST be rechecked on Quartz using `RSNA_EXPECT=full`.
- [A] 177 exact report duplicates, multilingual reports, varied vendors/physical spacings, signed pixels, opposite filename/IPP anatomical order signs; [U] duplicate PatientID groups across the whole collection.
- Data contracts: class names and order **exactly** `ACL`, `MCL`, `Medial Meniscus`, `Lateral Meniscus`, `Medial OA`, `Lateral OA`, `PF OA`, `Effusion`, `Synovitis`, `Baker's`, `Contusion`, `Fracture`. Labels are `{1,0,missing}`; missing NEVER means zero.
- Preserve unique `(StudyInstanceUID,SeriesInstanceUID)` and tie each `SOPInstanceUID` to its series. Validate train/test metadata schemas independently.

## 2. Model: **ONE final inference architecture** — QuartzKnee-HMIL [H]

### Q-ARCH-01 — architectural decision
A single shared **ImageNet-pretrained ConvNeXt-Tiny** slice encoder (`timm`, `convnext_tiny.fb_in22k_ft_in1k` *if and only if the exact checkpoint identifier exists offline and has appropriate rights; otherwise use an explicitly pinned available ConvNeXt-Tiny variant*), with **2.5D neighboring-slice input**, **label-query hierarchical MIL**, **compact cross-series transformer**, and **twelve study-level output logits**. **No image-text concatenation at inference.** Configurable EfficientNet-B0 *debug/fallback encoder only*, not a distinct final architecture. Output parameters/cost measured, not promised. Public 2026 competitor-reported ~0.95 AUC for a ConvNeXt-Small design is motivation, **not validation of this design**.

**Hierarchy:**

```
Study i
 ├─ Series j (up to 8; Sag / Cor / Ax; fluid, fat, geometry)
 │   ├─ ordered slice k (24 sampled; variable count and mask supported)
 │   └─ 2.5D image[k-1,k,k+1] -> shared ConvNeXt-T -> d=256
 ├─ label-conditioned gated slice-MIL -> per-(series,label) tokens
 ├─ metadata embeddings + 1-layer, 4-head, shared series context transformer
 ├─ label-conditioned gated series-MIL -> one embedding per label
 └─ 12 separate logits -> 12 sigmoid probabilities
```

The choice is appropriate to a few thousand studies, variable sequence counts, sparse/brittle textual labels, 3 anatomical planes, and mixed protocols. It allocates most compute to image features but allows flexible pooling. Full 3D ViT or Mamba, GNN of labels, cross-modal test encoder and VLM generation are **out of scope** until measured benefit.

### Q-ARCH-02 — tensor contract
- Training batch: `images [B,S,K,3,H,W] float32/float16`, `slice_mask [B,S,K] bool`, `series_mask [B,S] bool`, `plane [B,S] int{0,1,2,3=unknown}`, `fluid [B,S] int{0,1,2=unknown}`, `fat [B,S] int{0,1,2=unknown}`, `geometry_features [B,S,G] float` plus geometry-valid mask; `study_ids [B]` string for logging only. **No labels, reports, report embeddings or report-derived metadata keys appear in inference batch**.
- Labels `y [B,12]` and confidence `w [B,12]` are passed separately to the loss, NEVER forwarded into `Student.forward`.
- Defaults: `S=8`, `K=24`, `H=W=288`, `d=256`; select at least one series from each plane when available; favor fluid-sensitive and quality, with metadata diversity; do not use `y` or reports for series selection. For very long series use stratified whole-volume slice sampling and evaluate multiple interleaved grids if resource permits. Pad masks are strict. S and K configurable for V100.
- `slice_features F_theta(images_valid) -> [n_valid, C]`, linear projection `-> [B,S,K,d]` with zero-filled masked slots, never pass a zero pad as a valid slice.
- Safe implementations must support `B=1`, `S=1`, `K=1`, all-but-one series invalid and missing individual plane/contrast metadata; **all series invalid is an explicit recoverable study error**, never all-masked softmax NaNs.

### Q-ARCH-03 — mathematical aggregation
For label `l`, slice feature `h_{ijk} in R^d` and trainable query `q_l in R^d`:

`a_{ijkl} = q_l^T (tanh(W_t h_{ijk}) ⊙ sigmoid(W_g h_{ijk})) / sqrt(d)`;

`alpha_{ijkl} = masked_softmax_k(a_{ijkl}, slice_mask_ijk)`;

`z_{ijl} = Σ_k alpha_{ijkl} h_{ijk}`.

Compute one shared series summary `r_ij = masked_mean_k(h_ijk)`; add `plane/fluid/fat` embeddings, normalized geometry indicators and missing-metadata embeddings; feed through a **single transformer encoder block over the S series** with a key-padding mask to obtain `c_ij`. Then:

`t_ijl = LayerNorm(z_ijl + c_ij)`;

`beta_ijl = masked_softmax_j(q_l^T W_s t_ijl / sqrt(d), series_mask_ij)`;

`v_il = Σ_j beta_ijl t_ijl`;

`logit_il = w_l^T v_il + b_l`, `p_il = sigmoid(logit_il)`.

Use shared projections and tiny label queries, not twelve independent ConvNeXts. Mean/max MIL pooling is a mandatory ablation control. Attention is **a prioritization signal, not proof of localization**.

### Q-ARCH-04 — initialization and controls
- Pretrained weights are a runtime dependency for training, attached offline for inference; record exact model name, hash, originating URL and licence.
- New head initialized with zero/weak bias or non-gold **training-only** reliable prior (never gold). Backbone initially frozen; head learns on cached features. Unfreeze last stage only after stable loss/OOF and GPU profiling. Avoid batchnorm-statistics drift in tiny batches.
- Anatomically informed label queries are learned, **not** given fixed plane-to-label gates. Clinical hypotheses (e.g. coronal useful for MCL, sagittal useful for ACL, fluid sensitive useful for effusion) are only hypotheses; no hard-exclusion of other sequences.

## 3. DICOM processing (MUST be same in training and hidden inference)

### Q-IMG-01 — robust DICOM decode
- Use `pydicom` with pinned offline JPEG Lossless and JPEG 2000 decoders (`pylibjpeg-*` or compatible packaged alternatives); decode signed/unsigned PixelRepresentation correctly, per-slice RescaleSlope + RescaleIntercept before statistics, stored PixelPaddingValue, and flag corrupted files without crashing entire notebook. Some public reports document truncated DICOM slices; test explicit synthetic corrupt files.
- `stop_before_pixels=True` metadata scan creates resumable series-level manifest. Never assume filenames alphabetical equals anatomical order; use IPP projected onto `cross(IOP[0:3],IOP[3:6])`, sort by scalar position, stable tie-break. Fallback InstanceNumber/ SliceLocation with `ordering_unreliable` flag.
- Detect reversed normals, oblique orientation, duplicate positions, irregular z-gaps, inconsistent matrix/spacing/orientation, left/right/unknown. Do not silently discard an entire study because one series is corrupt. Record quality flags.

### Q-IMG-02 — intensity and geometry
- Per-series robust percentiles (default 0.5 and 99.5) from **nonpadding voxels**; clip and map to [0,1]. Store valid-background mask; constant/invalid series yields flagged failure and is excluded. No report-driven preprocessing.
- Canonical in-plane orientation only when geometry makes it reliable. Patient-frame directions must not silently invert medial/lateral semantics, especially for opposite knee laterality. No random left/right flips or mirroring by default. **Test-time mirroring is an optional ablation only with validated anatomical transforms.**
- **Primary resize:** preserve entire anatomy/FOV, aspect-preserving in-plane letterbox or carefully justified resize to 288; padding value 0, masks propagated. This replaces the earlier unvalidated 160 mm center-crop that could remove peripheral Baker cysts. Capture pixel spacing/FOV as metadata; **physical-FOV normalization or multiscale local crops are separate ablations**.
- Sample K uniformly by ordered physical position (accounting for nonuniform gaps) with deterministic endpoints; pad short series with false masks; preserve 2.5D nearest neighbors only within same valid series. Apply ImageNet mean/std after making the three channels.
- Consistent series-wide geometric augmentation: modest rotation ±10°, zoom ~0.9–1.1, low intensity jitter/noise, optional slice/series dropout while retaining at least one valid series per observed plane if possible; augmentation controlled by seeded RNG. Do not apply augmentation during deterministic inference.

### Q-IMG-03 — series selection
- Select up to 8 complete series deterministically without labels/reports. At least one each of sagittal, coronal, axial if present; prefer diagnostic quality, sufficient coverage, and two independent contrast flags; select remaining by diversity and slice coverage. If >8, record excluded series. If fewer, pad. During training, optional random selection constrained by required planes; at inference, deterministic only.
- The training audit observed `Fluid_Sensitive==Fat_Suppression` in all train series, **but test documentation explicitly says not guaranteed**. Keep separate embeddings with unknown states and an out-of-distribution metadata combo test.

### Q-IMG-04 — cache and parity
- Cache generated image tensors or frozen-backbone features on Quartz scratch with `preprocess_version + source header/content hash + selected slice indices + pretrained backbone hash` in cache manifest. Sharded NPZ/LMDB/Zarr only after measuring throughput. **Never** assume old `processed-3d-volumes/*.npz` exists or is comparable; parity-test before using.
- Exactly the same `preprocess_study()` library must run on Quartz and Kaggle. A test comparing serialized reference tensors to offline Kaggle reconstruction with tight tolerance is mandatory.

## 4. Report labels (Muse Glimmer) as privileged TRAINING data

### Q-LABEL-01 — ingest a completed extractor, do not restart it
- The user's local **Muse Glimmer** extraction is the first intended label producer. Import final outputs, don't train another LLM/XLM-R as a gate for basic training. Require `StudyInstanceUID`, exact 12 labels as `{0,1,null}`, per-cell `confidence` (if available), `status` (`positive|negative|uncertain|not_mentioned|not_processed` where supplied), `extractor_name/version`, `prompt_hash`, `report_sha256`; per-label evidence spans optional and **never checked into git**. If Glimmer provides no confidence or status, generate `confidence=1` only for explicit definitive 0/1 and document that it is uncalibrated, not a probability of correctness.
- Preserve all raw outputs in excluded scratch; export sanitized `weak_labels.parquet` plus `label_manifest.json` atomically on completed extraction. Require 4,349 non-gold rows (subject to full-data verification) and left join by UID; duplicate/conflicting IDs are hard errors; never overwrite original `train.csv`.
- **Semantics:** distinguish absence of mention from explicitly negated findings; don't treat uncertain/possible or outside-definition labels as definite negative. Avoid inferencing `Synovitis=1` from `Effusion=1` without a specified validated mapping. Use host label definitions, multilingual evidence and severity thresholds as given in `docs/label_definitions.md`.

### Q-LABEL-02 — QA without wrecking coverage
- Before GPU training, emit per-class `{n_positive,n_negative,n_ambiguous,n_notmentioned,n_missing}`, language distribution, label overlap, all-negative fraction and spot-check cases. Compare predictions on gold-58 as **development audit**, reporting positive/negative precision, recall where defined and Wilson CIs, coverage. Gold-58 has low support and enriched prevalence; it cannot certify every label/polarity at stringent sample sizes.
- Prior spec mandated >=20 predicted calls AND Wilson 95% lower bound >=0.8 for **every** label+polarity. **Remove this as a global blocking gate:** for small gold this can mask essentially all supervision and preclude training. Instead use deterministic confidence-weighting plus label-specific audit warnings/optional abstention mask; persist `label_quality_report.json`; predeclare any thresholds before inspecting model results. Weak labels are *weak*, not automatically validated truth.
- A separate expert-adjudicated set of non-gold reports, if obtainable, should be held out from training and used to measure extractor error by language/class. Do not use gold to repeatedly rewrite prompts and claim untouched gold validity.
- If weak labels are not ready, train smoke pipeline on synthetic labels ONLY to prove IO; **never pretend that synthetic-label output is a competitive trained model**.

### Q-LABEL-03 — masked, confidence-aware objective
For explicit observed targets only (`m_il = 1` for trusted 0 or 1):

`L_BCE = [Σ_i,l m_il w_il BCEWithLogits(z_il,y_il)] / max(1,Σ_i,l m_il w_il)`.

Start with **masked weighted BCE**, which is predictable under label noise. Compare ASL (gamma-=4, gamma+=0, clip=0.05) as a **controlled experiment**, not a mandatory step (ASL hard-negative suppression may accidentally suppress useful signals when negatives are noisy). `w_il` is non-gold extractor confidence (if meaningful), clipped into a documented interval; default 1 for definitive labels, 0 for null/uncertain. Empty-supervision mini-batch must skip optimizer step rather than NaN.
- To protect rare labels' macro-AUC, log per-label gradient/coverage; optional clipped per-label pos weight calculated from training fold only, never audit/test. Rank-based per-label validation decides weights.

## 5. LUPI, teacher and distillation — explicitly optional after viable MRI baseline

### Q-LUPI-01 — report teacher [H]
- Train optional report encoder (frozen multilingual text encoder + small text label head, or frozen report embeddings) strictly on **non-gold reports** using Glimmer targets; report-only teacher learns weak-label mapping but may merely amplify the same extraction errors. Better optional teacher uses MRI features + report embeddings, not a text-only oracle with identical targets; both are train-time only.
- After the image-only student converges, try teacher-to-student **per-label logit KD**: `L = L_BCE + lambda_KD*T^2*BCEWithLogits(z_student/T, sigmoid(z_teacher/T))`, with **teacher targets produced out-of-fold** for each training study to avoid in-sample memorization; missing/low-confidence classes masked. `T=2` and `lambda_KD=0.1` are **trial defaults**. For multilabel problems, use Bernoulli soft-target BCE, not 12-way softmax KL. No report encoder is bundled into inference package.
- Hard safeguard: a report teacher that directly encodes labels derived from the same report can trivially fit textual artifacts; only retain KD if paired OOF image AUC improves without suspicious dependence on text/report labels. No text-image InfoNCE, feature KD, attention KD or modality dropout until an actual gain is demonstrated.

### Q-LUPI-02 — inference-independent tests
- `Student.forward()` and inference signature accept *images, masks and test-available acquisition metadata only*. Train data exposes targets only in a separate training record; a canary Report does not change inference outputs; importing `oxalpha.inference` must not import transformers, text encoder or teacher modules.
- Every fold trainer persists IDs actually seen, and asserts disjointness from validation and gold IDs, patients, and report-near-duplicate clusters. No information from gold is permitted to initialize model bias, compute weights, or generate training supervision.

## 6. Validation, metrics and decision protocol

### Q-VAL-01 — grouping and folds
- Full manifest-derived SHA256 PatientID groups (if available), duplicated study detection, report text exact/near duplicate clusters, and study grouping. Entire clusters stay in same fold; no slice/series split. If PatientID missing, group by StudyInstanceUID and record limitation.
- **5-fold fixed groups** over only the 4,349 weak-labeled, non-gold studies is the desired research protocol; with the imminent deadline, prioritize **one reliable grouped 80:20 holdout** to select a deployable model, then train full non-gold model. Do not claim a 5-fold result if it was not executed. Freeze seed and `folds.csv` before first GPU fit.
- Multi-label balancing approximate/iterative only where labels observed; pass imputed values to stratifier only for stratification with a missing-mask and report when a class is single-class in validation. Never use gold in stratification.

### Q-VAL-02 — evaluation
- `macro_auc` = mean of twelve valid per-label scores ONLY when all 12 have positive and negative classes. Otherwise report partial-macro with explicit valid-label count and **never compare as though it were official macro**. Report per-class n_pos, n_neg, ROC-AUC, PR-AUC and grouped study bootstrap CIs (limited gold makes CIs extremely broad).
- Two distinct readouts: (1) **weak OOF** against Glimmer labels, optimistic/noisy; (2) **gold development audit** of 58 studies, selection-contaminated, small and enriched. Neither is an unbiased estimate of private test AUC; public leaderboard is also subject to overfitting.
- Paired bootstrap ΔAUC on identical studies is preferred for local ablations, report CI and coverage; predeclare improvement/latency thresholds; change selection protocol only with owner authorization if it modifies the previous policy.

### Q-VAL-03 — ablation order (not a giant grid)
1. Baseline frozen ConvNeXt-T with masked mean/max pooling vs label-query gated MIL (Q-ARCH-03).
2. Add separate plane/fluid/fat embeddings and one-layer series transformer.
3. Head-only vs partial unfreeze of ConvNeXt final stage.
4. Masked BCE vs ASL, with exactly same folds/labels.
5. 2D-repeated vs 2.5D input; full-FOV 224 vs 288/physical-crop variant if time.
6. Optional out-of-fold report KD **only after A1–A4 are working**.
Each keeps number of training studies, label version, folds and inference budget fixed where practical. Chosen model must provide AUC improvement **and** acceptable inference latency, or favor simpler.

## 7. Quartz training, compute and packaging

### Q-RUN-01 — runtime profiles
- Initial config: `S=8, K=24, image_size=288, D=256, heads=4, transformer_layers=1, AdamW, weight_decay=0.05, lr_head=5e-4, lr_backbone=3e-5, 2 epoch warmup, cosine 10–20 epochs, grad_clip=1, seed=42, study effective batch ~8`, all **trial defaults, subject to profiling**. Mixed precision **bf16 only on supporting H100; fp16+GradScaler for V100**. EMA optional. Do not assume fixed microbatch: profile with real study bag and log peak VRAM.
- **V100 path:** initial frozen ConvNeXt features cached with 224/256 input, then train attention/head on extracted 256-d embeddings; fine-tune late backbone only with microbatch 1 + gradient accumulation, slice microchunks, activation checkpointing if required.
- **H100 path:** enable 288 and late-stage fine-tuning once memory/time measured. Always keep one common checkpoint/inference format; test H100-to-Kaggle FP16 parity with tolerances.
- Slurm: jobs use `sbatch` scripts with explicit `--partition`, `--gres=gpu:1`, `--time`, `--cpus-per-task`, `--mem`, scratch from environment; **actual Quartz allocation, GPU memory, partition, account, site mounts unknown until probed**. Do not hard-code IU-specific account/resource names. Training data never in git.
- Use deterministic setup where feasible, data loader worker seeding, resumeable cached stage, checkpoints `last`/`best_oof`; OOM progressive fallback (`S,K,H,W`, microbatch) recorded in experiment metadata, not silent degradation.

### Q-RUN-02 — result and model provenance
- Every experiment tracked as sanitized `results/<run_id>/{metrics.json,provenance.json}` with run status `provisional|final`, spec version, git commit/dirty, config hash, labels hash, folds hash, manifest hash, preprocessing version, backbone weights hash, hardware, package versions, best epoch, per-label results, inference time. **No reports, DICOM, gold per-study labels, study-level outputs, credentials or raw patient IDs in git.** Logs/checkpoints under gitignored `work/` or Quartz scratch.
- Feature caches and processed DICOMs must never be shared as public code artifacts without validating competition data terms. Previous agents reported tracked DICOMs in repository history; make remote private and separately review/purge historical files, *but do not rewrite remote history or push without owner approval*.

### Q-SUB-01 — exact Kaggle submission contract
- One thin `notebooks/kaggle_inference.ipynb` imports packaged **student-only** inference code and pretrained weights as attached Kaggle dataset/model. No package downloads, no train.csv, no report, no API tokens, no external site at inference.
- Resolve inputs from `/kaggle/input/` after checking actual mount names; load `test.csv`, `test_series.csv`, `test_series/`; study IDs and order come from test.csv; use streaming decode and bounded CPU RAM/VRAM, never pre-load entire hidden test. Warmup model once; per-series and per-study exception handling. **If all series for a study fail:** emit a versioned fallback non-gold training prior (or 0.5, predefined) and a sanitized failure log, without dropping the test row.
- Write `/kaggle/working/submission.csv` with **exact** column names/order in Kaggle sample submission, exactly one row per test UID, probabilities finite and in [0,1], no duplicate/missing study IDs, and zero `Report` accesses. Force release of decoded image arrays between studies. Package pinned offline decoder wheels only if missing in notebook image.
- A **visible 3-study notebook test is insufficient**. Replay on a stratified collection of train DICOM studies with corrupted/oblique/compressed/tall series to estimate robust hidden-runtime, then commit the actual Kaggle Notebook and check *hidden rerun status*. Final selection occurs before deadline, not merely writing a CSV on Quartz.

### Q-EFF-01 — explicit efficiency profiling
- Measure `sec/study` p50/p95, peak RAM, peak VRAM, decoder share, GPU utilization and model parameter/FLOP estimates on V100/H100 (Quartz) **and Kaggle GPU** when available. Time the full path including DICOM decode. Keep efficiency and high-accuracy submission variants only if both tested and within nine hours; do not guess the unknown private-test `maxAUC` term.

## 8. Required tests / definition of done

| ID | Acceptance criteria / command |
|---|---|
| Q-RULE-01 | `python scripts/check_rules_contract.py`: verified official rules frozen with URL; signed-in terms flagged if unchecked |
| Q-DATA-01 | `RSNA_EXPECT=full python -m pytest tests/test_data_contract.py -q` on Quartz, with counts updated after actual inventory |
| Q-IMG-01 | `pytest tests/test_dicom_decode.py tests/test_dicom_sort.py -q` including signed pixel/IPP reverse/compressed/corrupt fixtures |
| Q-IMG-02/03 | `pytest tests/test_preprocess.py tests/test_selection.py -q` incl. laterality/masks/test metadata mismatch |
| Q-ARCH-02/03 | `pytest tests/test_model_shapes.py tests/test_masking.py -q` incl. gradients, all-masked-safe errors |
| Q-LABEL-01/02 | `python scripts/ingest_glimmer.py --validate-only ...` and `pytest tests/test_weak_labels.py -q` |
| Q-LUPI-02 | `pytest tests/test_no_text_leak.py tests/test_split_leak.py -q` |
| Q-VAL-02 | `pytest tests/test_metrics.py tests/test_paired_auc.py -q` |
| Q-RUN-02 | `python scripts/check_results.py results/<run_id>` |
| Q-SUB-01 | `pytest tests/test_submission.py tests/test_offline_inference.py -q` + **actual successful Kaggle hidden rerun** |
| Spec-as-source | `python scripts/spec_codegen.py --check && python scripts/spec_lint.py && pytest -q` |

**Success gate:** Complete valid 12-output offline inference notebook, successful hidden rerun, one measured MRI-only model trained on non-gold Glimmer labels, auditable validation/provenance, reproducible `quartz` commits. **Not yet achieved by these design artifacts.**

## 9. Primary research and adaptation notes

| Research | Established finding relevant here | Quartz adaptation / status |
|---|---|---|
| Bien et al., *MRNet*, PLOS Medicine 2018, https://doi.org/10.1371/journal.pmed.1002699 | Knee MRI can be learned from 2D representations aggregated over acquired planes [E] | Treat **all series within each plane** as instances, generalize original plane evidence to 12 labels; not proof of superiority [H] |
| Ilse et al., *Attention-based Deep MIL*, ICML 2018, https://proceedings.mlr.press/v80/ilse18a.html | Trainable permutation-invariant attention for bags [E] | Nested **slice-within-series, series-within-study** class-query gates [H]; compare mean/max pooling |
| Ridnik et al., *Asymmetric Loss*, ICCV 2021, https://openaccess.thecvf.com/content/ICCV2021/html/Ridnik_Asymmetric_Loss_for_Multi-Label_Classification_ICCV_2021_paper.html | Handles positive/negative class imbalance [E] | Mask uncertain report slots and compare against BCE before enabling [H] |
| Windsor et al., *Vision-Language Modelling for Radiological Imaging and Reports in the Low Data Regime*, MIDL 2024, https://proceedings.mlr.press/v227/windsor24a.html | Low-data domain adaptation and contrastive multimodal training explored [E] | Reports are privileged; **do not** add always-on image-text InfoNCE or text test path without ablation [H] |
| *Weakly supervised language models for automated extraction of critical findings from radiology reports*, npj Digital Medicine 2025, https://www.nature.com/articles/s41746-025-01522-4 | LLM-derived report pseudo-labels and validation are viable methodologies in a different radiology task [E] | Contract for locally generated Glimmer labels; uncertainty masking + per-class QA [H] |
| Wang et al., *Multitask learning for automatic detection of meniscal injury on 3D knee MRI*, J Orthop Res 2025, https://doi.org/10.1002/jor.26024 | Multitask/attention and multi-plane knee MRI approach [E] | No segmentation masks here; adapt only multi-plane and label-specific pooling, not segmentation-supervision claims [H] |
| Chang et al., *Enhancing Medical Large Vision-Language Models via Alignment Distillation*, AAAI 2026, https://doi.org/10.1609/aaai.v40i24.39079 | Spatial/attention alignment for VLMs in other medical tasks [E] | Explicitly **defer** spatial/attention distillation, because neither medical VLM task nor data regime matches this competition |
| *Multimodal foundation models exploit text to make medical image predictions*, Nature Communications 2026, https://www.nature.com/articles/s41467-026-74207-5 | Multimodal diagnostic benchmarks show text shortcuts can dominate visual interpretation [E] | Motivates **strict text-free inference** and skepticism toward report teacher [H] |

2024–26 ideas are context, not independently validated components of QuartzKnee. Cite actual clinical/imaging experiments when findings transfer; don't describe a novel mechanism as established.

## 10. Decision register and critical blockers

- **B-01 ENTRY:** Has competition participation/rules acceptance already been completed? Deadline for new entrants passed October 15. No account access here to verify.
- **B-02 ACCESS:** Actual IU Quartz V100/H100 memory, scheduler partition/account, dataset mounting and local labels complete? Unknown; code must inspect without embedding machine-specific assumptions.
- **B-03 LABELS:** Glimmer output schema, extraction completeness and reliability unknown until imported. This is critical path; no replacement LLM required by default.
- **B-04 DATA:** Prior 4,407/24,371/58 counts from other agent, not independently verified on full Quartz input.
- **B-05 RIGHTS:** Confirm specific weight licence and any external dataset rights; avoid agreement-gated data unless host explicitly allows it. Hosted LLM policy announcement exists; this project uses local Muse Glimmer so does not depend on third-party API.
- **B-06 GOLD:** Retain G-1 no training; no unbiased clean eval exists if gold used for gating and model selection. A separate sealed set would be preferable but infeasible without enough independent annotations.
- **B-07 REPO:** Existing worktree/remote not available in the uploaded files. **This package does not create or modify the actual `quartz` Git worktree.** No push, no remote changes.
- **B-08 KAGGLE:** Hidden rerun and actual Kaggle GPU type/mounts unverified; submission notebook must be measured and run successfully rather than assumed.
