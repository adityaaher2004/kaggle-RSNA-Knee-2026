# QuartzKnee-HMIL — Execution TODO

**Version 1.0.0 — 2026-10-21.** Each task has a SPEC anchor, role, artifact, and objective acceptance criterion. No task here is marked complete merely because this plan was written. `L` = local CPU/worktree, `Q` = IU Quartz V100/H100, `K` = Kaggle Notebook, `O` = owner. Priority: **P0 valid submission before the deadline**, P1 competitive training, P2 useful experiments, P3 optional research. Owner's prior commitment to a full pipeline is respected; P0 must not expand into a second separate architecture.

## Critical path (start immediately, parallel where possible)

- [ ] **Q-001 [P0, O]** Check Kaggle participation accepted before entry deadline (Oct 15). Check final date 2026-10-22 23:59 UTC and available Kaggle notebook GPU quota. *Accept:* participation confirmed and a Code Competition submission slot appears. `Q-RULE-01`.
- [ ] **Q-002 [P0, L]** Create/use `git worktree add ... quartz` **only if the real repo and branch are present**; do not overwrite `local` or `hpc`, don't push, no Git history rewrite. Inspect tracked data/secrets, `git status`, `git log`, `git branch -a`, installed skeleton. *Accept:* clean `quartz` HEAD and detailed diff baseline, `requirements` compatibility documented. `Q-RUN-02`.
- [ ] **Q-003 [P0, Q]** Inspect Quartz environment (`nvidia-smi`, `torch`, `sinfo`, writable scratch) and full dataset; run existing `tests/test_data_contract.py` with `RSNA_EXPECT=full`, `scripts/eda_local.py` on a representative sample. Update [A] findings in SPEC to verified [Q] if differences. *Accept:* filesystem mounts, GPU type/memory, full study/series counts, decoder availability recorded; no raw DICOM tracked. `Q-DATA-01/Q-RUN-01`.
- [ ] **Q-004 [P0, L/O]** Get completed Muse Glimmer output **when available** and freeze version; do not restart report labeling. Define exact JSON/CSV adapter and map to 12 labels. *Accept:* import validates 100% of encountered UIDs, recognized labels, and reports per-label coverage, without reading unknown as zero. `Q-LABEL-01`.
- [ ] **Q-005 [P0, L]** Freeze official column names/test data input semantics and write `RULES_AUDIT.md`; resolve rights of selected pretrained checkpoint and local label extraction. *Accept:* reproducible rules links, exact output schema, allowed offline assets. `Q-RULE-01`.

## Phase A — spec/code contracts and DICOM ingestion (parallel with Glimmer)

- [ ] **Q-010 [P0, L]** Add `specs/contracts.yaml` and `scripts/spec_codegen.py`, generated `src/oxalpha/generated/contracts.py`, config schema, and `--check` mode. Add `scripts/spec_lint.py` mapping `Q-*` to TODO/tests. *Accept:* `python scripts/spec_codegen.py --check && python scripts/spec_lint.py`. `§0`.
- [ ] **Q-011 [P0, L]** Preserve/verify `oxalpha.config.load_config`, stage `extends`, sensible debug defaults, `RSNA_DATA` env; add a single `quartz.yaml` with override-only V100/H100 configs. *Accept:* `pytest -q tests/test_config.py` with no machine-specific paths in source. `Q-RUN-01`.
- [ ] **Q-012 [P0, L]** Implement robust decoder in `data/dicom.py`: signed pixels, slope/intercept, padding, JPEG lossless/J2K offline, graceful per-slice failure. *Accept:* `pytest tests/test_dicom_decode.py -q`, plus available sample DICOMs; confirm decoder wheels bundled for Kaggle if necessary. `Q-IMG-01`.
- [ ] **Q-013 [P0, L]** Implement geometry ordering using IOP/IPP normal, duplicates, oblique flags, normal sign, fallback sort, irregular gaps. *Accept:* synthetic shuffled series are invariant to filename/InstanceNumber reversal; `pytest tests/test_dicom_sort.py -q`. `Q-IMG-01`.
- [ ] **Q-014 [P0, L]** Implement whole-series intensity normalization, anatomy-preserving in-plane resampling, masks, deterministic physical-position sampling, 2.5D channels and train-only augment. *Accept:* `pytest tests/test_preprocess.py -q` including `0` contrast, padding, and phantom with known pixel dimensions. `Q-IMG-02`.
- [ ] **Q-015 [P0, L]** Implement metadata-aware deterministic ≤8 series selection (3 planes, separate fluid and fat), overflow/underflow masks and label-blind train variation. *Accept:* `pytest tests/test_selection.py -q` with novel test combo `fluid=1,fat=0`, no labels/report passed. `Q-IMG-03`.
- [ ] **Q-016 [P0, Q]** Run header-only full train/test manifest scan; deduplicate and profile lengths/typical per-study decode time; list problem series and rare formats. *Accept:* resumable `work/manifest_{train,test}.parquet`, counted codec/failure rates, no raw patient info committed. `Q-IMG-01/Q-IMG-04`.
- [ ] **Q-017 [P1, Q]** Benchmark optional cache format (cached 2.5D uint8/fp16 tensors or ConvNeXt features) with preprocessing version/hash, checksum and source headers; reject legacy NPZ absent parity. *Accept:* reader correctness and read/decode throughput test, hit/miss logged. `Q-IMG-04`.

## Phase B — weak labels and leak-proof folds (parallel with Phase A)

- [ ] **Q-020 [P0, L]** Implement `labels/glimmer.py` schema adapter with atomic outputs and source/version hashes. Include ambiguous/negated/not-mentioned handling. *Accept:* `python scripts/ingest_glimmer.py --validate-only --input <local extraction file>`, `pytest tests/test_weak_labels.py -q`. `Q-LABEL-01`.
- [ ] **Q-021 [P0, L]** Complete `docs/label_definitions.md` from official description/host criteria; assert e.g. OA compartment, meniscus tear, injury vs tear, synovitis vs effusion not silently relabeled. *Accept:* all 12 written definitions and uncertainty policy signed off. `Q-LABEL-01`.
- [ ] **Q-022 [P0, L]** Produce QA dashboard/JSON: positive, negative, uncertain, missing, language, evidence disagreement; diagnostic comparisons on gold-58, **no gold re-tuning without recording selection contamination**. *Accept:* `work/label_quality_report.json`; classes with inadequate positives/negatives named (not hidden). `Q-LABEL-02`.
- [ ] **Q-023 [P0, L/Q]** Build patient/study/near-duplicate group graph and frozen grouped holdout (`folds.csv`), optional 5-fold assignment. Gold-58 excluded from all training/folds, verified against actual *iterated* train IDs. *Accept:* `pytest tests/test_split_leak.py -q` and persisted manifest/folds hashes. `Q-VAL-01/Q-LUPI-02`.
- [ ] **Q-024 [P0, L]** Implement macro/per-label AUC with missing masks and single-class explicit errors, weak OOF, gold *development audit* and paired grouped-bootstrap comparison. *Accept:* `pytest tests/test_metrics.py tests/test_paired_auc.py -q`. `Q-VAL-02`.

## Phase C — single competitive model and measured training

- [ ] **Q-030 [P0, L]** Implement `models/student.py`: test-available signature only; ConvNeXt-Tiny `timm` backbone and 2.5D channel handling, per-label gated slice MIL, shared series-context transformer, per-label gated series MIL, independent logits. *Accept:* `pytest tests/test_model_shapes.py tests/test_masking.py tests/test_no_text_leak.py -q`, including edge cases and finite gradients. `Q-ARCH-01/Q-ARCH-02/Q-ARCH-03/Q-ARCH-04`.
- [ ] **Q-031 [P0, L]** Add `losses/masked.py` masked weighted BCE; ASL optional experiment, no gradient for unknown label or zero confidence. *Accept:* synthetic perfect/no-label/rare-positive batch tests, `tests/test_masked_loss.py`. `Q-LABEL-03`.
- [ ] **Q-032 [P0, Q]** **A functioning high-priority train**: profile real bag VRAM; freeze image encoder, extract/cache per-slice features, train pooled head on non-gold Glimmer labels; save best model & OOF in sanitized results. *Accept:* non-trivial training loss decrease and valid 12-label holdout report. `Q-RUN-01`.
- [ ] **Q-033 [P1, Q]** Fine-tune last ConvNeXt stage with mixed precision and microbatch/gradient accum chosen from profile. V100 fallback: 224–256 px/16 K/6 S, and frozen cached backbone, but **change documented config**. H100: 288/24 K/8 S as budget permits. *Accept:* finite train+val for ≥1 epoch, resumable best checkpoint, recorded peak VRAM. `Q-RUN-01`.
- [ ] **Q-034 [P1, Q]** Run minimal controlled ablations: mean/max vs label-query MIL; metadata on/off; frozen vs partial unfreeze; if time, masked BCE vs ASL. Prefer one reliable fold + full-data train to speculative 5-fold ensembles before the deadline. *Accept:* paired per-label score + runtime table with model chosen explicitly. `Q-VAL-03/Q-EFF-01`.
- [ ] **Q-035 [P2, Q]** Optional 5-fold full CV and small ensemble after a successful single-model Kaggle dry run and only if compute budget permits. *Accept:* fold ID disjointness and <9h Kaggle runtime incl. decode. `Q-VAL-01/Q-SUB-01`.
- [ ] **Q-036 [P3, L/Q]** Optional frozen-report teacher, **OOF** soft predictions and Bernoulli logit KD ablation. **Never gate submission on this task.** *Accept:* no inference text dependencies and paired OOF improvement with acceptable latency; otherwise delete from final inference. `Q-LUPI-01`.

## Phase D — Kaggle export and successful hidden submission (do in parallel; do not postpone)

- [ ] **Q-040 [P0, L]** Implement `inference/predict.py` streaming DICOM→student; robust if missing series, long series, JPEG lossless/J2K, odd geometry, output dtype. Only MRI metadata at test. *Accept:* `pytest tests/test_offline_inference.py -q`, dummy checkpoint and provided 3-study Kaggle example run. `Q-SUB-01`.
- [ ] **Q-041 [P0, L]** Implement exact `submission.csv` schema/order validator with sample submission, UID set equality, no missing/non-finite/out-of-range values, no report references. *Accept:* `pytest tests/test_submission.py -q`, compare `test.csv` ids. `Q-SUB-01`.
- [ ] **Q-042 [P0, K]** Create a Kaggle **Notebook** that runs internet-off, attaches correct model/checkpoint and offline wheels if needed, reads hidden injected test paths and writes `/kaggle/working/submission.csv`. *Accept:* successful notebook Save & Run all cells on 3 visible examples, logs versions and elapsed time, `import oxalpha.inference` uses no HF/download path. `Q-SUB-01`.
- [ ] **Q-043 [P0, Q/K]** Stress test ≥100 train studies selected for codec/vendor/long series/corruption and time the complete 1,300-study extrapolation conservatively with margin <9h. Ensure any failed study receives valid predeclared fallback, fail log available for diagnosis. *Accept:* no uncaught exceptions, memory bounded; time projection documented with assumptions. `Q-EFF-01`.
- [ ] **Q-044 [P0, K/O]** **Commit notebook and submit actual competition version**, verify hidden rerun success, score and exact final selected version. Note that a locally produced CSV or successful three-example Save & Run is NOT submission validation. *Accept:* Kaggle submission status `COMPLETE` with no hidden traceback, recorded notebook version/checkpoint hash, before **2026-10-22 23:59 UTC**. `Q-SUB-01`.
- [ ] **Q-045 [P0, L]** Sanitize final `results/`, `git status`, manifests/checkpoint hashes, `spec_codegen --check`, test summary; handoff file with reproducible exact run commands, resume checkpoints and limitations. *Accept:* only code/spec/config/sanitized metrics tracked; no raw reports, DICOM, credentials or patient data. `Q-RUN-02`.

## Release gates

**Gate 0 — Imminent safety:** Q-001, 002, 003, 005, 010, 011. If Kaggle participation not accepted, no valid submission is possible despite software success.

**Gate 1 — Training:** Q-012–015, 020–024, 030–032. If Glimmer incomplete, only synthetic smoke tests may proceed; no false claim of competitive labels.

**Gate 2 — Kaggle-ready:** Q-040–044 are mandatory, including successful hidden rerun. This supersedes any tempting additional architecture experiments.

**Gate 3 — Stronger model:** Q-033/034, optionally Q-035. Do not break a Gate-2 model to add an untested improvement.

**Gate 4 — Research:** Q-036 and advanced spatial/contrastive training only when Gate 2 and measured ablations already complete.

## Recommended command contract (commands become executable as modules land)

```bash
# Never run on wrong worktree; `pwd` and `git branch --show-current` first
python -m pip install -e .
python scripts/spec_codegen.py --check
python scripts/spec_lint.py
RSNA_EXPECT=full python -m pytest tests/test_data_contract.py -q
python -m pytest -q

# Local ingestion: input path only after Glimmer finishes
python scripts/ingest_glimmer.py --validate-only --input "$GLIMMER_OUTPUT"

# Quartz, actual configuration and mounted data confirmed first
RSNA_DATA="$RSNA_DATA" python -m oxalpha.training.train --config configs/quartz.yaml --stage frozen --fold 0
RSNA_DATA="$RSNA_DATA" python -m oxalpha.training.train --config configs/quartz.yaml --stage finetune --fold 0

# Kaggle packaging requires actual attached checkpoint + offline assets
python scripts/validate_submission.py --submission submission.csv --sample "$RSNA_DATA/sample_submission.csv"
```

**Not executed:** The above new CLIs/tests are a target contract, not existing implementations. Do not mark tasks done until commands actually pass.
