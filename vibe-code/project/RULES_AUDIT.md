# RSNA Knee 2026 — Official rules and competition intelligence audit

**Researched October 21, 2026.** Public Kaggle Data/Evaluation content was accessible to web indexing; the full signed-in `/rules` page did not expose body text to our browsing tool. **Do not call unverified detailed rules independently confirmed**. This document separates publicly verified conditions, participant/host posts and unknowns.

## Primary competition sources

1. **Evaluation + code requirements (official):** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/evaluation
2. **Data description (official):** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data
3. **Rules (requires signed-in personal review):** https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules
4. **Host press report:** https://www.rsna.org/news/2026/august/ai-challenge-knee-mri

## Confirmed publicly [P]

| Topic | Contract | Engineering consequence |
|---|---|---|
| Score | Mean of 12 ROC-AUCs, `Σ AUC_i/12` | Never select checkpoint by accuracy/F1 alone |
| Labels | One probability each: `ACL,MCL,Medial Meniscus,Lateral Meniscus,Medial OA,Lateral OA,PF OA,Effusion,Synovitis,Baker's,Contusion,Fracture` | Exact `submission.csv` header/order |
| Submission | Code Competition via Kaggle Notebook, **no internet**, `.csv` produced under working directory | Bundle models/decode dependencies offline; not direct CSV-only |
| Runtime | GPU **≤9h**, CPU **≤9h** | Measure full hidden-style I/O/decode + inference |
| External pretrained models | Freely and publicly available external data and pretrained weights permitted in published Code Requirements | Verify chosen checkpoint/terms and attach offline; don't assume all external data contracts accepted |
| Entry/merger deadline | 2026-10-15 23:59 UTC | Existing registration MUST be verified; new entry may be impossible |
| Final submission | 2026-10-22 23:59 UTC = 2026-10-22 19:59 EDT | Actual notebook hidden rerun must complete in time |
| Winners material | 2026-11-05 UTC | Keep reproducible training code, weights, method documentation |
| Efficiency | `AUC / (Benchmark - maxAUC) + RuntimeSeconds / 32400`, lower is better | No scalar simulation until private benchmark/maxAUC known; profile Pareto frontier |
| Efficiency eligibility | Selected leaderboard submission and private AUC above sample benchmark | AUC still matters; minimize runtime at comparable AUC |
| Test metadata | `test.csv` with only study UID, `test_series.csv` metadata and DICOMs; `Report` absent | Strict MRI-only student |
| Sequence flags | `Fluid_Sensitive` and `Fat_Suppression` correlated but **not guaranteed equal** | Two distinct embeddings and unknown handling |
| Distribution | Test example is three studies but hidden test about 1,300; series typically 20–45 slices with long tail of hundreds | Streaming variable-K/variable-S, time margin |
| DICOM codecs | Explicit/Implicit VR Little Endian; JPEG Lossless; JPEG 2000 | Decoder smoke/hidden-format tests |
| Data size | ~569.76GB / 819,640 files | Metadata manifest + streaming decoding, no full-memory read |
| Dataset labels | Most train condition columns unpopulated, original multilingual reports provided to derive labels | Glimmer adapter first; unknown label ≠ negative |
| Multisite | 16 international sites from RSNA press report | Vendor/site shift analysis; leakage-safe folds |

## Host/forum statements requiring interpretation

- **Access-restricted external data:** a competition-host response dated in public forum says OAI would not violate challenge rules *if* generally accessible without elevated institutional review/sign-off and free of charge. This is a conditional statement, **not** a blanket permission for OAI, MRNet, SKM-TEA, or all click-through datasets. See https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/741819 . Another thread reports that KneeCoT was not permitted because of an institutional agreement; confirm primary host post before relying on that particular detail.
- **Hosted LLM labeling:** a public thread points to host announcement https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733965 , and third-party quotations say commercially hosted LLM inference is allowed subject to accessibility/minimal cost. **Original announcement body could not be independently opened; do not rely on secondhand paraphrase for data-sharing rights.** The planned Muse Glimmer extraction runs **locally** and does not require sending raw reports to a hosted third party.
- **Public competitor performance (NOT a guaranteed score):** an October forum post reports **0.950 public AUC** using ConvNeXt-Small + 3-slice windows + per-finding attention and localized crops; another reports 0.949 for CoAtNet. These are self-reported public-LB findings subject to leaderboard selection, changing splits, and external pretrained/segmentation rights. They motivate prioritizing stronger ConvNeXt representations and label-aware MIL over an assumed efficient B0 winner; they are **not a validated private-LB estimate**.
  - https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/749359
  - https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/746792
- **Corrupt DICOMs:** public participant reports contain truncated pixel-data slices and hidden notebook failure examples. These are diagnostic risk signals, not claims that every distributed file is bad. Build targeted failure fixtures.
  - https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737163
  - https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/745826

## Still unknown / mandatory checks [U]

1. Whether the owner has previously accepted Kaggle rules (entry deadline is past) and has an eligible team/submission slot.
2. Full signed-in Rules text: intellectual property, winning solution licensing, no-data-distribution, accessibility and prohibitions. Do **not** publish raw DICOMs or reports.
3. Exact legal distribution terms of the chosen ConvNeXt pretrained checkpoint and whether derived student weights are publicly distributable in the event of a prize.
4. Exact hidden Kaggle GPU type, memory, preinstalled decoders, allocation and rerun-time resource variability.
5. Actual full IU Quartz dataset path, GPU partition/account, V100 vs H100 allocation, and partial Glimmer extraction status.
6. Whether the 58 original gold labels' selection process matches hidden test distribution (legacy audit says no: zero all-negative studies).
7. Kaggle final notebook actual hidden-rerun result. A successful notebook test on three visible studies is not evidence of hidden success.
8. The origin of apparent `>5,000` exam description versus 4,407-train CSV count. This might include test; do not claim a resolved explanation.

## Exact submission schema

```csv
StudyInstanceUID,ACL,MCL,Medial Meniscus,Lateral Meniscus,Medial OA,Lateral OA,PF OA,Effusion,Synovitis,Baker's,Contusion,Fracture
```

For every hidden `test.csv` UID, write twelve **finite** scores in `[0,1]`, in input UID order, to `/kaggle/working/submission.csv`. Check row set, duplicates, no dropped invalid study, no extra columns. Log only sanitized failure counts, not private study content.

## Release checklist

- [ ] Competition eligibility and signed-in rules reviewed.
- [ ] Offline model, metadata code, preprocessing, decoder wheels installed or attached.
- [ ] Kaggle notebook tested, actual hidden rerun committed and completed with no errors.
- [ ] Full study runtime conservatively below nine hours.
- [ ] Training/report data and sensitive strings absent from deployed inference package.
- [ ] Main metric and any efficiency candidate selected on honest validation with per-label breakdown.
