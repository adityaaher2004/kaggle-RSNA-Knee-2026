# Q-017 — real study smoke gate & optional cache benchmark

Overlay this patch on the existing T-DATA-1/2/3 Quartz project. The modified
`src/oxalpha/data/dataset.py` adds only `selected_slice_indices` to the
non-inference StudyTensor metadata; `inference_inputs()` remains unchanged.

## Real-data smoke test (no cache, run alongside Q-016)

```bash
module load python/gpu/3.12.5
cd /N/scratch/aaher/RSNA/repo/quartz/vibe-code/project
python -m pytest -q tests/test_q017.py
python scripts/smoke_studies.py \
    --data-root "$RSNA_DATA" --split train \
    --limit 30 --random-sample --seed 42 \
    --out-dir work/q017_train
```

On CPU-bound systems, first test with `--limit 3` and/or a reduced documented
configuration `--size 224 --slices 16 --max-series 6`. Default configuration
matches the current model contract (S=8 K=24 H=W=288). Test split:

```bash
python scripts/smoke_studies.py --data-root "$RSNA_DATA" \
    --split test --limit 10 --random-sample --out-dir work/q017_test
```

Reports: `q017_summary.json` (aggregated timings/shape/plane results) and
`q017_results.jsonl` (per-study progress). Exit status is nonzero if any
sampled study fails. The log is restarted for each command invocation;
previous runs should use separate `--out-dir` directories.

## Optional cache experiment (not a full production caching system)

```bash
python scripts/smoke_studies.py --data-root "$RSNA_DATA" \
    --limit 3 --size 224 --slices 16 --max-series 6 \
    --cache --out-dir work/q017_cache
# Repeat the exact command to compare cache misses vs hits.
```

A strong SHA256 hash scans ALL input DICOM bytes for a study before each cache
lookup. The uncompressed NPZ stores complete float32 tensors, potentially
hundreds of MiB per study. This opt-in mode is only a *benchmark*; avoid enabling
it for all ~4k studies or on shared storage until cost and throughput have been
measured. Cache manifests contain input hash, preprocessing version, config,
selected series UIDs, true selected slice indices, checksum, and reserved
backbone hash field. The current cache is full preprocessed MRI tensors, **not**
a pretrained-backbone feature cache. No legacy NPZ is accepted as equivalent.

## Acceptance

- Actual train/test MRI decode and whole-study preprocessing work.
- `[S,K,3,H,W]` tensor shape, finite values, correct series/slice padding,
  separate fluid/fat flags, geometry-mask validity, and available plane
  coverage (unless a series fails decoding).
- Throughput measured as median/p95 wall seconds per study.
- Versioned cache write/read checksum equality; second run logs hit; edits
  to DICOM invalidate the cache via source hash.
- No reports or labels read, and no patient identifiers written.

## Caveats

- CPU memory for default studies can exceed 500 MB; audit one study first.
- Full Kaggle offline reconstruction parity remains a future integration test.
- Source hashing is intentionally expensive and should not be used as the
  production training hot path; this is evidence for choosing a cache design.
- The DICOM reader currently discovers only `*.dcm` under `train_series/<study>/<series>`.
  Verify directory structure and extensions before proceeding.
