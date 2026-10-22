# Q-016 — DICOM manifest audit patch

Copy `scripts/audit_dicom.py` and `tests/test_audit_dicom.py` into the root of the existing Quartz repo (with T-DATA-1/2/3 installed). The copied `src/` tree in this working directory was used only for local test verification and is **not** included in the ZIP.

## Requirements

Python >=3.10; `pydicom`, `numpy`, `pandas`, `pyarrow` (Parquet output only). Reads `oxalpha.data.dicom.read_header` and optionally `.read_slice` from T-DATA-1.

```bash
module load python/gpu/3.12.5
cd /N/scratch/aaher/RSNA/repo/quartz/vibe-code/project
python -m pip install pydicom numpy pandas pyarrow pytest
export RSNA_DATA=/absolute/path/to/dataset
python scripts/audit_dicom.py --data-root "$RSNA_DATA" --splits train --limit 20 --out-dir work/audit_smoke
python scripts/audit_dicom.py --data-root "$RSNA_DATA" --splits train test --out-dir work/audit_full
python scripts/audit_dicom.py --data-root "$RSNA_DATA" --splits train test --out-dir work/audit_decode --decode-samples 2
python -m pytest -q tests/test_audit_dicom.py
```

Dataset expected: `train_series.csv`, `test_series.csv`, `train_series/<StudyInstanceUID>/<SeriesInstanceUID>/*.dcm`, similarly for test.

**Never use the smoke-test output directory for the full scan:** resume config forbids mixing `--limit`, decode settings, and different input CSV contents. Re-running the identical command resumes; `--retry-failed` retries statuses other than `ok` while preserving an append-only log. A terminated job may leave an incomplete final JSONL line; correct only that final line before resuming.

Outputs for each split:
- `manifest_{split}.jsonl`: append-only per-series progress checkpoint (private work directory).
- `manifest_{split}.parquet`: latest per-series manifest (atomically replaced).
- `audit_{split}_summary.json`: counts, codec/vendor breakdown, error rates and geometry flags.
- `manifest_{split}.config.json`: guard against incorrectly resuming a different audit run.

`--decode-samples 2` tests only two distributed slices per valid series; header-only run is the default. Decoding every slice is intentionally avoided during initial audit. The scanner does not read pixel data in header mode. Per-series file scanning uses a bounded list of file paths, but the output stores all series in memory to write Parquet at the end. Patient IDs are omitted by default. For grouped splitting, `--patient-salt-env RSNA_PATIENT_SALT` hashes `PatientID` (set a private >=16 character environment secret, do not log/commit it). Identical patient IDs map to identical hashes *within a scan using the same salt*; pseudonymization does not alone make manifests public-safe.

Header errors record only basename, stage and exception class, never raw exception messages. Inspect `status=missing/partial/failed` and `flags` before beginning training. This is metadata validation, not validation of medical labels, model predictions, or complete pixel integrity. `--no-parquet` is useful if pyarrow is unavailable.
