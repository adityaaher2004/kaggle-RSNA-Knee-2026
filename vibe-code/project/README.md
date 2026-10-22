# T-DATA-3 / Q-015 — Study-level MRI series selection

Incremental patch for the `quartz` worktree, requiring T-DATA-1 (`dicom.py`) and T-DATA-2 (`preprocessing.py`). This patch **does not** modify your Git branch automatically.

## Files
- `src/oxalpha/data/dataset.py`: label-blind `SeriesRecord`, plane/contrast-diverse selection, padded `StudyTensor`, `preprocess_study`, `StudyBagDataset`, collate helper.
- `tests/test_selection.py`: eight synthetic tests.

## Contracts
- `S=8` max selected series (configurable), `K=24`, `H=W=288` by default.
- Study images `[S,K,3,H,W]`; per-series masks `[S]`, per-slice masks `[S,K]`.
- Plane IDs: sagittal=0, coronal=1, axial=2, unknown=3.
- Fluid-sensitive and fat-suppressed: 0=no, 1=yes, 2=unknown; *independent*, including `fluid=1,fat=0`.
- Geometry features `[S,6]` with matching presence masks, from available slice metadata.
- Inference output fields: `images,slice_mask,series_mask,plane,fluid,fat,geometry_features,geometry_mask,study_ids` **only**.
- Deterministic inference selection regardless of CSV row order. Training adds seeded metadata-rank jitter, preserving three-plane coverage if available.
- On a failed preferred DICOM series, tries other series while restoring missing planes when possible; errors are recorded, never interpreted as valid images.

## Integration
Copy the `src/` and `tests/` files into the existing `quartz` worktree. Run:

```bash
pip install numpy scipy pytest pydicom pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg
pytest -q tests/test_selection.py tests/test_preprocess.py tests/test_dicom_decode.py tests/test_dicom_sort.py
```

To construct records from `train_series.csv` or `test_series.csv`, call `SeriesRecord.from_row(row, series_root)` where `series_root` is the `train_series/` or `test_series/` folder. A `StudyBagDataset(records)` returns inference-only dictionaries. Use `collate_studies(batch)` for NumPy batch stacking; a PyTorch training pipeline may convert its arrays via `torch.from_numpy`.

## Validation boundary
31 tests passed, 2 skipped with the previous patch tests in this environment. Real DICOM round-trip and compressed transfer syntax are not verified here. The complete full-data manifest and memory/throughput profile require Quartz. Neither labels nor reports are loaded by this module. Series selection `quality` must be based on header or image quality only, never target labels.

## Known limits
- All `S×K×3×H×W` float32 inputs can be large; full-float batch collation is appropriate for smoke tests, but stream/decode on demand and use micro-batches on V100.
- `StudyBagDataset` is label-blind; a future *training adapter* should fetch masked labels independently, after dataset group and train/val exclusion checks.
- Geometry values are only produced when the DICOM metadata is provided; unknown dimensions remain masked.
- Full-data source manifest should verify DICOM paths, duplicate IDs, and the actual per-plane distribution.
