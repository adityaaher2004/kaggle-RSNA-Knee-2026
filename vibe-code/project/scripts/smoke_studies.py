#!/usr/bin/env python3
"""Q-017: real-data study smoke gate and optional versioned cache benchmark.

Reads train_series.csv/test_series.csv; never reads labels or reports.
Examples:
  python scripts/smoke_studies.py --data-root "$RSNA_DATA" --limit 25
  python scripts/smoke_studies.py --data-root "$RSNA_DATA" --limit 3 --cache --size 224 --slices 16
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from collections import Counter
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

from oxalpha.data.dataset import SeriesRecord, preprocess_study, PLANE_IDS
from oxalpha.data.preprocessing import PreprocessConfig, PREPROCESS_VERSION

FIELDS = ('images', 'slice_mask', 'series_mask', 'plane', 'fluid', 'fat', 'geometry_features', 'geometry_mask')
CACHE_FORMAT = 'quartz-study-v1'


def csv_records(root: Path, split: str) -> dict[str, list[SeriesRecord]]:
    path = root / (split + '_series.csv')
    if not path.is_file():
        raise FileNotFoundError(f'missing series metadata: {path}')
    groups: dict[str, list[SeriesRecord]] = {}
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if not {'StudyInstanceUID', 'SeriesInstanceUID'}.issubset(reader.fieldnames or []):
            raise ValueError('series metadata needs StudyInstanceUID and SeriesInstanceUID')
        for row in reader:
            rec = SeriesRecord.from_row(row, root / (split + '_series'))
            groups.setdefault(rec.study_uid, []).append(rec)
    for study, rows in groups.items():
        ids = [r.series_uid for r in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f'duplicate series UIDs in study {study}')
    return groups


def source_digest(records: list[SeriesRecord]) -> str:
    """Strong source hash for cache correctness; scans *all* source DICOM bytes.

    Expensive by design; opt-in cache mode only. Filename and contents hashed;
    changed, removed, or added DICOM files invalidate the key.
    """
    h = hashlib.sha256()
    for rec in sorted(records, key=lambda r: r.series_uid):
        h.update(rec.series_uid.encode())
        files = sorted(rec.path.glob('*.dcm'))
        h.update(str(len(files)).encode())
        for path in files:
            h.update(path.name.encode())
            with path.open('rb') as f:
                while True:
                    data = f.read(1024 * 1024)
                    if not data:
                        break
                    h.update(data)
    return h.hexdigest()


def cache_key(study: str, source_sha: str, cfg: PreprocessConfig, max_series: int,
              backbone_sha: str = 'none') -> str:
    payload = {'cache_format': CACHE_FORMAT, 'study': study,
               'source_sha256': source_sha, 'preprocess_version': PREPROCESS_VERSION,
               'preprocess_config': vars(cfg), 'max_series': max_series,
               'backbone_sha256': backbone_sha}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def validate_study(inputs: dict[str, Any], *, cfg: PreprocessConfig, max_series: int,
                   expected_planes: set[str] | None = None) -> dict[str, Any]:
    required = set(FIELDS) | {'study_ids'}
    if set(inputs) != required:
        raise ValueError(f'inference whitelist mismatch: {set(inputs) ^ required}')
    expected = {'images': (max_series, cfg.slices, 3, cfg.size, cfg.size),
                'slice_mask': (max_series, cfg.slices), 'series_mask': (max_series,),
                'plane': (max_series,), 'fluid': (max_series,), 'fat': (max_series,),
                'geometry_features': (max_series, 6), 'geometry_mask': (max_series, 6)}
    for name, shape in expected.items():
        if np.asarray(inputs[name]).shape != shape:
            raise ValueError(f'{name} shape: {np.shape(inputs[name])} != {shape}')
    images = np.asarray(inputs['images'])
    series = np.asarray(inputs['series_mask'])
    slices = np.asarray(inputs['slice_mask'])
    if images.dtype != np.float32 or series.dtype != np.bool_ or slices.dtype != np.bool_:
        raise ValueError('wrong tensor/mask dtype')
    if not all(np.isfinite(np.asarray(inputs[k])).all() for k in ('images', 'geometry_features')):
        raise ValueError('nonfinite pixels/geometry')
    if not series.any() or np.any(slices[~series]) or np.any(images[~series] != 0):
        raise ValueError('empty study, invalid padded slice masks or nonzero series padding')
    if not np.all(np.any(slices[series], axis=1)):
        raise ValueError('selected series contains no valid slices')
    if np.any((inputs['plane'] < 0) | (inputs['plane'] > 3)) or any(np.any((inputs[x] < 0) | (inputs[x] > 2)) for x in ('fluid','fat')):
        raise ValueError('out-of-range metadata category')
    if np.any(inputs['geometry_mask'][~series]):
        raise ValueError('padded geometry mask should be false')
    present = {name for name, idx in PLANE_IDS.items() if name != 'Unknown' and
               np.any(np.asarray(inputs['plane'])[series] == idx)}
    if expected_planes and len(expected_planes) <= max_series and not expected_planes.issubset(present):
        raise ValueError(f'lost available anatomical plane(s): {sorted(expected_planes - present)}')
    return {'valid_series': int(series.sum()), 'valid_slices': int(slices.sum()),
            'planes': sorted(present), 'image_megabytes': round(images.nbytes / 1e6, 2)}


def _array_checksum(arrays: dict[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for field in FIELDS:
        x = np.ascontiguousarray(arrays[field])
        h.update(field.encode()); h.update(str(x.dtype).encode()); h.update(str(x.shape).encode())
        h.update(x.tobytes())
    return h.hexdigest()


def write_cache(cache_dir: Path, key: str, inputs: dict[str, Any], *, source_sha: str,
                cfg: PreprocessConfig, selected: list[str], indices: dict[str, Any],
                backbone_sha: str = 'none') -> tuple[Path, dict[str, Any]]:
    """Atomic uncompressed NPZ; does not accept legacy NPZ without manifest."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    arrays = {k: np.asarray(inputs[k]) for k in FIELDS}
    checksum = _array_checksum(arrays)
    path = cache_dir / (key + '.npz')
    manifest = {'format': CACHE_FORMAT, 'key': key,
                'source_sha256': source_sha, 'preprocess_version': PREPROCESS_VERSION,
                'config': vars(cfg), 'backbone_sha256': backbone_sha,
                'study_uid': str(inputs['study_ids']), 'selected_series_uids': selected,
                'selected_slice_indices': indices, 'tensor_sha256': checksum}
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)
    mpath = path.with_suffix('.json')
    mtmp = mpath.with_name(mpath.name + '.tmp')
    with mtmp.open('w') as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    os.replace(mtmp, mpath)
    return path, manifest


def read_cache(cache_dir: Path, key: str) -> tuple[dict[str, Any], dict[str, Any]] | None:
    path = cache_dir / (key + '.npz')
    mpath = path.with_suffix('.json')
    if not path.is_file() or not mpath.is_file():
        return None
    with mpath.open() as f:
        manifest = json.load(f)
    if manifest.get('format') != CACHE_FORMAT or manifest.get('key') != key:
        raise ValueError('unsupported/stale cache manifest (legacy NPZ rejected)')
    with np.load(path, allow_pickle=False) as contents:
        if set(contents.files) != set(FIELDS):
            raise ValueError('cache tensor fields mismatch')
        arrays = {name: contents[name] for name in FIELDS}
    if _array_checksum(arrays) != manifest.get('tensor_sha256'):
        raise ValueError('cache checksum mismatch')
    arrays['study_ids'] = manifest['study_uid']
    return arrays, manifest


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.data_root).expanduser().resolve()
    groups = csv_records(root, args.split)
    studies = sorted(groups)
    if args.random_sample:
        studies = sorted(np.random.default_rng(args.seed).choice(studies, size=min(args.limit, len(studies)), replace=False).tolist())
    else:
        studies = studies[:args.limit]
    cfg = PreprocessConfig(size=args.size, slices=args.slices)
    dest = Path(args.out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    cache_dir = dest / 'cache'
    rows = []
    # One JSONL file per invocation; never append duplicate rows from previous runs.
    (dest / 'q017_results.jsonl').write_text('', encoding='utf8')
    counter = Counter()
    for n, study in enumerate(studies, 1):
        start = time.perf_counter()
        row: dict[str, Any] = {'study_uid': study, 'status': 'error', 'cache': 'disabled'}
        try:
            records = groups[study]
            source_sha = source_digest(records) if args.cache else None
            key = cache_key(study, source_sha, cfg, args.max_series, args.backbone_sha) if source_sha else None
            cached = read_cache(cache_dir, key) if key else None
            if cached is not None:
                inputs, manifest = cached
                row['cache'] = 'hit'
            else:
                result = preprocess_study(study, records, preprocess_config=cfg, max_series=args.max_series)
                inputs = result.inference_inputs()
                row['cache'] = 'miss' if args.cache else 'disabled'
                row['series_failures'] = {k: v[:300] for k,v in result.failures.items()}
                row['excluded_series_count'] = len(result.excluded_series_uids)
            present_planes = {r.plane for r in records if r.plane in ('Sagittal','Coronal','Axial')}
            details = validate_study(inputs, cfg=cfg, max_series=args.max_series,
                                     expected_planes=present_planes if not row.get('series_failures') else None)
            if key and cached is None:
                indices = result.selected_slice_indices
                write_cache(cache_dir, key, inputs, source_sha=source_sha, cfg=cfg,
                            selected=result.selected_series_uids, indices=indices,
                            backbone_sha=args.backbone_sha)
                reread = read_cache(cache_dir, key)
                if reread is None or not all(np.array_equal(inputs[f], reread[0][f]) for f in FIELDS):
                    raise ValueError('cache round-trip parity failure')
            row.update(details)
            row['status'] = 'ok'
        except Exception as err:
            row['error'] = f'{type(err).__name__}: {err}'[:700]
        row['seconds'] = round(time.perf_counter() - start, 3)
        counter[row['status']] += 1
        counter['cache_' + row['cache']] += 1
        rows.append(row)
        print(f'[{n}/{len(studies)}] {row["status"]} {row["seconds"]:.2f}s cache={row["cache"]}', flush=True)
        # Incremental JSONL writes keep partial progress after interruption.
        with (dest / 'q017_results.jsonl').open('a', encoding='utf8') as f:
            f.write(json.dumps(row, sort_keys=True) + '\n')
    times = sorted(r['seconds'] for r in rows if r['status'] == 'ok')
    summary = {'split': args.split, 'sampled': len(studies), 'ok': counter['ok'],
               'failed': counter['error'], 'cache_hits': counter['cache_hit'],
               'cache_misses': counter['cache_miss'], 'total_seconds': round(sum(r['seconds'] for r in rows), 2),
               'median_seconds': float(np.median(times)) if times else None,
               'p95_seconds': float(np.percentile(times,95)) if times else None,
               'preprocess_version': PREPROCESS_VERSION, 'size': cfg.size, 'slices': cfg.slices,
               'max_series': args.max_series, 'rows': rows}
    output = dest / 'q017_summary.json'
    temp = output.with_suffix('.json.tmp')
    with temp.open('w') as f:
        json.dump(summary, f, indent=2)
    os.replace(temp, output)
    print(f'Wrote {output}; ok={counter["ok"]}/{len(studies)}', flush=True)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--split', choices=('train','test'), default='train')
    parser.add_argument('--limit', type=int, default=25)
    parser.add_argument('--out-dir', default='work/q017')
    parser.add_argument('--size', type=int, default=288)
    parser.add_argument('--slices', type=int, default=24)
    parser.add_argument('--max-series', type=int, default=8)
    parser.add_argument('--random-sample', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cache', action='store_true', help='Expensive opt-in strong source hash + NPZ benchmark')
    parser.add_argument('--backbone-sha', default='none', help='Reserved for later backbone-feature cache identity')
    args = parser.parse_args(argv)
    if args.limit < 1 or args.max_series < 1:
        parser.error('limit and max-series must be positive')
    result = run(args)
    return 0 if result['failed'] == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
