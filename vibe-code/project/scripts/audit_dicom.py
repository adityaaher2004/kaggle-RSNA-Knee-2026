#!/usr/bin/env python3
"""Q-016: resumable, header-only RSNA DICOM manifest audit.

Run from project root: python scripts/audit_dicom.py --data-root "$RSNA_DATA"
Resume uses an append-only JSONL checkpoint; parquet files are atomically replaced.
No raw PatientID, patient names, reports, or pixel data are written.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any

# Allow running from repo root without editable installation.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, allow_nan=False, sort_keys=True)
        f.write('\n')
    os.replace(temp, path)


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        required = {'StudyInstanceUID', 'SeriesInstanceUID'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'{path}: missing columns {sorted(required - set(reader.fieldnames or []))}')
        return list(reader)


def discover_series(data_root: Path, split: str) -> tuple[list[dict[str, str]], list[str]]:
    metadata = data_root / f'{split}_series.csv'
    rows = load_rows(metadata)
    seen: set[tuple[str, str]] = set()
    issues = []
    for row in rows:
        pair = (row['StudyInstanceUID'].strip(), row['SeriesInstanceUID'].strip())
        if not all(pair):
            raise ValueError(f'{metadata}: empty UID')
        if pair in seen:
            issues.append(f'duplicate series metadata row: {pair[0]}/{pair[1]}')
        seen.add(pair)
    if issues:
        raise ValueError('; '.join(issues[:10]))
    return rows, issues


def _vec_normal(orientation: Any) -> tuple[float, float, float] | None:
    if orientation is None or len(orientation) != 6:
        return None
    try:
        a, b, c, d, e, f = (float(x) for x in orientation)
        n = (b*f-c*e, c*d-a*f, a*e-b*d)
        scale = math.sqrt(sum(x*x for x in n))
        if not math.isfinite(scale) or scale < 1e-7:
            return None
        return tuple(x / scale for x in n)
    except (ValueError, TypeError):
        return None


def _geometry(headers: list[Any]) -> tuple[list[str], int, float | None]:
    if not headers:
        return ['no_readable_headers'], 0, None
    flags: set[str] = set()
    shapes = Counter((h.rows, h.columns, h.pixel_spacing) for h in headers)
    if len(shapes) > 1:
        flags.add('inconsistent_geometry')
    normals = [_vec_normal(h.orientation) for h in headers]
    good = [n for n in normals if n is not None]
    if not good or any(n is None for n in normals):
        flags.add('ordering_unreliable')
    ref = good[0] if good else None
    if ref is not None:
        for n in good:
            if abs(sum(x*y for x, y in zip(n, ref))) < math.cos(math.radians(3.0)):
                flags.add('inconsistent_orientation')
                break
        axis = max(range(3), key=lambda j: abs(ref[j]))
        if math.degrees(math.acos(min(1., abs(ref[axis])))) > 15.:
            flags.add('oblique')
        if ref[axis] < 0:
            flags.add('reversed_normal')
    position_values = []
    if ref is not None and all(h.position is not None for h in headers):
        for h in headers:
            try:
                position_values.append(sum(x * float(y) for x, y in zip(ref, h.position)))
            except (TypeError, ValueError):
                flags.add('ordering_unreliable')
                position_values = []
                break
    else:
        flags.add('ordering_unreliable')
    duplicates = 0
    median_gap = None
    if position_values:
        values = sorted(position_values)
        distinct = [values[0]]
        for value in values[1:]:
            if abs(value - distinct[-1]) <= 0.1:
                duplicates += 1
            else:
                distinct.append(value)
        gaps = [y-x for x, y in zip(distinct, distinct[1:])]
        if gaps:
            median_gap = statistics.median(gaps)
        if len(gaps) >= 2 and statistics.mean(gaps) > 0:
            if statistics.pstdev(gaps) / statistics.mean(gaps) > 0.3:
                flags.add('irregular_spacing')
    return sorted(flags), duplicates, median_gap


def _uid(value: Any) -> str:
    return str(value).strip() if value is not None else ''


def audit_one(row: dict[str, str], series_root: Path, *, decode: bool = False,
              decode_limit: int = 2, patient_salt: str | None = None,
              read_header_fn=None, read_slice_fn=None) -> dict[str, Any]:
    """Audit one series, keeping all persisted fields safe for a shared manifest."""
    if read_header_fn is None or read_slice_fn is None:
        from oxalpha.data import dicom
    reader = read_header_fn if read_header_fn is not None else dicom.read_header
    decoder = read_slice_fn if read_slice_fn is not None else dicom.read_slice
    study, series = row['StudyInstanceUID'].strip(), row['SeriesInstanceUID'].strip()
    folder = series_root / study / series
    started = time.monotonic()
    out: dict[str, Any] = {
        'StudyInstanceUID': study, 'SeriesInstanceUID': series,
        'Anatomical_Plane': row.get('Anatomical_Plane', ''),
        'Fluid_Sensitive': row.get('Fluid_Sensitive', ''),
        'Fat_Suppression': row.get('Fat_Suppression', ''),
        'n_files': 0, 'n_headers_ok': 0, 'n_header_failed': 0,
        'n_decoded': 0, 'n_decode_failed': 0, 'n_duplicate_positions': 0,
        'median_gap_mm': None, 'manufacturer': None, 'transfer_syntax_uid': None,
        'magnetic_field_strength_t': None, 'rows': None, 'columns': None,
        'pixel_spacing_mm': None, 'patient_group_hash': None,
        'flags': [], 'status': 'ok', 'failure_examples': [], 'elapsed_s': 0.0,
    }
    flags: set[str] = set()
    try:
        if not folder.is_dir():
            flags.add('missing_series_directory')
            out['status'] = 'missing'
            return out
        files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in ('.dcm', '.dicom', ''))
        out['n_files'] = len(files)
        if not files:
            flags.add('no_dicom_files')
            out['status'] = 'missing'
            return out
        headers = []
        sop_seen = set()
        for path in files:
            try:
                h = reader(path)
                headers.append(h)
                if h.study_instance_uid and h.study_instance_uid != study:
                    flags.add('study_uid_mismatch')
                if h.series_instance_uid and h.series_instance_uid != series:
                    flags.add('series_uid_mismatch')
                if h.sop_instance_uid:
                    if h.sop_instance_uid in sop_seen:
                        flags.add('duplicate_sop_uid')
                    sop_seen.add(h.sop_instance_uid)
            except Exception as exc:
                out['n_header_failed'] += 1
                if len(out['failure_examples']) < 5:
                    out['failure_examples'].append({'file': path.name, 'stage': 'header', 'error_type': type(exc).__name__})
        out['n_headers_ok'] = len(headers)
        if not headers:
            flags.add('no_readable_headers')
            out['status'] = 'failed'
        else:
            from collections import Counter as _Counter
            common = lambda values: _Counter(x for x in values if x is not None).most_common(1)
            def major(values):
                c = common(values)
                return c[0][0] if c else None
            out['manufacturer'] = major([h.manufacturer for h in headers])
            out['transfer_syntax_uid'] = major([h.transfer_syntax_uid for h in headers])
            out['magnetic_field_strength_t'] = major([h.magnetic_field_strength_t for h in headers])
            out['rows'] = major([h.rows for h in headers])
            out['columns'] = major([h.columns for h in headers])
            spacing = major([h.pixel_spacing for h in headers])
            out['pixel_spacing_mm'] = list(spacing) if spacing else None
            geo, dup, gap = _geometry(headers)
            flags.update(geo)
            out['n_duplicate_positions'] = dup
            out['median_gap_mm'] = gap
            syntaxes = {h.transfer_syntax_uid for h in headers if h.transfer_syntax_uid}
            if len(syntaxes) > 1:
                flags.add('mixed_transfer_syntax')
            if decode:
                # Read deterministic, distributed samples; never decode entire volume in audit.
                n = min(max(0, decode_limit), len(headers))
                samples = sorted(set(round(i * (len(headers)-1) / max(n-1, 1)) for i in range(n)))
                for i in samples:
                    try:
                        decoder(headers[i].path)
                        out['n_decoded'] += 1
                    except Exception as exc:
                        out['n_decode_failed'] += 1
                        flags.add('pixel_decode_failure')
                        if len(out['failure_examples']) < 5:
                            out['failure_examples'].append({'file': Path(headers[i].path).name, 'stage': 'pixel', 'error_type': type(exc).__name__})
            if out['n_header_failed'] or out['n_decode_failed']:
                out['status'] = 'partial' if out['n_headers_ok'] else 'failed'
            if patient_salt:
                # Only hash patient IDs if a trusted local, explicitly provided salt is used.
                # This requires a separate local restricted-input pydicom read.
                try:
                    import pydicom
                    ds = pydicom.dcmread(str(headers[0].path), stop_before_pixels=True, specific_tags=['PatientID'])
                    patient = str(ds.get('PatientID', '')).strip()
                    if patient:
                        out['patient_group_hash'] = hashlib.sha256((patient_salt + '\0' + patient).encode()).hexdigest()
                except Exception:
                    flags.add('patient_id_unavailable')
    except Exception as exc:
        flags.add('unexpected_audit_error')
        out['status'] = 'failed'
        out['failure_examples'].append({'stage': 'audit', 'error_type': type(exc).__name__})
    finally:
        out['flags'] = sorted(flags)
        out['elapsed_s'] = round(time.monotonic() - started, 5)
    return out


def checkpoint_rows(path: Path) -> dict[str, dict[str, Any]]:
    done = {}
    if not path.exists():
        return done
    with path.open(encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                # A killed job may leave a partial final write. Fail, never silently skip.
                raise ValueError(f'{path}:{line_no}: corrupt checkpoint; truncate only the incomplete final line') from exc
            key = row['StudyInstanceUID'] + '/' + row['SeriesInstanceUID']
            done[key] = row
    return done


def write_parquet(path: Path, records: list[dict[str, Any]]) -> None:
    try:
        import pandas as pd
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise RuntimeError('Parquet output requires pandas and pyarrow: python -m pip install pandas pyarrow') from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp.parquet')
    prepared = [{**r, 'flags': json.dumps(r['flags']),
                 'failure_examples': json.dumps(r['failure_examples']),
                 'pixel_spacing_mm': json.dumps(r['pixel_spacing_mm'])} for r in records]
    pd.DataFrame(prepared).to_parquet(temp, index=False)
    os.replace(temp, path)


def summarize(records: list[dict[str, Any]], expected: int) -> dict[str, Any]:
    counts = Counter(r['status'] for r in records)
    flags = Counter(flag for r in records for flag in r['flags'])
    codecs = Counter(r['transfer_syntax_uid'] or 'unknown' for r in records)
    vendors = Counter(r['manufacturer'] or 'unknown' for r in records)
    grouped = Counter(r['StudyInstanceUID'] for r in records)
    total_files = sum(r['n_files'] for r in records)
    total_failed = sum(r['n_header_failed'] for r in records)
    decoded = sum(r['n_decoded'] for r in records)
    decode_failed = sum(r['n_decode_failed'] for r in records)
    timings = [r['elapsed_s'] for r in records]
    return {
        'expected_series': expected, 'audited_series': len(records),
        'studies': len(grouped), 'complete': len(records) == expected,
        'status_counts': dict(sorted(counts.items())),
        'flag_counts': dict(sorted(flags.items())),
        'transfer_syntax_counts': dict(sorted(codecs.items())),
        'manufacturer_counts': dict(sorted(vendors.items())),
        'total_dicom_files': total_files, 'header_failures': total_failed,
        'header_failure_rate': total_failed / total_files if total_files else None,
        'decoded_samples_ok': decoded, 'decoded_samples_failed': decode_failed,
        'pixel_decode_sample_failure_rate': decode_failed / (decoded + decode_failed) if decoded + decode_failed else None,
        'series_per_study_min': min(grouped.values()) if grouped else None,
        'series_per_study_max': max(grouped.values()) if grouped else None,
        'median_series_audit_seconds': statistics.median(timings) if timings else None,
        'limitations': ['Header-only scan unless --decode-samples is enabled.',
                        'Patient grouping unavailable unless --patient-salt-env is supplied.',
                        'Pixel decoding is sampled, not comprehensive.'],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=os.environ.get('RSNA_DATA'))
    parser.add_argument('--out-dir', type=Path, default=Path('work'))
    parser.add_argument('--splits', nargs='+', choices=['train', 'test'], default=['train', 'test'])
    parser.add_argument('--decode-samples', type=int, default=0, metavar='N', help='decode up to N distributed slices per series')
    parser.add_argument('--limit', type=int, default=0, help='audit first N series per split (smoke test)')
    parser.add_argument('--retry-failed', action='store_true', help='re-audit failed/partial/missing checkpoint rows')
    parser.add_argument('--patient-salt-env', help='environment-variable name with private salt for hashed PatientID grouping')
    parser.add_argument('--no-parquet', action='store_true', help='checkpoint and JSON summary only; no parquet export')
    args = parser.parse_args(argv)
    if args.data_root is None or not args.data_root.is_dir():
        parser.error('--data-root must point to existing directory (or set RSNA_DATA)')
    if args.decode_samples < 0 or args.limit < 0:
        parser.error('--decode-samples and --limit must be nonnegative')
    salt = None
    if args.patient_salt_env:
        salt = os.environ.get(args.patient_salt_env)
        if not salt or len(salt) < 16:
            parser.error('patient salt environment variable must contain at least 16 characters')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        rows, _ = discover_series(args.data_root, split)
        if args.limit:
            rows = rows[:args.limit]
        checkpoint = args.out_dir / f'manifest_{split}.jsonl'
        existing = checkpoint_rows(checkpoint)
        # Avoid mixing incompatible scans in one resumable checkpoint.
        config_file = args.out_dir / f'manifest_{split}.config.json'
        config = {'data_root': str(args.data_root.resolve()), 'split': split,
                  'decode_samples': args.decode_samples, 'limited': args.limit,
                  'patient_hash_enabled': bool(salt), 'schema_version': 1,
                  'metadata_sha256': hashlib.sha256((args.data_root / f'{split}_series.csv').read_bytes()).hexdigest()}
        if config_file.exists():
            previous = json.loads(config_file.read_text())
            if previous != config:
                raise ValueError(f'{config_file}: resume configuration mismatch; use a different --out-dir')
        else:
            atomic_json(config_file, config)
        print(f'{split}: {len(rows)} metadata series; {len(existing)} checkpoint rows', flush=True)
        # Appending a JSON line then flushing creates a stable restart point.
        with checkpoint.open('a', encoding='utf-8') as output:
            for i, row in enumerate(rows, 1):
                key = row['StudyInstanceUID'].strip() + '/' + row['SeriesInstanceUID'].strip()
                if key in existing and (not args.retry_failed or existing[key]['status'] == 'ok'):
                    continue
                result = audit_one(row, args.data_root / f'{split}_series',
                                   decode=args.decode_samples > 0,
                                   decode_limit=args.decode_samples, patient_salt=salt)
                existing[key] = result
                output.write(json.dumps(result, allow_nan=False, sort_keys=True) + '\n')
                output.flush()
                if i % 250 == 0:
                    print(f'{split}: {i}/{len(rows)}; latest status={result["status"]}', flush=True)
        selected = [existing[r['StudyInstanceUID'].strip() + '/' + r['SeriesInstanceUID'].strip()] for r in rows]
        summary = summarize(selected, len(rows))
        atomic_json(args.out_dir / f'audit_{split}_summary.json', summary)
        if not args.no_parquet:
            write_parquet(args.out_dir / f'manifest_{split}.parquet', selected)
        print(f'{split}: {json.dumps({k: summary[k] for k in ("audited_series", "studies", "status_counts", "header_failure_rate")})}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
