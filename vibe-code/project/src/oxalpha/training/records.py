"""Strict weak-label/fold loading for Q-031; gold exclusion is fail-closed.

Canonical CSV schema: StudyInstanceUID, 12 exact abnormality columns with
soft targets [0,1] or null; optional <label>__confidence, <label>__status.
Soft targets encode target probability, not confidence weights.
No report text, extractor evidence, or gold targets leave this module.
"""
from __future__ import annotations

import csv
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from oxalpha.models.student import LABELS


@dataclass(frozen=True)
class TargetRecord:
    targets: tuple[float, ...]  # [0,1] soft/hard targets; NaN for unavailable cells
    confidence: tuple[float, ...]  # always 0 for unavailable cells


@dataclass(frozen=True)
class FoldAssignment:
    split: str
    groups: tuple[tuple[str, str], ...]


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _read_csv(path: str | Path) -> tuple[list[str], list[dict[str, str]]]:
    if Path(path).suffix.lower() == '.parquet':
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError('Install pandas and pyarrow to read Glimmer parquet labels') from exc
        df = pd.read_parquet(path)
        fields = list(df.columns)
        if len(fields) != len(set(fields)) or 'StudyInstanceUID' not in fields:
            raise ValueError('Parquet requires distinct columns and StudyInstanceUID')
        return fields, df.to_dict(orient='records')
    with Path(path).open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)):
            raise ValueError(f'duplicate CSV column in {path}')
        if not fields or 'StudyInstanceUID' not in fields:
            raise ValueError(f'StudyInstanceUID required in {path}')
        return fields, list(reader)


def _uid(row: Mapping[str, str]) -> str:
    uid = str(row.get('StudyInstanceUID', '') or '').strip()
    if not uid or uid.lower() in {'nan', 'none', 'null'}:
        raise ValueError('empty/invalid StudyInstanceUID')
    return uid


def read_gold_ids(path: str | Path) -> set[str]:
    """Gold UID text (one per line) OR CSV with StudyInstanceUID column."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f'gold exclusion list is required: {source}')
    if source.suffix.lower() == '.csv':
        _, rows = _read_csv(source)
        ids = [_uid(r) for r in rows]
    else:
        ids = [line.strip() for line in source.read_text(encoding='utf-8-sig').splitlines()
               if line.strip() and not line.lstrip().startswith('#')]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('gold exclusion file must contain distinct nonempty study UIDs')
    return set(ids)


def _target(raw: str | None) -> float:
    """Parse a Bernoulli soft target in [0,1], or a missing/abstained value.

    Fractional inputs such as 0.25 are probabilities used directly by BCE.
    Missing values remain NaN; malformed or out-of-range data fail closed.
    """
    s = ('' if raw is None else str(raw)).strip().lower()
    if s in {'', 'nan', 'null', 'none', 'na', '<na>', 'unknown', 'uncertain'}:
        return float('nan')
    try:
        value = float(s)
    except (ValueError, TypeError) as exc:
        raise ValueError(f'invalid weak label {raw!r}: expected probability [0,1] or null') from exc
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f'invalid weak label {raw!r}: expected probability [0,1] or null')
    return value


def _cell(row: Mapping[str, str], name: str) -> tuple[float, float]:
    value = _target(row.get(name))
    status_key = name + '__status'
    if status_key in row:
        status = ('' if row.get(status_key) is None else str(row.get(status_key))).strip().lower()
        if status in {'uncertain', 'not_mentioned', 'not_processed', '', 'missing'}:
            return float('nan'), 0.0
        if status not in {'positive', 'negative'}:
            raise ValueError(f'unknown status {status!r} for {name}')
        expected = 1.0 if status == 'positive' else 0.0
        # A status encodes the extractor's hard call; a blended soft target
        # is an independent probability estimate and must not be forced to 0/1.
        # Exact contradictory hard labels still indicate a malformed record.
        if math.isfinite(value) and value in (0.0, 1.0) and value != expected:
            raise ValueError(f'contradictory value/status for {name}: {value}, {status}')
    if not math.isfinite(value):
        return float('nan'), 0.0
    conf_key = name + '__confidence'
    raw = row.get(conf_key) if conf_key in row else None
    if raw is None or str(raw).strip() == '':
        # Target probability and reliability/confidence are distinct.
        # If no confidence column is present, use weight 1 for all observed targets.
        conf = 1.0
    else:
        try:
            conf = float(raw)
        except ValueError as exc:
            raise ValueError(f'invalid confidence for {name}: {raw!r}') from exc
        if not math.isfinite(conf) or conf < 0:
            raise ValueError(f'invalid confidence for {name}: {raw!r}')
        conf = min(conf, 1.0)
    return value, conf


def load_weak_labels(path: str | Path, *, gold_ids: set[str]) -> dict[str, TargetRecord]:
    if not gold_ids:
        raise ValueError('explicit, nonempty gold exclusion set is mandatory')
    columns, rows = _read_csv(path)
    if set(LABELS) - set(columns):
        raise ValueError(f'missing weak label columns: {sorted(set(LABELS)-set(columns))}')
    labels: dict[str, TargetRecord] = {}
    for row in rows:
        uid = _uid(row)
        if uid in labels:
            raise ValueError(f'duplicate weak label StudyInstanceUID: {uid}')
        if uid in gold_ids:
            # Fail closed rather than just silently exclude: a source-contamination error.
            raise ValueError('gold study is present in weak-label table: abort training')
        pairs = [_cell(row, name) for name in LABELS]
        labels[uid] = TargetRecord(tuple(x for x, _ in pairs), tuple(w for _, w in pairs))
    return labels


def load_folds(path: str | Path, *, gold_ids: set[str]) -> dict[str, FoldAssignment]:
    columns, rows = _read_csv(path)
    if 'split' not in columns:
        raise ValueError('folds.csv must contain split column (train/val)')
    # Known grouping columns are validated as connected components over study IDs:
    # if any two studies share ANY group, they may not cross train/val boundary.
    group_columns = ('PatientGroup', 'patient_group', 'ClusterGroup', 'cluster_group',
                     'group_id', 'patient_hash', 'near_duplicate_cluster')
    available = [c for c in group_columns if c in columns]
    result: dict[str, FoldAssignment] = {}
    group_splits: dict[tuple[str, str], str] = {}
    for row in rows:
        uid = _uid(row)
        if uid in gold_ids:
            raise ValueError('gold study appears in folds.csv: abort training')
        if uid in result:
            raise ValueError(f'duplicate StudyInstanceUID in folds.csv: {uid}')
        split = (row.get('split') or '').strip().lower()
        if split not in {'train', 'val'}:
            raise ValueError(f'invalid split {split!r}; expected train or val')
        groups = []
        for col in available:
            group = (row.get(col) or '').strip()
            if group and group.lower() not in {'nan', 'none', 'null'}:
                key = (col, group)
                if key in group_splits and group_splits[key] != split:
                    raise ValueError(f'group leakage across train and val in {col}')
                group_splits[key] = split
                groups.append(key)
        result[uid] = FoldAssignment(split, tuple(groups))
    if not any(x.split == 'train' for x in result.values()):
        raise ValueError('no train studies in folds.csv')
    if not any(x.split == 'val' for x in result.values()):
        raise ValueError('no val studies in folds.csv')
    return result


def training_partition(labels: Mapping[str, TargetRecord], folds: Mapping[str, FoldAssignment],
                       available_studies: set[str], *, gold_ids: set[str]) -> tuple[list[str], list[str]]:
    """Require every fold UID is present in weak labels and series metadata."""
    if set(folds) & gold_ids or set(labels) & gold_ids:
        raise ValueError('gold-study leakage detected')
    missing_label = set(folds) - set(labels)
    missing_mri = set(folds) - available_studies
    if missing_label or missing_mri:
        raise ValueError(f'unmatched folds: labels missing={len(missing_label)} MRI missing={len(missing_mri)}')
    train = sorted(uid for uid, f in folds.items() if f.split == 'train')
    val = sorted(uid for uid, f in folds.items() if f.split == 'val')
    if not train or not val or set(train) & set(val):
        raise ValueError('invalid train/validation partition')
    return train, val


__all__ = ['TargetRecord', 'FoldAssignment', 'load_weak_labels', 'load_folds',
           'read_gold_ids', 'training_partition', 'file_sha256']
