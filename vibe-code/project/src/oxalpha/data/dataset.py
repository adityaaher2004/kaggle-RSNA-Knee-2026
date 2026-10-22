"""Study-level MRI selection and inference inputs. SPEC: Q-IMG-03 / Q-015.

No reports, labels or gold-study metadata enter this interface. Series are selected
using imaging metadata only. Both independent contrast flags have unknown states.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
import numpy as np

from .preprocessing import PreprocessConfig, SeriesTensor, preprocess_series
from .dicom import read_series

PLANE_IDS = {'Sagittal': 0, 'Coronal': 1, 'Axial': 2, 'Unknown': 3}
UNKNOWN_BINARY = 2
GEOMETRY_DIM = 6  # row/col spacing, thickness, between-slice distance, height, width


def _plane(raw: Any) -> str:
    if raw is None:
        return 'Unknown'
    x = str(raw).strip().casefold()
    return {'sagittal': 'Sagittal', 'sag': 'Sagittal', 'coronal': 'Coronal',
            'cor': 'Coronal', 'axial': 'Axial', 'ax': 'Axial',
            'transverse': 'Axial'}.get(x, 'Unknown')


def _binary(raw: Any) -> int:
    if raw is None:
        return UNKNOWN_BINARY
    if isinstance(raw, (float, np.floating)) and not np.isfinite(raw):
        return UNKNOWN_BINARY
    if isinstance(raw, (bool, np.bool_)):
        return int(raw)
    value = str(raw).strip().casefold()
    if value in ('1', '1.0', 'true', 'yes', 'y'):
        return 1
    if value in ('0', '0.0', 'false', 'no', 'n'):
        return 0
    return UNKNOWN_BINARY


def _number(value: Any) -> float | None:
    try:
        n = float(value)
        return n if np.isfinite(n) and n > 0 else None
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class SeriesRecord:
    study_uid: str
    series_uid: str
    path: Path
    plane: str = 'Unknown'
    fluid: int = UNKNOWN_BINARY
    fat: int = UNKNOWN_BINARY
    n_slices: int | None = None
    quality: float = 0.0

    @classmethod
    def from_row(cls, row: Mapping[str, Any], series_root: str | Path) -> 'SeriesRecord':
        study = str(row['StudyInstanceUID']).strip()
        series = str(row['SeriesInstanceUID']).strip()
        if not study or not series or study.casefold() == 'nan' or series.casefold() == 'nan':
            raise ValueError('invalid study/series UID')
        root = Path(series_root)
        n = _number(row.get('n_slices', row.get('NumSlices')))
        quality = row.get('quality', 0.0)
        try:
            q = float(quality)
        except (TypeError, ValueError):
            q = 0.
        if not np.isfinite(q):
            q = 0.
        return cls(study, series, root / study / series,
                   _plane(row.get('Anatomical_Plane')),
                   _binary(row.get('Fluid_Sensitive')),
                   _binary(row.get('Fat_Suppression')),
                   int(n) if n is not None else None, q)


def _base_score(rec: SeriesRecord) -> float:
    # Quality must be image-derived or header-derived; never report/label-derived.
    count = max(0, rec.n_slices or 0)
    coverage = min(count / 32., 1.0)
    return 4.0 * rec.quality + coverage + 0.12 * (rec.fluid == 1)


def select_series(records: Sequence[SeriesRecord], *, max_series: int = 8,
                  training: bool = False, seed: int | None = None,
                  randomness: float = 0.18) -> tuple[list[SeriesRecord], list[str]]:
    """Plane-first diversity-aware deterministic greedy selection.

    Inference ignores ``seed`` and is invariant to input ordering. Training can
    perturb image-metadata ranking without dropping any observed known plane.
    Returns selected records and excluded series UIDs; no mutable global RNG.
    """
    if max_series < 1:
        raise ValueError('max_series must be positive')
    if randomness < 0:
        raise ValueError('randomness must be nonnegative')
    if not records:
        return [], []
    study_ids = {r.study_uid for r in records}
    ids = [r.series_uid for r in records]
    if len(study_ids) != 1 or len(set(ids)) != len(ids):
        raise ValueError('series records must contain unique UIDs from one study')
    candidates = sorted(records, key=lambda r: r.series_uid)
    rng = np.random.default_rng(seed) if training else None
    jitter = {r.series_uid: float(rng.uniform(-randomness, randomness)) if rng is not None else 0.
              for r in candidates}
    chosen: list[SeriesRecord] = []
    remaining = list(candidates)

    def pick(pool: list[SeriesRecord]) -> SeriesRecord:
        signatures = {(r.plane, r.fluid, r.fat) for r in chosen}
        contrasts = {(r.fluid, r.fat) for r in chosen}
        def score(r: SeriesRecord) -> tuple[float, str]:
            diversity = 0.6 * ((r.fluid, r.fat) not in contrasts)
            diversity += 0.3 * ((r.plane, r.fluid, r.fat) not in signatures)
            return (_base_score(r) + diversity + jitter[r.series_uid], r.series_uid)
        return max(pool, key=score)

    # Cover all known planes when budget allows, regardless of series count imbalance.
    planes = [p for p in ('Sagittal', 'Coronal', 'Axial') if any(r.plane == p for r in remaining)]
    for p in planes[:max_series]:
        best = pick([r for r in remaining if r.plane == p])
        chosen.append(best)
        remaining.remove(best)
    while remaining and len(chosen) < max_series:
        best = pick(remaining)
        chosen.append(best)
        remaining.remove(best)
    excluded = sorted(r.series_uid for r in remaining)
    return chosen, excluded


@dataclass
class StudyTensor:
    images: np.ndarray       # [S,K,3,H,W] float32
    slice_mask: np.ndarray   # [S,K] bool
    series_mask: np.ndarray  # [S] bool
    plane: np.ndarray        # [S] int64, 3 unknown
    fluid: np.ndarray        # [S] int64, 2 unknown
    fat: np.ndarray          # [S] int64, 2 unknown
    geometry_features: np.ndarray  # [S,G] float32
    geometry_mask: np.ndarray      # [S,G] bool
    study_ids: str
    selected_series_uids: list[str] = field(default_factory=list)
    excluded_series_uids: list[str] = field(default_factory=list)
    failures: dict[str, str] = field(default_factory=dict)

    def inference_inputs(self) -> dict[str, Any]:
        """Whitelist only the exact spec-defined inference fields."""
        return {k: getattr(self, k) for k in (
            'images', 'slice_mask', 'series_mask', 'plane', 'fluid', 'fat',
            'geometry_features', 'geometry_mask', 'study_ids')}


def _geometry(meta: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    spacing = meta.get('pixel_spacing_mm') or (None, None)
    hw = meta.get('original_height_width') or (None, None)
    raw = (spacing[0], spacing[1], meta.get('slice_thickness_mm'),
           meta.get('spacing_between_slices_mm'), hw[0], hw[1])
    arr = np.asarray([_number(x) or 0. for x in raw], dtype=np.float32)
    mask = arr > 0
    # Stable rough numerical ranges to prevent raw pixel sizes dominating embeddings.
    arr /= np.array([1., 1., 10., 10., 1024., 1024.], dtype=np.float32)
    return arr, mask


def assemble_study(study_uid: str, records: Sequence[SeriesRecord],
                   tensors: Mapping[str, SeriesTensor], *, max_series: int = 8,
                   cfg: PreprocessConfig | None = None,
                   excluded: Sequence[str] = (), failures: Mapping[str, str] | None = None) -> StudyTensor:
    """Pad successfully decoded selected series; failure never becomes valid data."""
    cfg = cfg or PreprocessConfig()
    images = np.zeros((max_series, cfg.slices, 3, cfg.size, cfg.size), np.float32)
    slice_mask = np.zeros((max_series, cfg.slices), bool)
    series_mask = np.zeros(max_series, bool)
    plane = np.full(max_series, PLANE_IDS['Unknown'], np.int64)
    fluid = np.full(max_series, UNKNOWN_BINARY, np.int64)
    fat = np.full(max_series, UNKNOWN_BINARY, np.int64)
    geom = np.zeros((max_series, GEOMETRY_DIM), np.float32)
    geom_mask = np.zeros((max_series, GEOMETRY_DIM), bool)
    retained = []
    for rec in records:
        if rec.study_uid != study_uid:
            raise ValueError('mismatched study UID')
        if rec.series_uid not in tensors:
            continue
        t = tensors[rec.series_uid]
        if t.images.shape != (cfg.slices, 3, cfg.size, cfg.size) or t.slice_mask.shape != (cfg.slices,):
            raise ValueError(f'incorrect tensor shape: {rec.series_uid}')
        if not np.isfinite(t.images).all():
            raise ValueError(f'nonfinite MRI pixels: {rec.series_uid}')
        if not np.asarray(t.slice_mask, bool).any():
            continue
        i = len(retained)
        if i >= max_series:
            break
        images[i] = t.images
        slice_mask[i] = t.slice_mask
        series_mask[i] = True
        plane[i] = PLANE_IDS.get(rec.plane, 3)
        fluid[i], fat[i] = rec.fluid, rec.fat
        geom[i], geom_mask[i] = _geometry(t.metadata)
        retained.append(rec.series_uid)
    if not retained:
        raise ValueError(f'no usable MRI series in study {study_uid}')
    return StudyTensor(images, slice_mask, series_mask, plane, fluid, fat,
                       geom, geom_mask, study_uid, retained, list(excluded), dict(failures or {}))


def preprocess_study(study_uid: str, records: Sequence[SeriesRecord], *,
                     series_reader: Callable[[SeriesRecord], Any] | None = None,
                     preprocessor: Callable[..., SeriesTensor] | None = None,
                     preprocess_config: PreprocessConfig | None = None,
                     max_series: int = 8, training: bool = False,
                     seed: int | None = None) -> StudyTensor:
    """Shared Quartz/Kaggle inference preprocessing (no report or labels).

    Readers are injectable for tests/cache parity. Attempts other study series on
    decode failure so a corrupt preferred series need not lose its whole plane.
    """
    cfg = preprocess_config or PreprocessConfig()
    preprocessor = preprocessor or preprocess_series
    if series_reader is None:
        def series_reader(rec: SeriesRecord) -> Any:
            return read_series(sorted(rec.path.glob('*.dcm')))
    prioritized, excluded = select_series(records, max_series=max_series,
                                          training=training, seed=seed)
    # Fallback series are ranked deterministically and attempted after preferred.
    remaining = [r for r in sorted(records, key=lambda x: x.series_uid)
                 if r.series_uid in set(excluded)]
    tensors: dict[str, SeriesTensor] = {}
    failures: dict[str, str] = {}
    accepted: list[SeriesRecord] = []
    queue = prioritized + remaining
    attempted: set[str] = set()
    while queue and len(accepted) < max_series:
        # After a failure, prefer recovering uncovered anatomical planes.
        uncovered = {r.plane for r in records if r.plane in ('Sagittal', 'Coronal', 'Axial')} - {r.plane for r in accepted}
        matching = next((i for i, r in enumerate(queue) if r.plane in uncovered), None)
        rec = queue.pop(0 if matching is None else matching)
        attempted.add(rec.series_uid)
        try:
            raw = series_reader(rec)
            # Stable independent augmentation seed across workers/studies.
            sub_seed = None if seed is None else (seed + sum(bytearray(rec.series_uid.encode()))) % (2**32)
            t = preprocessor(raw, cfg, training=training, seed=sub_seed)
            if not np.asarray(t.slice_mask, bool).any():
                raise ValueError('no real slices')
        except (Exception) as err:
            failures[rec.series_uid] = f'{type(err).__name__}: {err}'
            continue
        tensors[rec.series_uid] = t
        accepted.append(rec)
    excluded_final = sorted(set(r.series_uid for r in records) - set(r.series_uid for r in accepted))
    return assemble_study(study_uid, accepted, tensors, max_series=max_series,
                          cfg=cfg, excluded=excluded_final, failures=failures)


class StudyBagDataset:
    """Label-blind per-study dataset; training labels must be joined outside this class."""
    def __init__(self, records: Sequence[SeriesRecord], *,
                 preprocess_config: PreprocessConfig | None = None,
                 max_series: int = 8, training: bool = False,
                 seed: int = 42,
                 series_reader: Callable[[SeriesRecord], Any] | None = None):
        if max_series < 1:
            raise ValueError('max_series must be positive')
        self.groups: dict[str, list[SeriesRecord]] = {}
        for rec in records:
            self.groups.setdefault(rec.study_uid, []).append(rec)
        self.study_ids = sorted(self.groups)
        self.preprocess_config = preprocess_config or PreprocessConfig()
        self.max_series, self.training, self.seed = max_series, training, seed
        self.epoch = 0
        self.series_reader = series_reader

    def __len__(self) -> int:
        return len(self.study_ids)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __getitem__(self, index: int) -> dict[str, Any]:
        study_uid = self.study_ids[index]
        result = preprocess_study(study_uid, self.groups[study_uid],
                                  series_reader=self.series_reader,
                                  preprocess_config=self.preprocess_config,
                                  max_series=self.max_series, training=self.training,
                                  seed=self.seed + index + 1000003 * self.epoch)
        return result.inference_inputs()


def collate_studies(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError('empty batch')
    names = ('images', 'slice_mask', 'series_mask', 'plane', 'fluid', 'fat',
             'geometry_features', 'geometry_mask')
    return {**{key: np.stack([sample[key] for sample in batch]) for key in names},
            'study_ids': [sample['study_ids'] for sample in batch]}


__all__ = ['SeriesRecord', 'StudyTensor', 'StudyBagDataset', 'PLANE_IDS',
           'UNKNOWN_BINARY', 'GEOMETRY_DIM', 'select_series', 'assemble_study',
           'preprocess_study', 'collate_studies']
