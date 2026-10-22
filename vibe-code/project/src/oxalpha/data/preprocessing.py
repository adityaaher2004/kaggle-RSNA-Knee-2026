"""Deterministic MRI series preprocessing. SPEC: Q-IMG-02 / Q-014 (T-DATA-2).

Uses only DICOM-derived pixel arrays and metadata, never labels or reports.
Inputs: SeriesReadResult from ``dicom.read_series``.
Output: ``SeriesTensor.images`` [K,3,H,W] float32 and masks.
All train-only stochastic transforms are controlled by an explicit RNG seed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import math
import numpy as np
from scipy.ndimage import zoom, affine_transform

from .dicom import SeriesReadResult

PREPROCESS_VERSION = 'quartz-q-img-02-v1'
IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


class PreprocessError(ValueError):
    """Series cannot be converted to reliable model inputs."""


@dataclass(frozen=True)
class PreprocessConfig:
    size: int = 288
    slices: int = 24
    low_percentile: float = 0.5
    high_percentile: float = 99.5
    imagenet_normalize: bool = True
    # Avoid random anatomical mirroring: medial/lateral targets are asymmetric.
    max_rotation_deg: float = 10.0
    zoom_min: float = 0.9
    zoom_max: float = 1.1
    intensity_jitter: float = 0.1
    noise_std: float = 0.02
    slice_dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.size <= 0 or self.slices <= 0:
            raise ValueError('size and slices must be positive')
        if not 0 <= self.low_percentile < self.high_percentile <= 100:
            raise ValueError('invalid percentile bounds')
        if not 0 < self.zoom_min <= self.zoom_max:
            raise ValueError('invalid augmentation zoom range')
        if min(self.max_rotation_deg, self.intensity_jitter, self.noise_std) < 0:
            raise ValueError('negative augmentation parameter')
        if not 0 <= self.slice_dropout < 1:
            raise ValueError('slice_dropout must be in [0,1)')


@dataclass
class SeriesTensor:
    images: np.ndarray  # K x 3 x H x W, ready for ImageNet backbone
    slice_mask: np.ndarray  # K bool, false for padding or train-time dropped slices
    spatial_mask: np.ndarray  # K x H x W bool; foreground valid voxels only
    selected_indices: np.ndarray  # K int, -1 for padding
    flags: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)


def normalize_intensity(volume: np.ndarray, valid_mask: np.ndarray, *,
                        low: float = 0.5, high: float = 99.5) -> tuple[np.ndarray, set[str], tuple[float, float]]:
    """Whole-series percentile normalization over valid voxels; invalid -> zero.

    Pixel padding was already removed prior to modality rescaling in Q-IMG-01.
    Zero-valued *valid* MRI intensities are retained in percentile statistics.
    """
    vol = np.asarray(volume, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)
    if vol.ndim != 3 or valid.shape != vol.shape:
        raise PreprocessError('expected matching [N,H,W] volume and valid_mask')
    valid = valid & np.isfinite(vol)
    out = np.zeros(vol.shape, dtype=np.float32)
    flags: set[str] = set()
    if not valid.any():
        flags.add('empty_valid_volume')
        return out, flags, (float('nan'), float('nan'))
    lower, upper = (float(x) for x in np.percentile(vol[valid], [low, high]))
    if not np.isfinite(lower + upper) or upper <= lower + 1e-8:
        flags.add('constant_intensity')
        return out, flags, (lower, upper)
    out[valid] = np.clip((vol[valid] - lower) / (upper - lower), 0., 1.).astype(np.float32)
    return out, flags, (lower, upper)


def physical_slice_positions(series: SeriesReadResult) -> np.ndarray | None:
    """Project positions onto the reference slice normal; monotonic orientation.

    Returns None when geometry unavailable/unreliable; caller then samples by
    ordinal slice index and records a flag instead of inventing spacing.
    """
    hs = series.headers
    if not hs or any(h.position is None or h.orientation is None for h in hs):
        return None
    o = np.asarray(hs[0].orientation, dtype=np.float64)
    n = np.cross(o[:3], o[3:])
    length = np.linalg.norm(n)
    if length < 1e-7:
        return None
    n /= length
    positions = np.asarray([np.dot(np.asarray(h.position, dtype=np.float64), n) for h in hs])
    if not np.isfinite(positions).all():
        return None
    # Accept either direction (Q-IMG-01 sorts to a standard physical direction).
    if len(positions) > 1 and np.all(np.diff(positions) < 0):
        positions = -positions
    if len(positions) > 1 and not np.all(np.diff(positions) > 0):
        return None
    return positions


def sample_slice_indices(n: int, k: int, positions: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic physical coverage. No interpolated synthetic slices.

    N<K: all observed slices included, remainder padded. N>K: nearest
    distinct slices to K uniformly spaced physical targets incl. endpoints.
    """
    if n < 1 or k < 1:
        raise ValueError('n and k must be >= 1')
    if positions is not None:
        p = np.asarray(positions, dtype=np.float64)
        if p.shape != (n,) or not np.isfinite(p).all() or (n > 1 and not np.all(np.diff(p) > 0)):
            raise ValueError('positions must be strictly increasing and finite')
    else:
        p = np.arange(n, dtype=np.float64)
    if n <= k:
        indices = np.full(k, -1, dtype=np.int64)
        indices[:n] = np.arange(n)
        return indices, indices >= 0
    if k == 1:
        return np.asarray([int(np.argmin(np.abs(p - (p[0] + p[-1]) / 2)))], dtype=np.int64), np.ones(1, bool)
    targets = np.linspace(p[0], p[-1], k)
    # Nearest available distinct index subject to remaining capacity.
    selected = np.empty(k, dtype=np.int64)
    selected[0], selected[-1] = 0, n - 1
    for t in range(1, k - 1):
        lower = int(selected[t - 1]) + 1
        upper = n - (k - t)
        selected[t] = np.argmin(np.abs(p[lower:upper + 1] - targets[t])) + lower
    return selected, np.ones(k, dtype=bool)


def letterbox(image: np.ndarray, mask: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    """Preserve full in-plane FOV and aspect ratio, padding symmetrically with 0."""
    arr = np.asarray(image, dtype=np.float32)
    valid = np.asarray(mask, dtype=bool)
    if arr.ndim != 2 or valid.shape != arr.shape:
        raise PreprocessError('letterbox expects matched two-dimensional image and mask')
    h, w = arr.shape
    if h == 0 or w == 0 or size < 1:
        raise PreprocessError('invalid input or target spatial dimensions')
    scale = min(size / h, size / w)
    hh, ww = min(size, max(1, int(round(h * scale)))), min(size, max(1, int(round(w * scale))))
    # Explicit scipy interpolation; no reliance on OpenCV presence in offline notebook.
    resized = zoom(arr, (hh / h, ww / w), order=1, mode='nearest', prefilter=False)
    rm = zoom(valid.astype(np.uint8), (hh / h, ww / w), order=0, mode='nearest', prefilter=False) > 0
    # scipy zoom dimensions are rounded; crop/pad defensive parity guard.
    resized = resized[:hh, :ww]
    rm = rm[:hh, :ww]
    top, left = (size - resized.shape[0]) // 2, (size - resized.shape[1]) // 2
    out = np.zeros((size, size), dtype=np.float32)
    out_mask = np.zeros((size, size), dtype=bool)
    out[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    out_mask[top:top + rm.shape[0], left:left + rm.shape[1]] = rm
    out[~out_mask] = 0.
    return out, out_mask, (top, left, resized.shape[0], resized.shape[1])


def _augment_series(images: np.ndarray, masks: np.ndarray, cfg: PreprocessConfig,
                    rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """One identical spatial affine across the series; independent low image noise."""
    angle = math.radians(float(rng.uniform(-cfg.max_rotation_deg, cfg.max_rotation_deg)))
    scale = float(rng.uniform(cfg.zoom_min, cfg.zoom_max))
    cosine, sine = math.cos(angle), math.sin(angle)
    matrix = np.asarray(((cosine, -sine), (sine, cosine)), dtype=float) / scale
    center = (np.asarray(images.shape[1:], dtype=float) - 1) / 2
    offset = center - matrix @ center
    result = np.stack([affine_transform(x, matrix, offset=offset, output_shape=x.shape,
                                        order=1, mode='constant', cval=0., prefilter=False)
                       for x in images]).astype(np.float32)
    valid = np.stack([affine_transform(m.astype(np.uint8), matrix, offset=offset,
                                       output_shape=m.shape, order=0, mode='constant', cval=0) > 0
                      for m in masks])
    contrast = float(rng.uniform(1 - cfg.intensity_jitter, 1 + cfg.intensity_jitter))
    brightness = float(rng.uniform(-cfg.intensity_jitter, cfg.intensity_jitter))
    result = np.clip(result * contrast + brightness, 0, 1)
    if cfg.noise_std:
        result = np.clip(result + rng.normal(0., cfg.noise_std, result.shape), 0., 1)
    result[~valid] = 0.
    return result.astype(np.float32), valid


def preprocess_series(series: SeriesReadResult, cfg: PreprocessConfig | None = None,
                      *, training: bool = False, seed: int | None = None) -> SeriesTensor:
    """Transform an ordered decoded series into label-free 2.5D model inputs.

    Padding slots are never constructed from adjacent real slices. Neighboring
    channels always index ORIGINAL adjacent native-grid slices, even if center
    slices are subsampled for compute. Geometric augment is shared across slices.
    """
    cfg = cfg or PreprocessConfig()
    volume, valid = series.volume, series.valid_mask
    norm, flags, limits = normalize_intensity(volume, valid, low=cfg.low_percentile,
                                              high=cfg.high_percentile)
    flags.update(series.flags)
    if 'empty_valid_volume' in flags or 'constant_intensity' in flags:
        raise PreprocessError('series has no usable intensity contrast: ' + ','.join(sorted(flags)))
    positions = physical_slice_positions(series)
    if positions is None:
        flags.add('position_sampling_fallback')
    indices, selected_mask = sample_slice_indices(len(volume), cfg.slices, positions)
    frames, frame_masks = [], []
    placements = []
    for im, ma in zip(norm, valid):
        a, b, placement = letterbox(im, ma, cfg.size)
        frames.append(a)
        frame_masks.append(b)
        placements.append(placement)
    frames = np.stack(frames)
    frame_masks = np.stack(frame_masks)
    if training:
        rng = np.random.default_rng(seed)
        frames, frame_masks = _augment_series(frames, frame_masks, cfg, rng)
    else:
        rng = None
    images = np.zeros((cfg.slices, 3, cfg.size, cfg.size), dtype=np.float32)
    masks = np.zeros((cfg.slices, cfg.size, cfg.size), dtype=bool)
    for out_idx, src in enumerate(indices):
        if src < 0:
            continue
        # Neighbors must be from the same original series, not adjacent sampled centers.
        neighbor_indices = (max(0, int(src) - 1), int(src), min(len(frames) - 1, int(src) + 1))
        images[out_idx] = frames[list(neighbor_indices)]
        masks[out_idx] = frame_masks[int(src)]
    if training and cfg.slice_dropout > 0:
        assert rng is not None
        keep = rng.random(cfg.slices) >= cfg.slice_dropout
        keep &= selected_mask
        if not keep.any():
            keep[int(np.flatnonzero(selected_mask)[0])] = True
        dropped = selected_mask & ~keep
        selected_mask = keep
        images[dropped] = 0.
        masks[dropped] = False
        indices = indices.copy()
        indices[dropped] = -1
    # Normalize only valid slices; padded windows stay exact zeros for mask audits.
    if cfg.imagenet_normalize:
        images[selected_mask] = (images[selected_mask] - IMAGENET_MEAN[None, :, None, None]) / IMAGENET_STD[None, :, None, None]
    header = series.headers[0]
    meta = {
        'preprocess_version': PREPROCESS_VERSION,
        'original_slices': len(volume),
        'original_height_width': (int(volume.shape[1]), int(volume.shape[2])),
        'pixel_spacing_mm': header.pixel_spacing,
        'slice_thickness_mm': header.thickness_mm,
        'spacing_between_slices_mm': header.spacing_between_slices_mm,
        'full_fov_mm': ((header.pixel_spacing[0] * volume.shape[1], header.pixel_spacing[1] * volume.shape[2])
                        if header.pixel_spacing is not None else None),
        'intensity_percentiles': limits,
        'canonical_inplane_orientation': 'unchanged',
        'placement': placements[0],
    }
    return SeriesTensor(images, selected_mask, masks, indices, flags, meta)


__all__ = ['PREPROCESS_VERSION', 'PreprocessConfig', 'PreprocessError', 'SeriesTensor',
           'normalize_intensity', 'physical_slice_positions', 'sample_slice_indices',
           'letterbox', 'preprocess_series']
