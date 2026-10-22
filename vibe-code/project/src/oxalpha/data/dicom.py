"""DICOM decoding and physical slice ordering. SPEC: Q-IMG-01 (Q-012, Q-013).

No series intensity normalization, spatial resampling or model input construction lives
here: those are Q-IMG-02 / T-DATA-2. Header scanning never decodes pixel data.

The public entry points are ``read_slice``, ``read_series`` and
``scan_slice_headers``. Invalid slices produce structured failures; a series with
no usable slices raises ``SeriesDecodeError``. All images remain at their native
pixel grid and retain float32 modality-rescaled intensities.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence
import math

import numpy as np


class DicomReadError(RuntimeError):
    """A DICOM file cannot be opened, validated or decoded."""


class SeriesDecodeError(RuntimeError):
    """No usable slices were recovered from an MRI series."""


@dataclass(frozen=True)
class SliceFailure:
    path: str
    reason: str


@dataclass(frozen=True)
class SliceHeader:
    path: str
    sop_instance_uid: str | None
    series_instance_uid: str | None
    study_instance_uid: str | None
    rows: int
    columns: int
    pixel_spacing: tuple[float, float] | None
    orientation: tuple[float, ...] | None
    position: tuple[float, ...] | None
    instance_number: int | None
    slice_location: float | None
    laterality: str | None
    image_laterality: str | None
    thickness_mm: float | None
    spacing_between_slices_mm: float | None
    manufacturer: str | None
    magnetic_field_strength_t: float | None
    transfer_syntax_uid: str | None
    pixel_representation: int | None
    rescale_slope: float
    rescale_intercept: float
    pixel_padding_value: float | None
    pixel_padding_range_limit: float | None


@dataclass(frozen=True)
class DecodedSlice:
    header: SliceHeader
    image: np.ndarray  # float32, rescale applied; padding remains zeroed
    valid_mask: np.ndarray  # bool; excludes pixel padding, before rescale


@dataclass
class SeriesReadResult:
    slices: list[DecodedSlice]
    failures: list[SliceFailure] = field(default_factory=list)
    flags: set[str] = field(default_factory=set)
    duplicate_positions: int = 0
    rejected_geometry: int = 0
    @property
    def volume(self) -> np.ndarray:
        return np.stack([s.image for s in self.slices])
    @property
    def valid_mask(self) -> np.ndarray:
        return np.stack([s.valid_mask for s in self.slices])
    @property
    def headers(self) -> list[SliceHeader]:
        return [s.header for s in self.slices]


def _pydicom():
    try:
        import pydicom
    except ImportError as exc:
        raise DicomReadError(
            "pydicom is required: install pydicom and offline decompression plugins "
            "(pylibjpeg, pylibjpeg-libjpeg, pylibjpeg-openjpeg) on Quartz and Kaggle"
        ) from exc
    return pydicom


def _float_attr(ds: Any, name: str) -> float | None:
    value = getattr(ds, name, None)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _int_attr(ds: Any, name: str) -> int | None:
    val = _float_attr(ds, name)
    return int(val) if val is not None and val.is_integer() else None


def _tuple_attr(ds: Any, name: str, length: int) -> tuple[float, ...] | None:
    seq = getattr(ds, name, None)
    if seq is None:
        return None
    try:
        values = tuple(float(x) for x in seq)
    except (TypeError, ValueError):
        return None
    if len(values) != length or not all(math.isfinite(x) for x in values):
        return None
    return values


def _string_attr(ds: Any, name: str) -> str | None:
    value = getattr(ds, name, None)
    return str(value) if value is not None and str(value).strip() else None


def header_from_dataset(ds: Any, path: str | Path = '<memory>') -> SliceHeader:
    """Build a header from a Dataset or a header-like object (no pixels)."""
    rows = _int_attr(ds, 'Rows')
    cols = _int_attr(ds, 'Columns')
    if rows is None or cols is None or rows <= 0 or cols <= 0:
        raise DicomReadError(f'{path}: missing/invalid Rows or Columns')
    slope = _float_attr(ds, 'RescaleSlope')
    intercept = _float_attr(ds, 'RescaleIntercept')
    ts = getattr(getattr(ds, 'file_meta', None), 'TransferSyntaxUID', None)
    return SliceHeader(
        path=str(path), sop_instance_uid=_string_attr(ds, 'SOPInstanceUID'),
        series_instance_uid=_string_attr(ds, 'SeriesInstanceUID'),
        study_instance_uid=_string_attr(ds, 'StudyInstanceUID'),
        rows=rows, columns=cols,
        pixel_spacing=_tuple_attr(ds, 'PixelSpacing', 2),
        orientation=_tuple_attr(ds, 'ImageOrientationPatient', 6),
        position=_tuple_attr(ds, 'ImagePositionPatient', 3),
        instance_number=_int_attr(ds, 'InstanceNumber'),
        slice_location=_float_attr(ds, 'SliceLocation'),
        laterality=_string_attr(ds, 'Laterality'),
        image_laterality=_string_attr(ds, 'ImageLaterality'),
        thickness_mm=_float_attr(ds, 'SliceThickness'),
        spacing_between_slices_mm=_float_attr(ds, 'SpacingBetweenSlices'),
        manufacturer=_string_attr(ds, 'Manufacturer'),
        magnetic_field_strength_t=_float_attr(ds, 'MagneticFieldStrength'),
        transfer_syntax_uid=str(ts) if ts is not None else None,
        pixel_representation=_int_attr(ds, 'PixelRepresentation'),
        rescale_slope=slope if slope is not None else 1.,
        rescale_intercept=intercept if intercept is not None else 0.,
        pixel_padding_value=_float_attr(ds, 'PixelPaddingValue'),
        pixel_padding_range_limit=_float_attr(ds, 'PixelPaddingRangeLimit'),
    )


def read_header(path: str | Path) -> SliceHeader:
    """Header-only scan, explicitly skipping pixel payload."""
    pydicom = _pydicom()
    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True)
        return header_from_dataset(ds, path)
    except Exception as exc:
        raise DicomReadError(f'{path}: header scan failed: {exc}') from exc


def scan_slice_headers(paths: Iterable[str | Path]) -> tuple[list[SliceHeader], list[SliceFailure]]:
    headers: list[SliceHeader] = []
    failures: list[SliceFailure] = []
    for path in paths:
        try:
            headers.append(read_header(path))
        except DicomReadError as exc:
            failures.append(SliceFailure(str(path), str(exc)))
    return headers, failures


def decode_dataset(ds: Any, path: str | Path = '<memory>') -> DecodedSlice:
    """Decode one classic single-frame MR DICOM, preserving signed values.

    pydicom's pixel_array handles signed PixelRepresentation and transfers syntax
    with an available codec. Pixel padding is masked BEFORE modality rescaling.
    Enhanced multiframe objects are rejected explicitly rather than misinterpreted.
    """
    header = header_from_dataset(ds, path)
    if getattr(ds, 'PixelData', None) is None:
        raise DicomReadError(f'{path}: missing PixelData')
    frames = _int_attr(ds, 'NumberOfFrames')
    if frames is not None and frames != 1:
        raise DicomReadError(f'{path}: multiframe DICOM ({frames} frames) is unsupported')
    if header.pixel_representation not in (0, 1):
        raise DicomReadError(f'{path}: invalid/missing PixelRepresentation')
    try:
        pixels = np.asarray(ds.pixel_array)
    except Exception as exc:
        raise DicomReadError(
            f'{path}: pixel decode failed (transfer syntax {header.transfer_syntax_uid}); '
            f'check pylibjpeg/GDCM codec installation: {exc}'
        ) from exc
    if pixels.ndim == 3 and pixels.shape[0] == 1:
        pixels = pixels[0]
    if pixels.ndim != 2 or pixels.shape != (header.rows, header.columns):
        raise DicomReadError(f'{path}: unexpected pixel shape {pixels.shape}')
    if not np.issubdtype(pixels.dtype, np.number):
        raise DicomReadError(f'{path}: nonnumeric pixel dtype {pixels.dtype}')
    if header.pixel_padding_value is not None:
        raw = pixels.astype(np.float64, copy=False)
        if header.pixel_padding_range_limit is None:
            valid = raw != header.pixel_padding_value
        else:
            lower, upper = sorted((header.pixel_padding_value, header.pixel_padding_range_limit))
            valid = (raw < lower) | (raw > upper)
    else:
        valid = np.ones(pixels.shape, dtype=bool)
    image = pixels.astype(np.float32) * np.float32(header.rescale_slope)
    image += np.float32(header.rescale_intercept)
    valid &= np.isfinite(image)
    image[~valid] = 0.
    return DecodedSlice(header=header, image=image, valid_mask=valid)


def read_slice(path: str | Path) -> DecodedSlice:
    pydicom = _pydicom()
    try:
        ds = pydicom.dcmread(str(path))
    except Exception as exc:
        raise DicomReadError(f'{path}: unable to read DICOM: {exc}') from exc
    return decode_dataset(ds, path)


def _normal(h: SliceHeader) -> np.ndarray | None:
    if h.orientation is None:
        return None
    row = np.asarray(h.orientation[:3], dtype=np.float64)
    col = np.asarray(h.orientation[3:], dtype=np.float64)
    n = np.cross(row, col)
    length = np.linalg.norm(n)
    if length < 1e-7 or not np.isfinite(length):
        return None
    return n / length


def _id_key(h: SliceHeader) -> tuple[str, str]:
    return h.sop_instance_uid or '', h.path


def order_slices(slices: Sequence[DecodedSlice], *, duplicate_tol_mm: float = 0.1,
                 orientation_tol_deg: float = 3.0, oblique_tol_deg: float = 15.0,
                 max_gap_cv: float = 0.3) -> SeriesReadResult:
    """Geometry-check, canonical physically increasing sort and duplicate removal.

    Inconsistent geometry is excluded with a flag. A negative dominant component
    of the normal is canonicalized by projecting onto the sign-corrected normal.
    This changes slice *order* only, not in-plane laterality or image orientation.
    """
    if not slices:
        raise SeriesDecodeError('empty series')
    if duplicate_tol_mm < 0:
        raise ValueError('duplicate_tol_mm must be nonnegative')
    flags: set[str] = set()
    failures: list[SliceFailure] = []
    signature_counts = Counter((s.header.rows, s.header.columns,
                                s.header.pixel_spacing) for s in slices)
    majority = max(signature_counts, key=lambda x: (signature_counts[x], str(x)))
    candidates: list[DecodedSlice] = []
    for s in slices:
        h = s.header
        if (h.rows, h.columns, h.pixel_spacing) != majority:
            failures.append(SliceFailure(h.path, 'inconsistent matrix/pixel spacing'))
        else:
            candidates.append(s)
    if len(candidates) != len(slices):
        flags.add('inconsistent_geometry')
    if not candidates:
        raise SeriesDecodeError('no geometrically consistent slices')
    normals = [(s, _normal(s.header)) for s in candidates]
    normal_candidates = [n for _, n in normals if n is not None]
    if normal_candidates:
        ref_n = normal_candidates[0]
        geometric: list[DecodedSlice] = []
        for s, n in normals:
            if n is None or abs(float(np.dot(n, ref_n))) < math.cos(math.radians(orientation_tol_deg)):
                failures.append(SliceFailure(s.header.path, 'inconsistent/missing orientation'))
                flags.add('inconsistent_orientation')
            else:
                geometric.append(s)
        candidates = geometric
        if not candidates:
            raise SeriesDecodeError('no slices have consistent orientations')
        axis = int(np.argmax(np.abs(ref_n)))
        if ref_n[axis] < 0:
            ref_n = -ref_n
            flags.add('reversed_normal')
        if np.degrees(np.arccos(np.clip(abs(ref_n[axis]), -1., 1.))) > oblique_tol_deg:
            flags.add('oblique')
    else:
        ref_n = None
    if ref_n is not None and all(s.header.position is not None for s in candidates):
        scored = [(float(np.dot(ref_n, s.header.position)), s) for s in candidates]
        scored.sort(key=lambda pair: (pair[0], pair[1].header.instance_number or 0, _id_key(pair[1].header)))
        ordered: list[DecodedSlice] = []
        positions: list[float] = []
        dup = 0
        for pos, s in scored:
            if positions and abs(pos - positions[-1]) <= duplicate_tol_mm:
                dup += 1
                failures.append(SliceFailure(s.header.path, 'duplicate slice position'))
            else:
                positions.append(pos)
                ordered.append(s)
        if len(positions) >= 3:
            gaps = np.diff(positions)
            if gaps.mean() > 0 and float(gaps.std() / gaps.mean()) > max_gap_cv:
                flags.add('irregular_spacing')
    else:
        flags.add('ordering_unreliable')
        dup = 0
        if all(s.header.instance_number is not None for s in candidates):
            ordered = sorted(candidates, key=lambda s: (s.header.instance_number, _id_key(s.header)))
        elif all(s.header.slice_location is not None for s in candidates):
            ordered = sorted(candidates, key=lambda s: (s.header.slice_location, _id_key(s.header)))
        else:
            ordered = sorted(candidates, key=lambda s: _id_key(s.header))
            flags.add('fallback_uid_order')
    if len(ordered) == 0:
        raise SeriesDecodeError('no valid slices after ordering')
    study_uids = {s.header.study_instance_uid for s in ordered if s.header.study_instance_uid}
    series_uids = {s.header.series_instance_uid for s in ordered if s.header.series_instance_uid}
    if len(study_uids) > 1 or len(series_uids) > 1:
        raise SeriesDecodeError('mixed StudyInstanceUID or SeriesInstanceUID in series')
    return SeriesReadResult(ordered, failures, flags, dup, len(slices) - len(candidates))


def read_series(paths: Iterable[str | Path], *, max_failure_fraction: float = 0.1,
                **ordering_kwargs: Any) -> SeriesReadResult:
    """Read a series despite isolated corrupted files; reject unsafe failure rates.

    ``max_failure_fraction`` applies to corrupt/geometry-invalid input slices,
    not to benign duplicate slice positions. A series with some valid slices but
    above-threshold corruption raises SeriesDecodeError for caller to log/skip.
    """
    file_paths = sorted((str(p) for p in paths))
    if not file_paths:
        raise SeriesDecodeError('no files in series')
    decoded: list[DecodedSlice] = []
    failures: list[SliceFailure] = []
    for path in file_paths:
        try:
            decoded.append(read_slice(path))
        except DicomReadError as exc:
            failures.append(SliceFailure(path, str(exc)))
    if not decoded:
        raise SeriesDecodeError(f'all {len(file_paths)} DICOM slices failed decoding; first={failures[0].reason}')
    ordered = order_slices(decoded, **ordering_kwargs)
    ordered.failures = failures + ordered.failures
    if failures:
        ordered.flags.add('decode_failures')
    invalid_count = len(failures) + ordered.rejected_geometry
    if invalid_count / len(file_paths) > max_failure_fraction:
        raise SeriesDecodeError(f'{invalid_count}/{len(file_paths)} slices corrupt/inconsistent; '
                                f'exceeds max_failure_fraction={max_failure_fraction}')
    return ordered
