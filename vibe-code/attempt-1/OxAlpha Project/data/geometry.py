"""SPEC: S3.2

Sort slices by dot(IPP, cross(IOP row, IOP col)) with logged fallbacks; drop duplicates/localisers; canonical per-plane orientation; side flag; in-plane resample to spacing_mm; foreground-centred crop/pad to size.

API (planned): sort_slices(headers) -> order, z_mm; canonicalize(vol, iop, plane) -> vol; resample_inplane(vol, spacing, target_mm) -> vol; crop_fg(vol, size) -> vol
"""

raise NotImplementedError("stub -- see specs.txt S3.2")
