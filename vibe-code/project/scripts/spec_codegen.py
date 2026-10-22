#!/usr/bin/env python3
"""Generate/check source contracts from specs/contracts.yaml.

This generates contract constants and JSON Schema; it does NOT write image-model
implementation. Spec changes fail --check until generated code is refreshed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def generate(root: Path) -> dict[Path, bytes]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError('PyYAML required for code generation (not inference)') from exc
    src = root / 'specs/contracts.yaml'
    raw = src.read_bytes()
    doc = yaml.safe_load(raw)
    sha = hashlib.sha256(raw).hexdigest()
    labels = doc['labels']
    assert len(labels) == len(set(labels)) == 12, '12 distinct ordered labels required'
    fields = doc['inference_inputs']
    forbidden = doc['forbidden_inference_inputs']
    assert not set(fields).intersection(forbidden), 'inference leak!'
    assert doc['metadata']['fluid_and_fat_are_independent'] is True
    assert doc['config_defaults']['gold']['use_for_training'] is False
    assert doc['shapes']['logits'] == '[B,12]'

    # Checked in/generated file imports only standard library; no YAML dependency in Kaggle.
    py = f'''# AUTO-GENERATED; DO NOT EDIT. Source: specs/contracts.yaml SHA256={sha}\nfrom __future__ import annotations\nfrom typing import TypedDict, Any\nSPEC_VERSION = {doc['spec_version']!r}\nMODEL_NAME = {doc['model_name']!r}\nCONTRACT_SHA256 = {sha!r}\nLABELS = {tuple(labels)!r}\nSUBMISSION_COLUMNS = ({doc['submission']['id_column']!r},) + LABELS\nPLANES = {tuple(doc['metadata']['planes'])!r}\nUNKNOWN_METADATA_VALUE = 2\nINFERENCE_FIELDS = {tuple(fields)!r}\nFORBIDDEN_INFERENCE_FIELDS = {tuple(forbidden)!r}\nTENSOR_SHAPES = {doc['shapes']!r}\nCONFIG_DEFAULTS = {doc['config_defaults']!r}\n\nclass StudentBatch(TypedDict):\n    images: Any\n    slice_mask: Any\n    series_mask: Any\n    plane: Any\n    fluid: Any\n    fat: Any\n    geometry_features: Any\n    geometry_mask: Any\n    study_ids: Any\n\ndef assert_inference_fields(keys: object) -> None:\n    names = set(keys)\n    if names != set(INFERENCE_FIELDS):\n        raise ValueError(f"Invalid inference keys: missing={{set(INFERENCE_FIELDS) - names}}, unexpected={{names - set(INFERENCE_FIELDS)}}")\n    if names.intersection(FORBIDDEN_INFERENCE_FIELDS):\n        raise ValueError('Report/label supervision has leaked into inference')\n'''
    # The first schema is an interface schema, not a data-content validator for tensors.
    schema = {
        '$schema': 'https://json-schema.org/draft/2020-12/schema',
        'title': 'QuartzKnee inference contract',
        'x-spec-version': doc['spec_version'],
        'x-source-sha256': sha,
        'type': 'object',
        'additionalProperties': False,
        'properties': {k: {'description': doc['shapes'].get(k, 'batch study ids'), 'x-runtime-validator': 'tensor-shape-test-or-id-list'} for k in fields},
        'required': fields,
        'x-output-columns': [doc['submission']['id_column'], *labels],
        'x-default-config': doc['config_defaults'],
    }
    # Do not pretend this schema describes tensor internals; runtime tensor tests do that.
    schema['x-tensor-runtime-validation-required'] = True
    paths = {
        root/'src/oxalpha/generated/contracts.py': py.encode('utf-8'),
        root/'src/oxalpha/generated/config.schema.json': (json.dumps(schema, indent=2, ensure_ascii=False) + '\n').encode('utf-8'),
    }
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--check', action='store_true', help='Fail if outputs are absent/stale; do not write')
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = ap.parse_args()
    outputs = generate(args.root)
    failed = False
    for path, expected in outputs.items():
        if args.check:
            if not path.exists() or path.read_bytes() != expected:
                print('STALE', path)
                failed = True
            else:
                print('OK', path)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(expected)
            print('WROTE', path)
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
