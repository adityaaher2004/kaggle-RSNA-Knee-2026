#!/usr/bin/env python3
"""Fail when source-of-truth requirement IDs are not mapped to work items.

This is a lightweight spec-anchoring smoke test, not a proof of code implementation.
"""
from pathlib import Path
import re
import sys


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    spec = (root / 'SPEC.md').read_text()
    todo = (root / 'TODO.md').read_text()
    definition_ids = set(re.findall(r'^#{2,4} (Q-[A-Z]+-\d{2})\b', spec, re.M))
    referenced_ids = set(re.findall(r'\bQ-[A-Z]+-\d{2}\b', todo))
    missed = definition_ids - referenced_ids
    if missed:
        print('Requirements missing TODO coverage:', *sorted(missed), sep='\n  ')
        return 1
    if not definition_ids:
        print('No requirement definitions found')
        return 1
    print(f'Spec lint OK: {len(definition_ids)} requirement definitions; all referenced by TODO')
    return 0


if __name__ == '__main__':
    sys.exit(main())
