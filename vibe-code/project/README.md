# QuartzKnee-HMIL — Design handoff bundle

Created 2026-10-21 from user-provided `SPEC(2).md`, `TODO(2).md`, and `tree.txt` plus current public RSNA/Kaggle pages. These are **new proposed artifacts**, not changes to the user's actual repo or Git worktree.

## Files

- `SPEC.md` — authoritative architecture, data/report constraints, methods, losses, validation, leakage contract, compute deployment, experiments and research bibliography.
- `TODO.md` — ordered implementation checklist, parallel workstreams, P0 release gate and command contracts.
- `STRUCTURE.md` — complete proposed `oxalpha/` package tree and migration delta from old skeleton.
- `RULES_AUDIT.md` — verifiable competition facts, contest deadline, rights/eligibility unknowns and public-source links.
- `specs/contracts.yaml` — machine-readable label names/order, inference fields, metadata and trial defaults.
- `scripts/spec_codegen.py` — working, verified code generator (`--check` mode), emits Python contracts and interface JSON schema.
- `scripts/spec_lint.py` — working, verified requirement-to-TODO coverage check.
- `src/oxalpha/generated/` — source generated from the included contract; no model/decoder exists yet.

## Verified in this exported bundle

```bash
python scripts/spec_codegen.py --check
python scripts/spec_lint.py
```

Both checks ran successfully in the sandbox. Contract import/strict inference whitelist also passed a local smoke check. **Nothing else has been executed on IU Quartz or Kaggle.**

## Migration policy

1. Confirm actual repo and worktree before copying anything; make `quartz` without overwriting existing `local` or `hpc` history.
2. Preserve working `src/oxalpha/config.py`, `scripts/eda_local.py`, and known-good tests; resolve conflicting paths manually.
3. Copy the four Markdown documents into the appropriate repo root. Import the `specs/`, `scripts/spec_codegen.py`, `scripts/spec_lint.py`, `src/oxalpha/generated/` additions. Check compatibility with `pyproject.toml` and existing imports.
4. `python scripts/spec_codegen.py --check && python scripts/spec_lint.py`; only then implement Phase A–D from `TODO.md`.
5. Never commit competition DICOMs, reports, report-derived raw text, credentials, or model checkpoints. No remote push/history rewrite without owner direction.

## Immediate project blocker

The user reports that Muse Glimmer is still extracting report labels locally. The extraction schema/coverage, full Quartz dataset, Slurm allocation, and Kaggle entry/submission status have not been supplied to this runtime, so a trained Kaggle model cannot truthfully be claimed from these design files. `TODO.md` marks how to verify these blockers; synthetic smoke tests may proceed in parallel.
