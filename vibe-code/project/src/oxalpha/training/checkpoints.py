"""Atomic resumable checkpoints; state is restricted to safe torch-load objects."""
from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import torch


def canonical_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def capture_rng() -> dict[str, Any]:
    return {'torch_cpu': torch.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'python': random.getstate()}


def restore_rng(data: dict[str, Any]) -> None:
    torch.set_rng_state(data['torch_cpu'])
    if data['torch_cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(data['torch_cuda'])
    random.setstate(data['python'])


def save_training_checkpoint(path: str | Path, *, model: torch.nn.Module,
                             optimizer: torch.optim.Optimizer, scheduler: Any,
                             scaler: Any, epoch: int, best_score: float | None,
                             provenance: dict[str, Any],
                             best_rule: str | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'format': 'quartz-train-q031-v1',
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler is not None else None,
        'scaler': scaler.state_dict() if scaler is not None else None,
        'best_score': best_score, 'best_rule': best_rule,
        'provenance': provenance, 'rng': capture_rng(),
    }
    temp = path.with_name(path.name + '.tmp')
    try:
        torch.save(payload, temp)
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def load_training_checkpoint(path: str | Path, *, model: torch.nn.Module,
                             optimizer: torch.optim.Optimizer, scheduler: Any,
                             scaler: Any, expected_provenance: dict[str, Any]) -> tuple[int, float | None, str | None]:
    checkpoint = torch.load(Path(path), map_location='cpu', weights_only=True)
    if checkpoint.get('format') != 'quartz-train-q031-v1':
        raise ValueError('unknown checkpoint format')
    if checkpoint.get('provenance') != expected_provenance:
        raise ValueError('checkpoint data/config provenance mismatch: refuse unsafe resume')
    model.load_state_dict(checkpoint['model'], strict=True)
    optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None and checkpoint['scheduler'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])
    if scaler is not None and checkpoint['scaler'] is not None:
        scaler.load_state_dict(checkpoint['scaler'])
    restore_rng(checkpoint['rng'])
    return checkpoint['epoch'] + 1, checkpoint.get('best_score'), checkpoint.get('best_rule')


__all__ = ['save_training_checkpoint', 'load_training_checkpoint', 'canonical_hash']
