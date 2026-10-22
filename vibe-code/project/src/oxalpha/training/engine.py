"""Q-031 training engine: masked BCE, exact accumulation, V100 fp16 AMP.

Targets are supplied separately; the only call to Student.forward uses the MRI
inference field whitelist from Q-030.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch import nn, Tensor

from oxalpha.losses.masked import masked_weighted_bce
from oxalpha.models.student import LABELS, prepare_student_batch


@dataclass
class EpochStats:
    loss: float | None
    supervised_cells: int
    supervised_weight: float
    optimizer_steps: int
    empty_batches: int
    examples: int
    label_coverage: list[int]
    peak_vram_gib: float | None = None

    def asdict(self) -> dict[str, Any]:
        return vars(self).copy()


def _autocast(device: torch.device, amp: bool):
    return torch.autocast(device_type='cuda', dtype=torch.float16) if amp else nullcontext()


def _data_tensors(targets: Any, confidences: Any, device: torch.device) -> tuple[Tensor, Tensor]:
    return (torch.as_tensor(targets, device=device, dtype=torch.float32),
            torch.as_tensor(confidences, device=device, dtype=torch.float32))


def _flush_gradients(model: nn.Module, optimizer: torch.optim.Optimizer,
                     scaler: torch.amp.GradScaler | None, denominator: float,
                     grad_clip: float) -> bool:
    """Accumulate BCE NUMERATORS and divide gradients by the exact window weight.

    This avoids the common error of taking a mean of unequal-confidence batch
    losses, and does not retain previous image graphs in GPU memory.
    """
    if denominator == 0:
        optimizer.zero_grad(set_to_none=True)
        return False
    if scaler is not None and scaler.is_enabled():
        scaler.unscale_(optimizer)
    for p in model.parameters():
        if p.grad is not None:
            p.grad.div_(denominator)
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip, error_if_nonfinite=True)
    if scaler is not None and scaler.is_enabled():
        old = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        stepped = scaler.get_scale() >= old  # overflow leads to decreased scale / skipped update
    else:
        optimizer.step()
        stepped = True
    optimizer.zero_grad(set_to_none=True)
    return stepped


def train_epoch(model: nn.Module, batches: Iterable[tuple[Mapping[str, Any], Any, Any]],
                optimizer: torch.optim.Optimizer, *, device: torch.device | str,
                accumulate: int = 8, amp: bool = True, grad_clip: float = 1.0,
                scaler: torch.amp.GradScaler | None = None,
                pos_weight: Tensor | None = None,
                seen_ids: list[str] | None = None) -> EpochStats:
    """One streamed train epoch; supervised denominator is exact per grad window."""
    if accumulate < 1 or grad_clip < 0:
        raise ValueError('accumulate must be >=1 and grad_clip >=0')
    device = torch.device(device)
    use_amp = bool(amp and device.type == 'cuda')
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_num = total_den = window_den = 0.0
    n_cell = n_empty = n_examples = steps = 0
    coverage = torch.zeros(len(LABELS), dtype=torch.long)
    window_items = 0
    for inference, y, w in batches:
        if seen_ids is not None:
            seen_ids.extend(inference['study_ids'])
        x = prepare_student_batch(inference, device=device)
        y, w = _data_tensors(y, w, device)
        with _autocast(device, use_amp):
            logits = model(**x)
            loss = masked_weighted_bce(logits, y, w, pos_weight=pos_weight)
        window_items += 1
        n_examples += int(y.shape[0])
        n_cell += loss.supervised_cells
        coverage += loss.label_coverage.cpu()
        if loss.has_supervision:
            value = float(loss.weight_sum.item())
            # Do not compute backward on an empty-supervision batch.
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss.numerator).backward()
            else:
                loss.numerator.backward()
            total_num += float(loss.numerator.detach().item())
            total_den += value
            window_den += value
        else:
            n_empty += 1
        if window_items == accumulate:
            if _flush_gradients(model, optimizer, scaler, window_den, grad_clip):
                steps += 1
            window_den = 0.0
            window_items = 0
    if window_items:
        if _flush_gradients(model, optimizer, scaler, window_den, grad_clip):
            steps += 1
    peak = (torch.cuda.max_memory_allocated(device) / 1024**3) if device.type == 'cuda' else None
    return EpochStats(total_num / total_den if total_den else None,
                      n_cell, total_den, steps, n_empty, n_examples, coverage.tolist(), peak)


def _binary_auc(targets: np.ndarray, scores: np.ndarray) -> float:
    """Mann-Whitney AUC with average ranks for tied predictions; no sklearn required."""
    npos = int((targets == 1).sum())
    nneg = int((targets == 0).sum())
    if not npos or not nneg:
        return float('nan')
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    cumulative = np.cumsum(counts)
    avg_ranks = cumulative - (counts - 1) / 2
    positive_rank_sum = avg_ranks[inverse][targets == 1].sum()
    return float((positive_rank_sum - npos * (npos + 1) / 2) / (npos * nneg))


def masked_auc_report(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> dict[str, Any]:
    """Official macro only if all 12 class AUCs are well defined."""
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    if y.ndim != 2 or y.shape != p.shape or y.shape != w.shape or y.shape[1] != len(LABELS):
        raise ValueError('expected matching [N,12] arrays')
    per_label = {}
    valid_auc = []
    # ROC-AUC is defined against observed binary ground truth. Fractional
    # pseudo-labels are supervision for soft BCE, *not* hard AUC outcomes.
    observed_soft = np.isfinite(y) & (y > 0.) & (y < 1.) & np.isfinite(w) & (w > 0)
    n_soft_total = int(observed_soft.sum())
    for i, name in enumerate(LABELS):
        mask = np.isfinite(y[:, i]) & np.isfinite(p[:, i]) & np.isfinite(w[:, i]) & (w[:, i] > 0) & np.isin(y[:, i], (0., 1.))
        truth, preds = y[mask, i], p[mask, i]
        npos, nneg = int((truth == 1).sum()), int((truth == 0).sum())
        auc = _binary_auc(truth, preds)
        if np.isfinite(auc):
            valid_auc.append(auc)
        per_label[name] = {'positives': npos, 'negatives': nneg,
                           'observed': int(mask.sum()),
                           'soft_excluded_from_auc': int(observed_soft[:, i].sum()),
                           'auc': float(auc) if np.isfinite(auc) else None}
    complete = len(valid_auc) == len(LABELS)
    # If any soft validation labels exist, a macro over remaining hard labels
    # must not be presented as an official complete-label AUC.
    eligible_official = complete and n_soft_total == 0
    return {'official_macro_auc': float(np.mean(valid_auc)) if eligible_official else None,
            'partial_macro_auc': float(np.mean(valid_auc)) if valid_auc else None,
            'soft_targets_excluded_from_auc': n_soft_total,
            'hard_macro_auc': float(np.mean(valid_auc)) if complete else None,
            'valid_labels': len(valid_auc), 'total_labels': len(LABELS),
            'per_label': per_label,
            'warning': ('Soft pseudo-labels are excluded from ROC-AUC; checkpoint by validation soft BCE'
                        if n_soft_total else (None if complete else
                        'Partial macro only: at least one label lacks positive or negative examples'))}


@torch.no_grad()
def validate_epoch(model: nn.Module, batches: Iterable[tuple[Mapping[str, Any], Any, Any]],
                   *, device: torch.device | str, amp: bool = True,
                   pos_weight: Tensor | None = None) -> tuple[EpochStats, dict[str, Any]]:
    device = torch.device(device)
    model.eval()
    num = den = 0.0
    n_cells = n_examples = n_empty = 0
    coverage = torch.zeros(len(LABELS), dtype=torch.long)
    all_y, all_p, all_w = [], [], []
    for inference, targets, confidence in batches:
        x = prepare_student_batch(inference, device=device)
        y, w = _data_tensors(targets, confidence, device)
        with _autocast(device, amp and device.type == 'cuda'):
            logits = model(**x)
            loss = masked_weighted_bce(logits, y, w, pos_weight=pos_weight)
        if loss.has_supervision:
            num += float(loss.numerator.detach().item())
            den += float(loss.weight_sum.item())
        else:
            n_empty += 1
        n_cells += loss.supervised_cells
        n_examples += y.shape[0]
        coverage += loss.label_coverage.cpu()
        all_y.append(y.cpu().numpy())
        all_p.append(torch.sigmoid(logits.float()).cpu().numpy())
        all_w.append(w.cpu().numpy())
    stats = EpochStats(num / den if den else None, n_cells, den,
                       0, n_empty, n_examples, coverage.tolist())
    if all_y:
        metrics = masked_auc_report(np.concatenate(all_y), np.concatenate(all_p), np.concatenate(all_w))
    else:
        metrics = masked_auc_report(np.empty((0, len(LABELS))), np.empty((0, len(LABELS))), np.empty((0, len(LABELS))))
    return stats, metrics


__all__ = ['train_epoch', 'validate_epoch', 'masked_auc_report', 'EpochStats']
