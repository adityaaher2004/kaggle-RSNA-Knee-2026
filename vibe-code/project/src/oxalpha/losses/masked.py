"""Q-LABEL-03: confidence-weighted masked multilabel BCE.

The objective is sum_{i,l}(m*w*BCEWithLogits) / sum_{i,l}(m*w).
Missing, ambiguous and zero-confidence cells contribute exactly zero loss/gradient.
This module is training-only; it must never enter the student inference path.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class MaskedLossResult:
    loss: Tensor
    numerator: Tensor
    weight_sum: Tensor
    supervised_cells: int
    label_coverage: Tensor   # [L] counts of supervised cells; detached

    @property
    def has_supervision(self) -> bool:
        return bool(self.weight_sum.item() > 0)


def masked_weighted_bce(
    logits: Tensor,
    targets: Tensor,
    confidence: Tensor | None = None,
    *,
    observed: Tensor | None = None,
    pos_weight: Tensor | None = None,
    max_confidence: float = 1.0,
) -> MaskedLossResult:
    """Compute a fully mask-safe loss without turning unknowns into negatives.

    Args:
        logits: finite [B,L] raw logits.
        targets: [B,L] values 0/1 for observed; NaN/other for unknown.
        confidence: optional [B,L] nonnegative finite weights, default 1.
            Weights are capped at max_confidence (1.0 by default).
            Zero stays zero; this function NEVER floors positive confidence.
        observed: optional bool mask; combined with binary/finite target and w>0.
        pos_weight: optional positive-class BCE multiplier [L] from TRAIN ONLY.
    Empty batches return a differentiable zero and ``has_supervision=False``.
    """
    if logits.ndim != 2 or targets.shape != logits.shape:
        raise ValueError("logits and targets must both be [B,L] with same shape")
    if not logits.is_floating_point() or not targets.is_floating_point():
        raise TypeError("logits and targets must be floating-point tensors")
    if not torch.isfinite(logits).all().item():
        raise ValueError("logits contain NaN or infinity")
    if max_confidence <= 0 or not torch.isfinite(torch.tensor(max_confidence)).item():
        raise ValueError("max_confidence must be positive and finite")
    if confidence is None:
        confidence = torch.ones_like(targets)
    if confidence.shape != logits.shape or not confidence.is_floating_point():
        raise ValueError("confidence must be floating point [B,L]")
    if observed is not None and (observed.shape != logits.shape or observed.dtype != torch.bool):
        raise ValueError("observed must be bool [B,L]")
    # Known binary target is an explicit hard gate; NaNs and -1 never supervise.
    binary = torch.isfinite(targets) & ((targets == 0) | (targets == 1))
    requested = observed if observed is not None else torch.ones_like(binary)
    active = requested & binary
    if not torch.isfinite(confidence[active]).all().item() or (confidence[active] < 0).any().item():
        raise ValueError("active confidences must be finite and nonnegative")
    if pos_weight is not None:
        if pos_weight.shape != (logits.shape[1],) or not torch.isfinite(pos_weight).all().item() or (pos_weight <= 0).any().item():
            raise ValueError("pos_weight must be finite positive [L]")
        pos_weight = pos_weight.to(device=logits.device, dtype=torch.float32)
    active = active & (confidence > 0)
    weights = torch.where(active, confidence.clamp(min=0, max=max_confidence), 0.).float()
    safe_targets = torch.where(active, targets, 0.).float()
    # torch.float32 avoids overflow of BCE/weighted reduction under CUDA fp16 AMP.
    bce = F.binary_cross_entropy_with_logits(
        logits.float(), safe_targets, pos_weight=pos_weight, reduction="none"
    )
    numerator = (bce * weights).sum()
    denominator = weights.sum().detach()
    loss = numerator / denominator.clamp_min(1.)
    return MaskedLossResult(
        loss, numerator, denominator, int(active.sum().item()),
        active.sum(dim=0).detach().to(dtype=torch.long),
    )


__all__ = ["masked_weighted_bce", "MaskedLossResult"]
