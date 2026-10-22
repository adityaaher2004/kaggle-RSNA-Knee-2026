"""Mask-safe class-query pooling for the slice and series hierarchies."""
from __future__ import annotations

import math
import torch
from torch import Tensor, nn


def masked_softmax(logits: Tensor, mask: Tensor, dim: int) -> Tensor:
    """Return exactly zero for invalid entries and completely masked bags.

    Use a finite sentinel so even a fully invalid row has a stable softmax
    backward pass; explicit normalization makes valid rows sum to 1.
    """
    mask = torch.broadcast_to(mask.to(torch.bool), logits.shape)
    probs = torch.softmax(logits.float().masked_fill(~mask, -1e4), dim=dim)
    probs = probs * mask.to(probs.dtype)
    probs = probs / probs.sum(dim=dim, keepdim=True).clamp_min(1e-12)
    return probs.to(logits.dtype)


class SliceMIL(nn.Module):
    """[B,S,K,D] to label-specific [B,S,L,D]."""

    def __init__(self, d_model: int, n_labels: int = 12, pooling: str = "gated") -> None:
        super().__init__()
        if pooling not in ("gated", "mean", "max"):
            raise ValueError("slice pooling must be gated, mean or max")
        self.pooling = pooling
        self.n_labels = n_labels
        self.d_model = d_model
        self.tanh_proj = nn.Linear(d_model, d_model)
        self.gate_proj = nn.Linear(d_model, d_model)
        self.queries = nn.Parameter(torch.randn(n_labels, d_model) * (d_model ** -0.5))

    def forward(self, features: Tensor, slice_mask: Tensor) -> tuple[Tensor, Tensor | None]:
        b, s, k, d = features.shape
        if self.pooling == "gated":
            gate = torch.tanh(self.tanh_proj(features)) * torch.sigmoid(self.gate_proj(features))
            scores = torch.einsum("bskd,ld->bskl", gate, self.queries) / math.sqrt(d)
            # [B,S,K,L] -> softmax across *slices* K
            attn = masked_softmax(scores, slice_mask[..., None], dim=2)
            pooled = torch.einsum("bskl,bskd->bsld", attn, features)
            return pooled, attn
        if self.pooling == "mean":
            mask = slice_mask[..., None]
            mean = (features * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1)
            return mean[:, :, None, :].expand(-1, -1, self.n_labels, -1), None
        masked = features.masked_fill(~slice_mask[..., None], torch.finfo(features.dtype).min)
        maximum = masked.amax(dim=2)
        maximum = torch.where(slice_mask.any(dim=2)[..., None], maximum, torch.zeros_like(maximum))
        return maximum[:, :, None, :].expand(-1, -1, self.n_labels, -1), None


class SeriesMIL(nn.Module):
    """[B,S,L,D] to study-level [B,L,D]."""

    def __init__(self, d_model: int, n_labels: int = 12, pooling: str = "gated") -> None:
        super().__init__()
        if pooling not in ("gated", "mean", "max"):
            raise ValueError("series pooling must be gated, mean or max")
        self.pooling = pooling
        self.proj = nn.Linear(d_model, d_model)
        self.queries = nn.Parameter(torch.randn(n_labels, d_model) * (d_model ** -0.5))

    def forward(self, tokens: Tensor, series_mask: Tensor) -> tuple[Tensor, Tensor | None]:
        if self.pooling == "gated":
            scores = torch.einsum("bsld,ld->bsl", self.proj(tokens), self.queries) / math.sqrt(tokens.shape[-1])
            attn = masked_softmax(scores, series_mask[..., None], dim=1)
            return torch.einsum("bsl,bsld->bld", attn, tokens), attn
        if self.pooling == "mean":
            m = series_mask[..., None, None]
            return (tokens * m).sum(dim=1) / m.sum(dim=1).clamp_min(1), None
        masked = tokens.masked_fill(~series_mask[..., None, None], torch.finfo(tokens.dtype).min)
        return masked.amax(dim=1), None
