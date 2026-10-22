"""QuartzKnee-HMIL: 12 independent MRI-only study-level logits.

Forward accepts exactly the nine inference fields in specs/contracts.yaml.
The caller supplies supervision separately to a training loss, not this model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Any, Sequence

import torch
from torch import Tensor, nn

from .backbone import DEFAULT_CONVNEXT, SliceEncoder
from .mil import SliceMIL, SeriesMIL
from .series_context import SeriesContext


LABELS: tuple[str, ...] = (
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
)


class InvalidStudyError(ValueError):
    """Recoverable inference error: study contains no valid MRI series."""


@dataclass(frozen=True)
class StudentConfig:
    d_model: int = 256
    n_labels: int = 12
    geometry_dim: int = 6
    heads: int = 4
    dropout: float = 0.1
    backbone_name: str = DEFAULT_CONVNEXT
    pretrained: bool = True
    backbone_weights: str | None = None
    freeze_backbone: bool = True
    chunk_size: int = 16
    activation_checkpointing: bool = False
    slice_pool: str = "gated"  # gated/mean/max ablation
    series_pool: str = "gated"


class QuartzKneeStudent(nn.Module):
    """Shared ConvNeXt, class-query slice MIL, context transformer, series MIL.

    Encoder can be injected for CPU unit tests; production requires timm.
    The inference API has no report, text, teacher, or supervision arguments.
    All images are 2.5D RGB-shaped neighboring MRI slices from T-DATA-2.
    """

    def __init__(self, config: StudentConfig | None = None, *,
                 encoder: SliceEncoder | None = None) -> None:
        super().__init__()
        cfg = config or StudentConfig()
        if cfg.n_labels != len(LABELS):
            raise ValueError(f"QuartzKnee requires exactly {len(LABELS)} labels")
        self.config = cfg
        self.encoder = encoder or SliceEncoder(
            d_model=cfg.d_model, model_name=cfg.backbone_name,
            pretrained=cfg.pretrained, weights_path=cfg.backbone_weights,
            chunk_size=cfg.chunk_size, freeze_backbone=cfg.freeze_backbone,
            activation_checkpointing=cfg.activation_checkpointing,
        )
        self.slice_mil = SliceMIL(cfg.d_model, cfg.n_labels, cfg.slice_pool)
        self.context = SeriesContext(cfg.d_model, cfg.geometry_dim, cfg.heads, cfg.dropout)
        self.series_mil = SeriesMIL(cfg.d_model, cfg.n_labels, cfg.series_pool)
        self.token_norm = nn.LayerNorm(cfg.d_model)
        self.classifier_weight = nn.Parameter(torch.empty(cfg.n_labels, cfg.d_model))
        self.classifier_bias = nn.Parameter(torch.zeros(cfg.n_labels))
        nn.init.normal_(self.classifier_weight, mean=0.0, std=0.02)

    def set_backbone_trainable(self, trainable: bool) -> None:
        self.encoder.set_backbone_trainable(trainable)

    def _validate(self, images: Tensor, slice_mask: Tensor, series_mask: Tensor,
                  plane: Tensor, fluid: Tensor, fat: Tensor,
                  geometry_features: Tensor, geometry_mask: Tensor,
                  study_ids: Sequence[str] | None) -> None:
        if images.ndim != 6 or images.shape[3] != 3:
            raise ValueError("images must be [B,S,K,3,H,W]")
        b, s, k = images.shape[:3]
        if min(b, s, k) < 1:
            raise ValueError("B, S, K must be >=1")
        if not images.is_floating_point():
            raise TypeError("images must be floating point")
        if slice_mask.shape != (b, s, k) or slice_mask.dtype != torch.bool:
            raise ValueError("slice_mask must be bool [B,S,K]")
        if series_mask.shape != (b, s) or series_mask.dtype != torch.bool:
            raise ValueError("series_mask must be bool [B,S]")
        if (slice_mask & ~series_mask[..., None]).any().item():
            raise ValueError("invalid series cannot contain valid slices")
        if (series_mask & ~slice_mask.any(dim=-1)).any().item():
            raise ValueError("valid series must contain >=1 valid slice")
        if (~series_mask.any(dim=-1)).any().item():
            raise InvalidStudyError("at least one study has no usable MRI series")
        for name, value, top in (("plane", plane, 3), ("fluid", fluid, 2), ("fat", fat, 2)):
            if value.shape != (b, s) or value.dtype not in (torch.int64, torch.int32):
                raise ValueError(f"{name} must be integer [B,S]")
            if ((value < 0) | (value > top)).any().item():
                raise ValueError(f"{name} enum out of range")
        if geometry_features.shape != (b, s, self.config.geometry_dim):
            raise ValueError("geometry_features shape mismatch")
        if geometry_mask.shape != geometry_features.shape or geometry_mask.dtype != torch.bool:
            raise ValueError("geometry_mask must match geometry_features with bool dtype")
        # Only real, valid metadata values must be finite; invalid slots are masked.
        if not torch.isfinite(geometry_features[geometry_mask & series_mask[..., None]]).all().item():
            raise ValueError("valid geometry metadata contains non-finite values")
        if study_ids is not None and (len(study_ids) != b or not all(isinstance(x, str) for x in study_ids)):
            raise ValueError("study_ids must be one string per study")

    def forward(self, images: Tensor, slice_mask: Tensor, series_mask: Tensor,
                plane: Tensor, fluid: Tensor, fat: Tensor,
                geometry_features: Tensor, geometry_mask: Tensor,
                study_ids: Sequence[str] | None = None, *,
                return_attention: bool = False) -> Tensor | tuple[Tensor, dict[str, Tensor | None]]:
        """Return raw [B,12] logits; sigmoid is applied only for inference.

        All inputs can be supplied as numpy/CPU tensors by caller AFTER conversion
        to torch tensors on a common device; this forward deliberately has no
        silent per-batch tensor transfers or label/text inputs.
        """
        self._validate(images, slice_mask, series_mask, plane, fluid, fat,
                       geometry_features, geometry_mask, study_ids)
        valid = slice_mask & series_mask[..., None]
        features = self.encoder(images, valid)
        slice_tokens, slice_attention = self.slice_mil(features, valid)
        count = valid.sum(dim=2).clamp_min(1)
        summary = features.sum(dim=2) / count[..., None]
        context = self.context(summary, series_mask, plane, fluid, fat,
                               geometry_features, geometry_mask)
        tokens = self.token_norm(slice_tokens + context[:, :, None, :])
        tokens = tokens.masked_fill(~series_mask[..., None, None], 0.)
        study_tokens, series_attention = self.series_mil(tokens, series_mask)
        logits = torch.einsum("bld,ld->bl", study_tokens, self.classifier_weight)
        logits = logits + self.classifier_bias
        if return_attention:
            return logits, {"slice": slice_attention, "series": series_attention}
        return logits


def prepare_student_batch(batch: Mapping[str, Any], *,
                          device: torch.device | str = "cpu") -> dict[str, Any]:
    """Convert T-DATA-3 numpy collate output to the MRI-only forward API.

    Fail closed on unknown fields: labels, reports and teacher features cannot
    accidentally enter the inference path. This is not the training loss API.
    Keep images float32; run CUDA forward under autocast for mixed precision.
    """
    names = ("images", "slice_mask", "series_mask", "plane", "fluid", "fat",
             "geometry_features", "geometry_mask", "study_ids")
    unexpected = set(batch) - set(names)
    if unexpected:
        raise ValueError(f"non-inference batch keys: {sorted(unexpected)}")
    missing = set(names[:-1]) - set(batch)
    if missing:
        raise ValueError(f"missing inference batch keys: {sorted(missing)}")
    floats = {"images", "geometry_features"}
    masks = {"slice_mask", "series_mask", "geometry_mask"}
    out: dict[str, Any] = {}
    for name in names:
        if name not in batch:
            continue
        if name == "study_ids":
            out[name] = batch[name]
        else:
            dtype = torch.float32 if name in floats else (torch.bool if name in masks else torch.long)
            out[name] = torch.as_tensor(batch[name], dtype=dtype, device=device)
    return out


# Alias for scripts that name their loaded inference model Student.
Student = QuartzKneeStudent

__all__ = ["QuartzKneeStudent", "Student", "StudentConfig", "InvalidStudyError", "LABELS", "prepare_student_batch"]
