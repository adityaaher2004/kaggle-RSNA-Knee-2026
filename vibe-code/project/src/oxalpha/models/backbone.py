"""Shared 2.5D slice encoder. No text or training labels enter this module."""
from __future__ import annotations

from pathlib import Path
from typing import Callable
import warnings

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


DEFAULT_CONVNEXT = "convnext_tiny.fb_in22k_ft_in1k"


class SliceEncoder(nn.Module):
    """Encode ONLY valid MRI slices, in small chunks.

    Backbones return [N,C] feature vectors; timm ConvNeXt-Tiny is default.
    A feature extractor can be injected for tests without installing timm.

    Frozen forward runs without autograd; fine-tuning can use non-reentrant
    activation checkpointing. Chunking *alone* does not cap saved activations
    during backward; enable checkpointing when unfreezing on limited VRAM.
    """

    def __init__(
        self,
        d_model: int = 256,
        *,
        model_name: str = DEFAULT_CONVNEXT,
        pretrained: bool = True,
        weights_path: str | Path | None = None,
        backbone: nn.Module | None = None,
        feature_dim: int | None = None,
        chunk_size: int = 16,
        freeze_backbone: bool = True,
        activation_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if d_model <= 0 or chunk_size <= 0:
            raise ValueError("d_model and chunk_size must be positive")
        if backbone is None:
            try:
                import timm
            except ImportError as exc:
                raise ImportError("Install timm to construct the production ConvNeXt encoder") from exc
            if not pretrained and weights_path is None:
                warnings.warn("ConvNeXt has no pretrained weights: debug-only initialization", RuntimeWarning, stacklevel=2)
            # An explicit local checkpoint takes precedence; avoid network requests.
            backbone = timm.create_model(model_name, pretrained=(pretrained and weights_path is None),
                                         num_classes=0, global_pool="avg", in_chans=3)
            if weights_path is not None:
                # Input must be a trusted checkpoint in timm state_dict format.
                state = torch.load(Path(weights_path), map_location="cpu", weights_only=True)
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                if not isinstance(state, dict):
                    raise ValueError("weights_path must contain a backbone state_dict")
                backbone.load_state_dict(state, strict=True)
        if feature_dim is None:
            feature_dim = getattr(backbone, "num_features", None)
        if not isinstance(feature_dim, int) or feature_dim <= 0:
            raise ValueError("feature_dim required for an injected backbone without num_features")
        self.backbone = backbone
        self.project = nn.Linear(feature_dim, d_model)
        self.chunk_size = chunk_size
        self.activation_checkpointing = activation_checkpointing
        self._frozen = False
        self.set_backbone_trainable(not freeze_backbone)

    @property
    def backbone_frozen(self) -> bool:
        return self._frozen

    def set_backbone_trainable(self, trainable: bool) -> None:
        self._frozen = not trainable
        self.backbone.requires_grad_(trainable)
        if self._frozen:
            self.backbone.eval()

    def train(self, mode: bool = True) -> "SliceEncoder":
        super().train(mode)
        if self._frozen:
            self.backbone.eval()  # avoid BatchNorm/dropout drift while frozen
        return self

    def _encode_chunk(self, x: Tensor) -> Tensor:
        h = self.backbone(x)
        if isinstance(h, (tuple, list)):
            raise TypeError("backbone must return one [N,C] tensor; set timm num_classes=0/global_pool=avg")
        if h.ndim == 4:
            h = h.mean(dim=(-2, -1))
        if h.ndim != 2 or h.size(0) != x.size(0):
            raise ValueError("backbone feature shape must be [N,C]")
        return h

    def forward(self, images: Tensor, valid: Tensor) -> Tensor:
        """images [B,S,K,3,H,W], valid [B,S,K] -> [B,S,K,d]."""
        if images.ndim != 6 or images.shape[3] != 3:
            raise ValueError("images must have shape [B,S,K,3,H,W]")
        if valid.shape != images.shape[:3] or valid.dtype != torch.bool:
            raise ValueError("valid must be a bool [B,S,K] mask")
        if not valid.any().item():
            raise ValueError("no valid image slices")
        b, s, k = images.shape[:3]
        valid_images = images.reshape(b * s * k, *images.shape[3:])[valid.reshape(-1)]
        if not torch.isfinite(valid_images).all().item():
            raise ValueError("valid MRI slices contain non-finite pixels")
        chunks = []
        for chunk in valid_images.split(self.chunk_size):
            if self._frozen:
                with torch.no_grad():
                    encoded = self._encode_chunk(chunk)
            elif self.training and self.activation_checkpointing:
                encoded = checkpoint(self._encode_chunk, chunk, use_reentrant=False)
            else:
                encoded = self._encode_chunk(chunk)
            chunks.append(self.project(encoded))
        features_valid = torch.cat(chunks, dim=0)
        # Out-of-place indexing ensures autograd routes gradients to the projection.
        all_features = features_valid.new_zeros((b * s * k, features_valid.shape[-1]))
        all_features = all_features.index_copy(0, valid.reshape(-1).nonzero(as_tuple=True)[0], features_valid)
        return all_features.reshape(b, s, k, -1)
