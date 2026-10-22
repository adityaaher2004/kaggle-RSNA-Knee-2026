"""MRI metadata embeddings and one shared masked series transformer."""
from __future__ import annotations

import torch
from torch import Tensor, nn


class SeriesContext(nn.Module):
    def __init__(self, d_model: int = 256, geometry_dim: int = 6,
                 heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be divisible by transformer heads")
        if geometry_dim <= 0:
            raise ValueError("geometry_dim must be positive")
        self.geometry_dim = geometry_dim
        self.plane = nn.Embedding(4, d_model)
        self.fluid = nn.Embedding(3, d_model)
        self.fat = nn.Embedding(3, d_model)
        # Explicitly represent which of the G geometry values are missing.
        self.geometry = nn.Sequential(nn.Linear(2 * geometry_dim, d_model),
                                      nn.GELU(), nn.Linear(d_model, d_model))
        block = nn.TransformerEncoderLayer(d_model=d_model, nhead=heads,
                                            dim_feedforward=4 * d_model,
                                            dropout=dropout, activation="gelu",
                                            batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(block, num_layers=1, enable_nested_tensor=False)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, summary: Tensor, series_mask: Tensor, plane: Tensor,
                fluid: Tensor, fat: Tensor, geometry_features: Tensor,
                geometry_mask: Tensor) -> Tensor:
        # Handle nonfinite *invalid* metadata by masking before the MLP.
        # Nonfinite valid entries should be caught by the student validator.
        numeric = torch.where(geometry_mask, geometry_features, torch.zeros_like(geometry_features))
        numeric = torch.where(series_mask[..., None], numeric, torch.zeros_like(numeric))
        inputs = summary + self.plane(plane) + self.fluid(fluid) + self.fat(fat)
        inputs = inputs + self.geometry(torch.cat((numeric, geometry_mask.to(summary.dtype)), dim=-1))
        inputs = inputs.masked_fill(~series_mask[..., None], 0.)
        output = self.transformer(inputs, src_key_padding_mask=~series_mask)
        return self.output_norm(output).masked_fill(~series_mask[..., None], 0.)
