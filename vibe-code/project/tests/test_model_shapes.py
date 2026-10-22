import pytest
import torch
from torch import nn

from oxalpha.models.backbone import SliceEncoder
from oxalpha.models.student import QuartzKneeStudent, StudentConfig, LABELS


class TinyBackbone(nn.Module):
    num_features = 8
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.GELU(), nn.AdaptiveAvgPool2d(1))
        self.seen = 0

    def forward(self, x):
        self.seen += len(x)
        return self.net(x).flatten(1)


def make_model(*, freeze=False, slice_pool='gated', series_pool='gated', checkpointing=False):
    cfg = StudentConfig(d_model=16, heads=4, geometry_dim=6, dropout=0.,
                        slice_pool=slice_pool, series_pool=series_pool)
    backbone = TinyBackbone()
    encoder = SliceEncoder(d_model=16, backbone=backbone, chunk_size=2,
                           freeze_backbone=freeze, activation_checkpointing=checkpointing)
    return QuartzKneeStudent(cfg, encoder=encoder), backbone


def make_inputs(b=2, s=3, k=4, *, device='cpu'):
    x = {
        'images': torch.randn(b, s, k, 3, 12, 12, device=device),
        'slice_mask': torch.ones(b, s, k, device=device, dtype=torch.bool),
        'series_mask': torch.ones(b, s, device=device, dtype=torch.bool),
        'plane': torch.tensor([0, 1, 2][:s], device=device).repeat(b, 1) if s <= 3 else torch.zeros(b, s, dtype=torch.long, device=device),
        'fluid': torch.zeros(b, s, dtype=torch.long, device=device),
        'fat': torch.ones(b, s, dtype=torch.long, device=device),
        'geometry_features': torch.randn(b, s, 6, device=device),
        'geometry_mask': torch.ones(b, s, 6, dtype=torch.bool, device=device),
        'study_ids': [f'study-{i}' for i in range(b)],
    }
    return x


def test_output_shape_and_exact_labels():
    torch.manual_seed(11)
    model, backbone = make_model()
    batch = make_inputs()
    out, att = model(**batch, return_attention=True)
    assert out.shape == (2, 12)
    assert att['slice'].shape == (2, 3, 4, 12)
    assert att['series'].shape == (2, 3, 12)
    assert backbone.seen == 2 * 3 * 4
    assert len(LABELS) == 12 and LABELS[-1] == 'Fracture'
    assert torch.isfinite(out).all()


def test_b1_s1_k1_backward():
    model, backbone = make_model()
    data = make_inputs(b=1, s=1, k=1)
    output = model(**data)
    assert output.shape == (1, 12)
    output.square().mean().backward()
    assert model.classifier_weight.grad is not None
    assert model.slice_mil.queries.grad is not None
    assert model.series_mil.queries.grad is not None
    assert backbone.net[0].weight.grad is not None
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_frozen_backbone_stays_eval_and_does_not_receive_gradients():
    model, backbone = make_model(freeze=True)
    model.train()
    assert not backbone.training
    model(**make_inputs(b=1, s=2, k=2)).square().mean().backward()
    assert backbone.net[0].weight.grad is None
    assert model.encoder.project.weight.grad is not None


@pytest.mark.parametrize('pool', ['gated', 'mean', 'max'])
def test_ablation_pool_modes(pool):
    model, _ = make_model(slice_pool=pool, series_pool=pool)
    logits, att = model(**make_inputs(b=1, s=2, k=3), return_attention=True)
    assert logits.shape == (1, 12)
    assert torch.isfinite(logits).all()
    assert (att['slice'] is None) == (pool != 'gated')
    assert (att['series'] is None) == (pool != 'gated')


def test_checkpointed_encoder_gradients():
    model, backbone = make_model(checkpointing=True)
    model.train()
    model(**make_inputs(b=1, s=2, k=3)).sum().backward()
    assert torch.isfinite(backbone.net[0].weight.grad).all()
