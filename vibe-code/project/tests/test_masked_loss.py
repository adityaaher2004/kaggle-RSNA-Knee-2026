"""Q-031 acceptance: perfect/no-label/rare-positive, zero gradients, confidence."""
import math

import pytest
import torch
import torch.nn.functional as F

from oxalpha.losses.masked import masked_weighted_bce


def test_perfect_logits_have_near_zero_loss():
    z = torch.tensor([[20., -20., 20.], [-20., 20., -20.]], requires_grad=True)
    y = torch.tensor([[1., 0., 1.], [0., 1., 0.]])
    r = masked_weighted_bce(z, y)
    assert r.has_supervision and r.supervised_cells == 6
    assert r.loss.item() < 1e-8
    r.loss.backward()
    assert torch.isfinite(z.grad).all()


def test_all_unknown_returns_zero_and_no_gradient():
    z = torch.randn(2, 12, requires_grad=True)
    y = torch.full((2, 12), float('nan'))
    r = masked_weighted_bce(z, y)
    assert not r.has_supervision and r.supervised_cells == 0 and r.loss.item() == 0.
    r.loss.backward()
    assert torch.equal(z.grad, torch.zeros_like(z))


def test_masked_cells_have_exact_zero_gradient():
    z = torch.tensor([[.3, -.8, .7, .2, .4]], requires_grad=True)
    y = torch.tensor([[1., 0., float('nan'), 0., 1.]])
    w = torch.tensor([[1., .0, 1., .3, 0.]])
    r = masked_weighted_bce(z, y, w)
    r.loss.backward()
    assert z.grad[0, 0] != 0
    assert z.grad[0, 3] != 0
    assert z.grad[0, 1].item() == 0
    assert z.grad[0, 2].item() == 0
    assert z.grad[0, 4].item() == 0


def test_weighted_average_correctly_normalized():
    z = torch.tensor([[0., 1., -2.]])
    y = torch.tensor([[1., 0., 1.]])
    w = torch.tensor([[.5, 1., .25]])
    r = masked_weighted_bce(z, y, w)
    expected = (F.binary_cross_entropy_with_logits(z, y, reduction='none') * w).sum() / w.sum()
    assert torch.allclose(r.loss, expected, atol=1e-7)


def test_rare_positive_with_pos_weight():
    z = torch.zeros(3, 2, requires_grad=True)
    y = torch.tensor([[1., float('nan')], [0., 0.], [0., 0.]])
    pw = torch.tensor([3., 2.])
    r = masked_weighted_bce(z, y, pos_weight=pw)
    assert r.supervised_cells == 5
    assert r.label_coverage.tolist() == [3, 2]
    r.loss.backward()
    assert z.grad[0, 0] < 0  # rare positive has useful descent gradient
    assert z.grad[0, 1].item() == 0


def test_zero_confidence_all_does_not_step():
    z = torch.randn(2, 3, requires_grad=True)
    r = masked_weighted_bce(z, torch.ones_like(z), torch.zeros_like(z))
    assert not r.has_supervision and r.weight_sum.item() == 0


def test_confidence_clipped_without_floors():
    z = torch.zeros(1, 3)
    y = torch.ones_like(z)
    r = masked_weighted_bce(z, y, torch.tensor([[0., .1, 3.]]))
    assert r.supervised_cells == 2
    assert r.weight_sum.item() == pytest.approx(1.1, abs=1e-6)


def test_active_negative_or_nan_confidence_rejected():
    z = torch.zeros(1, 2)
    y = torch.ones_like(z)
    with pytest.raises(ValueError):
        masked_weighted_bce(z, y, torch.tensor([[1., -1.]]))
    with pytest.raises(ValueError):
        masked_weighted_bce(z, y, torch.tensor([[1., float('nan')]]))


def test_malformed_tensor_rejected():
    with pytest.raises(ValueError):
        masked_weighted_bce(torch.zeros(2, 12), torch.zeros(1, 12))
    with pytest.raises(ValueError):
        masked_weighted_bce(torch.zeros(2, 12), torch.zeros(2, 12), pos_weight=torch.ones(3))
    with pytest.raises(ValueError):
        masked_weighted_bce(torch.zeros(2, 12), torch.zeros(2, 12), observed=torch.ones(2, 12))
