import pytest
import torch

from test_model_shapes import make_inputs, make_model
from oxalpha.models.mil import masked_softmax
from oxalpha.models.student import InvalidStudyError


def test_all_masked_softmax_is_safe_with_gradients():
    x = torch.randn(2, 4, 3, requires_grad=True)
    mask = torch.tensor([[[True], [False], [False], [False]],
                         [[False], [False], [False], [False]]])
    y = masked_softmax(x, mask, dim=1)
    assert torch.allclose(y[1], torch.zeros_like(y[1]))
    assert torch.allclose(y[0].sum(0), torch.ones(3))
    y.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_padded_slices_and_series_never_encoded_or_attended():
    torch.manual_seed(13)
    model, backbone = make_model()
    model.eval()
    data = make_inputs(b=1, s=3, k=4)
    data['series_mask'][0, 2] = False
    data['slice_mask'][0, 2] = False
    data['slice_mask'][0, 0, 2:] = False
    data['images'][0, 2] = float('nan')
    data['images'][0, 0, 2:] = float('nan')
    data['geometry_features'][0, 2] = float('nan')
    logits, att = model(**data, return_attention=True)
    assert backbone.seen == 4 + 2
    assert torch.isfinite(logits).all()
    assert (att['slice'][0, 2] == 0).all()
    assert (att['slice'][0, 0, 2:] == 0).all()
    assert (att['series'][0, 2] == 0).all()
    assert torch.allclose(att['slice'][0, 0, :2].sum(dim=0), torch.ones(12), atol=1e-6)
    assert torch.allclose(att['series'][0, :2].sum(dim=0), torch.ones(12), atol=1e-6)
    # Poison invalid inputs with arbitrary finite pixel values. Output is unchanged.
    data['images'][0, 2] = torch.randn_like(data['images'][0, 2]) * 900
    data['images'][0, 0, 2:] = 999.
    data['geometry_features'][0, 2] = 999.
    again = model(**data)
    assert torch.allclose(logits, again, atol=1e-5)


def test_invalid_study_raises_recoverable_error():
    model, _ = make_model()
    data = make_inputs(b=2, s=2, k=2)
    data['series_mask'][1] = False
    data['slice_mask'][1] = False
    with pytest.raises(InvalidStudyError):
        model(**data)


def test_masks_must_be_consistent():
    model, _ = make_model()
    data = make_inputs(b=1, s=2, k=2)
    data['series_mask'][0, 0] = False
    with pytest.raises(ValueError, match='invalid series'):
        model(**data)
    data['slice_mask'][0, 0] = False
    data['slice_mask'][0, 1] = False
    with pytest.raises(ValueError, match='valid series'):
        model(**data)


def test_masked_inputs_receive_no_image_gradients():
    model, _ = make_model()
    data = make_inputs(b=1, s=2, k=2)
    data['images'].requires_grad_(True)
    data['slice_mask'][0, 0, 1] = False
    model(**data).sum().backward()
    assert data['images'].grad is not None
    assert torch.all(data['images'].grad[0, 0, 1] == 0)


def test_independent_fluid_fat_metadata():
    model, _ = make_model()
    model.eval()
    data = make_inputs(b=1, s=2, k=2)
    data['fluid'][0, 0] = 1
    data['fat'][0, 0] = 0
    assert torch.isfinite(model(**data)).all()
    data['fat'][0, 0] = 1
    # Compare context explicitly: independent embeddings must affect tokens.
    with torch.no_grad():
        a = model.context.fluid(torch.tensor(1)) + model.context.fat(torch.tensor(0))
        b = model.context.fluid(torch.tensor(1)) + model.context.fat(torch.tensor(1))
    assert not torch.equal(a, b)


def test_series_order_is_permutation_invariant_at_inference():
    torch.manual_seed(19)
    model, _ = make_model()
    model.eval()
    data = make_inputs(b=1, s=3, k=2)
    with torch.no_grad():
        base = model(**data)
        reordered = dict(data)
        idx = torch.tensor([2, 0, 1])
        for key in ('images', 'slice_mask', 'series_mask', 'plane', 'fluid', 'fat',
                    'geometry_features', 'geometry_mask'):
            reordered[key] = data[key].index_select(1, idx)
        alternate = model(**reordered)
    assert torch.allclose(base, alternate, atol=1e-6, rtol=1e-6)
