import inspect
import pytest
import torch

from test_model_shapes import make_inputs, make_model
from oxalpha.models.student import QuartzKneeStudent


def test_forward_api_only_contains_declared_inference_inputs():
    parameters = inspect.signature(QuartzKneeStudent.forward).parameters
    assert set(parameters) == {'self', 'images', 'slice_mask', 'series_mask', 'plane',
                               'fluid', 'fat', 'geometry_features', 'geometry_mask',
                               'study_ids', 'return_attention'}
    for forbidden in ('report', 'text', 'y', 'labels', 'teacher_logits', 'weak_labels'):
        assert forbidden not in parameters


def test_extra_text_or_supervision_fields_rejected():
    model, _ = make_model()
    data = make_inputs(b=1, s=1, k=1)
    for key in ('Report', 'text_embedding', 'labels', 'teacher_logits', 'y'):
        with pytest.raises(TypeError, match='unexpected keyword'):
            model(**{**data, key: torch.ones(1)})


def test_study_ids_not_used_as_features():
    model, _ = make_model()
    model.eval()
    data = make_inputs(b=1, s=1, k=2)
    with torch.no_grad():
        a = model(**data)
        data['study_ids'] = ['another-patient-or-study-id']
        b = model(**data)
    assert torch.equal(a, b)


def test_numpy_collate_adapter_allows_only_inference_fields():
    import numpy as np
    from oxalpha.models.student import prepare_student_batch
    data = make_inputs(b=1, s=1, k=2)
    np_data = {k: (v.numpy() if torch.is_tensor(v) else v) for k, v in data.items()}
    result = prepare_student_batch(np_data)
    assert result['images'].dtype == torch.float32
    assert result['slice_mask'].dtype == torch.bool
    assert result['plane'].dtype == torch.long
    model, _ = make_model()
    assert model(**result).shape == (1, 12)
    with pytest.raises(ValueError, match='non-inference batch keys'):
        prepare_student_batch({**np_data, 'labels': np.zeros((1, 12))})
