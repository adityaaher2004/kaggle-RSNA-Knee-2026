import numpy as np
import pytest
import torch
from torch import nn

from oxalpha.models.student import LABELS
from oxalpha.losses.masked import masked_weighted_bce
from oxalpha.training.engine import train_epoch, validate_epoch, masked_auc_report


class TinyStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(12))
        self.scale = nn.Parameter(torch.ones(12) * 0.05)

    def forward(self, images, slice_mask, series_mask, plane, fluid, fat,
                geometry_features, geometry_mask, study_ids=None):
        assert images.ndim == 6
        image_mean = images.mean(dim=(1,2,3,4,5))
        return self.bias[None,:] + self.scale[None,:] * image_mean[:,None]


def batch(value, labels, weights, uid='u'):
    x = {
        'images': np.full((1,1,1,3,4,4), value, np.float32),
        'slice_mask': np.ones((1,1,1), bool),
        'series_mask': np.ones((1,1), bool),
        'plane': np.zeros((1,1), np.int64),
        'fluid': np.zeros((1,1), np.int64),
        'fat': np.zeros((1,1), np.int64),
        'geometry_features': np.zeros((1,1,6), np.float32),
        'geometry_mask': np.zeros((1,1,6), bool),
        'study_ids': [uid],
    }
    return x, np.asarray(labels, np.float32).reshape(1,12), np.asarray(weights, np.float32).reshape(1,12)


def make_y(pos, neg=None):
    y = [float('nan')] * 12
    y[pos] = 1.
    if neg is not None:
        y[neg] = 0.
    return y


def make_w(idx):
    w = [0.] * 12
    for i,v in idx.items():
        w[i]=v
    return w


def test_exact_gradient_accumulation_matches_full_weighted_objective():
    m=TinyStudent()
    initial={k:v.detach().clone() for k,v in m.state_dict().items()}
    data=[batch(1.,make_y(0),make_w({0:1}),uid='1'),
          batch(2.,make_y(0),make_w({0:.2}),uid='2'),
          batch(3.,make_y(1),make_w({1:0}),uid='3'),
          batch(4.,make_y(0),make_w({0:.9}),uid='4')]
    optim=torch.optim.SGD(m.parameters(),lr=.1)
    stats=train_epoch(m, data, optim, device='cpu', amp=False, accumulate=4, grad_clip=0)
    assert stats.optimizer_steps==1 and stats.empty_batches==1
    assert stats.supervised_weight == pytest.approx(2.1,abs=1e-6)
    assert stats.label_coverage[0]==3
    other=TinyStudent()
    other.load_state_dict(initial)
    z=[]; y=[]; w=[]
    for x,t,c in data:
        z.append(other(**{k: torch.as_tensor(v) if k != 'study_ids' else v for k,v in x.items()}))
        y.append(torch.tensor(t))
        w.append(torch.tensor(c))
    loss=masked_weighted_bce(torch.cat(z),torch.cat(y),torch.cat(w)).loss
    loss.backward()
    optim2=torch.optim.SGD(other.parameters(),lr=.1)
    optim2.step()
    for k,v in m.state_dict().items():
        assert torch.allclose(v, other.state_dict()[k], atol=1e-7), k


def test_all_missing_skips_optimizer_step():
    m=TinyStudent()
    before={k:p.detach().clone() for k,p in m.state_dict().items()}
    optim=torch.optim.AdamW(m.parameters(),lr=.1,weight_decay=.1)
    data=[batch(1.,[float('nan')]*12,[0.]*12)]
    stats=train_epoch(m,data,optim,device='cpu',amp=False,accumulate=1)
    assert stats.optimizer_steps==0 and stats.loss is None
    for k,v in m.state_dict().items():
        assert torch.equal(v,before[k])


def test_auc_no_silent_official_macro_when_single_class():
    y=np.full((3,12),np.nan)
    p=np.full((3,12),.5)
    w=np.zeros((3,12))
    y[:,0]=[0,1,0]
    p[:,0]=[.1,.9,.2]
    w[:,0]=1
    y[:,1]=[1,1,1]
    w[:,1]=1
    metrics=masked_auc_report(y,p,w)
    assert metrics['official_macro_auc'] is None
    assert metrics['valid_labels']==1 and metrics['partial_macro_auc']==1
    assert metrics['per_label']['MCL']['auc'] is None


def test_tied_auc_is_half():
    y=np.full((2,12),np.nan)
    p=np.ones_like(y)*.5
    w=np.zeros_like(y)
    y[:,0]=[0,1]
    w[:,0]=1
    assert masked_auc_report(y,p,w)['partial_macro_auc']==.5


def test_validation_does_not_update_model():
    m=TinyStudent()
    before={k:p.detach().clone() for k,p in m.state_dict().items()}
    stats, report=validate_epoch(m,[batch(1.,make_y(0),make_w({0:1}))],device='cpu',amp=False)
    assert stats.supervised_cells==1
    assert report['valid_labels']==0
    assert all(torch.equal(p,before[k]) for k,p in m.state_dict().items())
