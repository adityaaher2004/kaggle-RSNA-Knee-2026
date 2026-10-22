import random

import pytest
import torch
from torch import nn

from oxalpha.training.checkpoints import (
    save_training_checkpoint, load_training_checkpoint, canonical_hash,
)


def test_checkpoint_resume_weights_optimizer_rng(tmp_path):
    m=nn.Linear(2,1)
    opt=torch.optim.AdamW(m.parameters(),lr=.001)
    sched=torch.optim.lr_scheduler.StepLR(opt,step_size=1)
    torch.manual_seed(9)
    random.seed(10)
    state=m(torch.randn(2,2)).sum()
    state.backward()
    opt.step()
    sched.step()
    expected={k:v.detach().clone() for k,v in m.state_dict().items()}
    prov={'labels':'abc','folds':'def'}
    path=tmp_path/'last.pt'
    save_training_checkpoint(path,model=m,optimizer=opt,scheduler=sched,scaler=None,
                             epoch=3,best_score=.91,best_rule='official_macro_auc',
                             provenance=prov)
    next_torch=torch.randn(2)
    next_py=random.random()
    for p in m.parameters(): p.data.zero_()
    opt2=torch.optim.AdamW(m.parameters(),lr=.001)
    sched2=torch.optim.lr_scheduler.StepLR(opt2,step_size=1)
    epoch, score, rule=load_training_checkpoint(path,model=m,optimizer=opt2,
              scheduler=sched2,scaler=None,expected_provenance=prov)
    assert epoch==4 and score==.91 and rule=='official_macro_auc'
    assert all(torch.equal(expected[k],v) for k,v in m.state_dict().items())
    assert torch.equal(torch.randn(2),next_torch)
    assert random.random()==next_py
    with pytest.raises(ValueError,match='provenance'):
        load_training_checkpoint(path,model=m,optimizer=opt2,scheduler=sched2,
                                 scaler=None,expected_provenance={'labels':'changed'})


def test_canonical_hash_independent_key_order():
    assert canonical_hash({'b':2,'a':1})==canonical_hash({'a':1,'b':2})
