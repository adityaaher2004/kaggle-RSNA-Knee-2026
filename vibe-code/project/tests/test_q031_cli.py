"""Full CLI run using synthetic MRI bags, never touching real DICOM or downloading weights."""
import csv
import importlib.util
import json
import sys

import numpy as np
import torch
from torch import nn

from oxalpha.models.student import LABELS


def load_cli():
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/'scripts'/'train_student.py'
    spec=importlib.util.spec_from_file_location('train_student_q031',path)
    mod=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=mod
    spec.loader.exec_module(mod)
    return mod


class FakeBag:
    def __init__(self, records, **kwargs):
        self.study_ids=sorted({r.study_uid for r in records})
        self.epoch=0
    def __len__(self): return len(self.study_ids)
    def set_epoch(self,epoch): self.epoch=epoch
    def __getitem__(self,index):
        study=self.study_ids[index]
        return {
            'images':np.ones((1,1,3,8,8),np.float32)*(index+1),
            'slice_mask':np.ones((1,1),bool),
            'series_mask':np.ones(1,bool),
            'plane':np.zeros(1,np.int64),
            'fluid':np.zeros(1,np.int64),
            'fat':np.zeros(1,np.int64),
            'geometry_features':np.zeros((1,6),np.float32),
            'geometry_mask':np.zeros((1,6),bool),
            'study_ids':study,
        }


class FakeStudent(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg=cfg
        self.head=nn.Linear(1,12)
    def forward(self,images,slice_mask,series_mask,plane,fluid,fat,
                geometry_features,geometry_mask,study_ids=None):
        return self.head(images.mean(dim=(1,2,3,4,5)).reshape(-1,1))


def _csv(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def test_end_to_end_cli_with_resume(tmp_path,monkeypatch):
    cli=load_cli()
    monkeypatch.setattr(cli,'StudyBagDataset',FakeBag)
    monkeypatch.setattr(cli,'QuartzKneeStudent',FakeStudent)
    dat=tmp_path/'data';dat.mkdir()
    _csv(dat/'train_series.csv',[{'StudyInstanceUID':u,'SeriesInstanceUID':'s'+u}
                                   for u in ('a','b','c','d')])
    weak=tmp_path/'weak.csv'
    _csv(weak,[{'StudyInstanceUID':u,**{l:str((i+j)%2) for j,l in enumerate(LABELS)}}
               for i,u in enumerate(('a','b','c','d'))])
    folds=tmp_path/'folds.csv'
    _csv(folds,[{'StudyInstanceUID':u,'split':'train' if u in ('a','b') else 'val','group_id':'group'+u}
                for u in ('a','b','c','d')])
    gold=tmp_path/'gold.txt';gold.write_text('gold1\ngold2\n')
    weights=tmp_path/'weights.pth';weights.write_bytes(b'dummy_checkpoint_only_for_fake_model')
    out=tmp_path/'work'/'run'
    args=['--data-root',str(dat),'--weak-labels',str(weak),'--folds',str(folds),
          '--gold-ids',str(gold),'--out-dir',str(out),
          '--epochs','2','--warmup-epochs','0','--batch-size','1',
          '--accumulate','2','--backbone-weights',str(weights),'--cpu']
    assert cli.main(args)==0
    assert (out/'last.pt').is_file() and (out/'best_val.pt').is_file()
    assert (out/'provenance.json').is_file() and (out/'coverage.json').is_file()
    assert set((out/'train_seen_ids.txt').read_text().splitlines())=={'a','b'}
    assert len((out/'history.jsonl').read_text().splitlines())==2
    summary=json.loads((out/'metrics.json').read_text())
    assert summary['epoch']==1
    assert summary['weak_validation_metrics']['official_macro_auc'] is not None
    # No more epochs, but checkpoint full restore and provenance verification exercise.
    assert cli.main(args+['--resume',str(out/'last.pt')])==0


def test_cli_rejects_unprotected_training(tmp_path):
    cli=load_cli()
    try:
        cli._args(['--data-root','x','--weak-labels','y','--folds','z',
                   '--out-dir','work/run','--pretrained'])
    except SystemExit as e:
        assert e.code==2
    else:
        raise AssertionError('Gold ID list must be mandatory')
