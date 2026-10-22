"""Soft-label regression tests for blended Glimmer supervision (Q-031)."""
import csv
import importlib.util
import json
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from oxalpha.losses.masked import masked_weighted_bce
from oxalpha.models.student import LABELS
from oxalpha.training.engine import masked_auc_report
from oxalpha.training.records import load_weak_labels


def _save_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _weak(uid='study', **overrides):
    row = {'StudyInstanceUID': uid, **{name: '0.5' for name in LABELS}}
    row.update(overrides)
    return row


def test_blended_csv_probabilities_preserved_without_rebinning(tmp_path):
    path = tmp_path / 'blend.csv'
    _save_csv(path, [_weak(ACL='0.25', MCL='0.75', Effusion='0', Fracture='1')])
    r = load_weak_labels(path, gold_ids={'gold'})['study']
    assert r.targets[:2] == (0.25, 0.75)
    assert r.confidence[:2] == (1., 1.)
    assert r.targets[LABELS.index('Effusion')] == 0.
    assert r.targets[LABELS.index('Fracture')] == 1.


def test_missing_and_zero_confidence_are_still_masked(tmp_path):
    path = tmp_path / 'blend.csv'
    row = _weak(ACL='0.25', MCL='0.75', Fracture='')
    row.update({'ACL__confidence': '0', 'MCL__status': 'uncertain', 'Fracture__status': 'not_processed'})
    _save_csv(path, [row])
    r = load_weak_labels(path, gold_ids={'gold'})['study']
    assert r.targets[0] == .25 and r.confidence[0] == 0.
    assert np.isnan(r.targets[1]) and r.confidence[1] == 0.
    assert np.isnan(r.targets[-1]) and r.confidence[-1] == 0.


@pytest.mark.parametrize('bad', ['1.01', '-0.01', 'inf', '-inf', 'x', '1e309'])
def test_out_of_range_or_malformed_probability_fails(tmp_path, bad):
    path = tmp_path / 'bad.csv'
    _save_csv(path, [_weak(ACL=bad)])
    with pytest.raises(ValueError, match='weak label'):
        load_weak_labels(path, gold_ids={'gold'})


def test_soft_value_and_status_can_coexist_but_hard_contradiction_fails(tmp_path):
    path = tmp_path / 'blend.csv'
    row = _weak(ACL='0.25')
    row['ACL__status'] = 'positive'
    _save_csv(path, [row])
    r = load_weak_labels(path, gold_ids={'gold'})['study']
    assert r.targets[0] == .25
    row['ACL'] = '0'
    _save_csv(path, [row])
    with pytest.raises(ValueError, match='contradictory'):
        load_weak_labels(path, gold_ids={'gold'})


def test_soft_bce_exact_values_and_gradients():
    z = torch.tensor([[0., -0.4, 2.]], requires_grad=True)
    y = torch.tensor([[.25, .75, 1.]])
    w = torch.tensor([[.1, .3, .2]])
    result = masked_weighted_bce(z, y, w)
    expected = (F.binary_cross_entropy_with_logits(z, y, reduction='none') * w).sum() / w.sum()
    assert torch.allclose(result.loss, expected, atol=1e-7)
    assert result.supervised_cells == 3
    result.loss.backward()
    assert torch.allclose(z.grad, (torch.sigmoid(z.detach()) - y) * (w / w.sum()), atol=1e-7)


def test_soft_target_tiny_confidence_is_normalized_not_scaled_down():
    z = torch.tensor([[1.]], requires_grad=True)
    y = torch.tensor([[.25]])
    result = masked_weighted_bce(z, y, torch.tensor([[.01]]))
    assert torch.allclose(result.loss, F.binary_cross_entropy_with_logits(z, y))
    result.loss.backward()
    assert z.grad.item() == pytest.approx(torch.sigmoid(z.detach()).item() - .25, abs=1e-6)


def test_soft_and_missing_cells_have_exact_zero_gradient():
    z = torch.tensor([[1., 1., 1., 1., 1.]], requires_grad=True)
    y = torch.tensor([[.25, .75, float('nan'), -1., 1.3]])
    with pytest.raises(ValueError, match='targets must'):
        masked_weighted_bce(z, y)
    y[0, 4] = .5
    w = torch.tensor([[1., 0., 1., 1., 0.]])
    result = masked_weighted_bce(z, y, w)
    result.loss.backward()
    assert result.supervised_cells == 1
    assert z.grad[0, 0] != 0
    assert torch.count_nonzero(z.grad[0, 1:]) == 0


def test_soft_targets_are_not_thresholded_into_auc():
    y = np.full((3, 12), np.nan)
    p = np.full((3, 12), .5)
    w = np.zeros((3, 12))
    y[:, 0] = [0, 1, .25]
    p[:, 0] = [.1, .9, .2]
    w[:, 0] = 1
    r = masked_auc_report(y, p, w)
    assert r['per_label'][LABELS[0]]['auc'] == 1.
    assert r['per_label'][LABELS[0]]['soft_excluded_from_auc'] == 1
    assert r['soft_targets_excluded_from_auc'] == 1
    assert r['official_macro_auc'] is None
    assert 'Soft' in r['warning']


def test_all_soft_validation_keeps_auc_undefined():
    y = np.full((2, 12), .25)
    p = np.full((2, 12), .5)
    w = np.ones((2, 12))
    r = masked_auc_report(y, p, w)
    assert r['official_macro_auc'] is None
    assert r['partial_macro_auc'] is None
    assert r['soft_targets_excluded_from_auc'] == 24


def test_soft_label_coverage_and_training_gate(tmp_path):
    # Coverage must treat soft targets as trainable, not require hard positives/negatives.
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / 'scripts' / 'train_student.py'
    spec = importlib.util.spec_from_file_location('train_student_soft', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    weak = tmp_path / 'soft.csv'
    _save_csv(weak, [_weak('study', ACL='.25')])
    parsed = load_weak_labels(weak, gold_ids={'gold'})
    coverage = mod._coverage(parsed, ['study'])
    assert coverage['ACL']['soft'] == 1
    assert coverage['ACL']['pos'] == coverage['ACL']['neg'] == 0
    assert coverage['ACL']['expected_positives'] == pytest.approx(.25)


def test_all_soft_training_cli_saves_checkpoint(tmp_path, monkeypatch):
    # Exercise the full trainer acceptance gates, checkpoint rule and metadata.
    from pathlib import Path
    import importlib.util
    old_path = Path(__file__).resolve().parent / 'test_q031_cli.py'
    spec = importlib.util.spec_from_file_location('_q031_cli_helpers', old_path)
    helpers = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = helpers
    spec.loader.exec_module(helpers)
    cli = helpers.load_cli()
    monkeypatch.setattr(cli, 'StudyBagDataset', helpers.FakeBag)
    monkeypatch.setattr(cli, 'QuartzKneeStudent', helpers.FakeStudent)
    data_root = tmp_path / 'data'
    data_root.mkdir()
    ids = ['a', 'b', 'c', 'd']
    _save_csv(data_root / 'train_series.csv', [
        {'StudyInstanceUID': u, 'SeriesInstanceUID': 's' + u} for u in ids
    ])
    weak = tmp_path / 'llm_labels_v4_blend.csv'
    _save_csv(weak, [
        {'StudyInstanceUID': u, **{l: str(.25 if (i + j) % 2 else .75)
                                   for j, l in enumerate(LABELS)}}
        for i, u in enumerate(ids)
    ])
    folds = tmp_path / 'folds.csv'
    _save_csv(folds, [
        {'StudyInstanceUID': u,
         'split': 'train' if u in ('a', 'b') else 'val',
         'group_id': 'group' + u} for u in ids
    ])
    gold = tmp_path / 'gold.txt'
    gold.write_text('protected_gold\n')
    weights = tmp_path / 'fake_model_weights.pth'
    weights.write_bytes(b'fake_backbone_for_injected_model')
    out = tmp_path / 'work' / 'soft_only'
    args = [
        '--data-root', str(data_root), '--weak-labels', str(weak),
        '--folds', str(folds), '--gold-ids', str(gold),
        '--out-dir', str(out), '--epochs', '1', '--warmup-epochs', '0',
        '--batch-size', '1', '--accumulate', '2',
        '--backbone-weights', str(weights), '--cpu',
    ]
    assert cli.main(args) == 0
    assert (out / 'last.pt').exists()
    result = json.loads((out / 'metrics.json').read_text())
    assert result['train']['supervised_cells'] == 24
    assert result['best_rule'] == 'negative_val_loss'
    assert result['weak_validation_metrics']['official_macro_auc'] is None
    assert result['weak_validation_metrics']['soft_targets_excluded_from_auc'] == 24
    provenance = json.loads((out / 'provenance.json').read_text())
    assert provenance['target_semantics'] == 'soft_bernoulli_bce_v2'
