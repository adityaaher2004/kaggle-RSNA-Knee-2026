import csv
from pathlib import Path

import pytest

from oxalpha.models.student import LABELS
from oxalpha.training.records import (
    load_weak_labels, load_folds, read_gold_ids, training_partition,
)


def write_csv(path, rows):
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_labels(uid, value='1'):
    return {'StudyInstanceUID':uid, **{name:value for name in LABELS}}


def test_status_uncertain_and_notmentioned_do_not_become_negative(tmp_path):
    row = make_labels('a')
    row['ACL__status'] = 'uncertain'
    row['MCL__status'] = 'negative'
    row['MCL'] = '0'
    row['Medial Meniscus__confidence'] = '0'
    # DictWriter must include all optional headers on all rows; here only one row.
    path = tmp_path/'weak.csv'
    write_csv(path,[row])
    record = load_weak_labels(path,gold_ids={'gold'})['a']
    assert record.confidence[0] == 0
    assert record.confidence[1] == 1
    assert record.targets[1] == 0
    assert record.confidence[2] == 0
    assert record.confidence[3] == 1


def test_gold_must_be_excluded(tmp_path):
    path = tmp_path/'weak.csv'
    write_csv(path,[make_labels('gold')])
    with pytest.raises(ValueError, match='gold'):
        load_weak_labels(path,gold_ids={'gold'})
    with pytest.raises(ValueError, match='mandatory'):
        load_weak_labels(path,gold_ids=set())


def test_contradictory_status_rejected(tmp_path):
    row=make_labels('a')
    row['ACL__status']='negative'
    path=tmp_path/'weak.csv'
    write_csv(path,[row])
    with pytest.raises(ValueError,match='contradictory'):
        load_weak_labels(path,gold_ids={'gold'})


def test_group_cross_fold_rejected(tmp_path):
    folds=tmp_path/'folds.csv'
    write_csv(folds,[{'StudyInstanceUID':'a','split':'train','PatientGroup':'p1'},
                     {'StudyInstanceUID':'b','split':'val','PatientGroup':'p1'}])
    with pytest.raises(ValueError,match='leakage'):
        load_folds(folds,gold_ids={'gold'})


def test_fold_and_label_partition(tmp_path):
    lpath=tmp_path/'weak.csv'
    write_csv(lpath,[make_labels('a'),make_labels('b')])
    fpath=tmp_path/'folds.csv'
    write_csv(fpath,[{'StudyInstanceUID':'a','split':'train','group_id':'g1'},
                     {'StudyInstanceUID':'b','split':'val','group_id':'g2'}])
    gold=tmp_path/'gold.txt'
    gold.write_text('gold\n')
    ids=read_gold_ids(gold)
    labels=load_weak_labels(lpath,gold_ids=ids)
    folds=load_folds(fpath,gold_ids=ids)
    assert training_partition(labels, folds, {'a','b'},gold_ids=ids)==(['a'],['b'])
    with pytest.raises(ValueError,match='unmatched'):
        training_partition(labels, folds, {'a'},gold_ids=ids)


def test_duplicate_uids_hard_error(tmp_path):
    p=tmp_path/'weak.csv'
    write_csv(p,[make_labels('a'),make_labels('a')])
    with pytest.raises(ValueError,match='duplicate'):
        load_weak_labels(p,gold_ids={'gold'})
