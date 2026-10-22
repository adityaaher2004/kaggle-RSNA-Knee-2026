import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from oxalpha.data.dataset import (SeriesRecord, select_series, assemble_study,
                                  preprocess_study, StudyBagDataset, collate_studies)
from oxalpha.data.preprocessing import PreprocessConfig, SeriesTensor


def record(n, plane='Sagittal', fluid=0, fat=0, count=24, study='study'):
    return SeriesRecord(study, f'series-{n:02d}', Path('/unused') / f'{n}', plane,
                        fluid, fat, count)


def fake_tensor(cfg, *, nonzero=True):
    image = np.ones((cfg.slices, 3, cfg.size, cfg.size), dtype=np.float32)
    mask = np.ones(cfg.slices, bool) if nonzero else np.zeros(cfg.slices, bool)
    return SeriesTensor(image, mask, np.ones((cfg.slices,cfg.size,cfg.size),bool),
                        np.arange(cfg.slices), metadata={'pixel_spacing_mm': (.3,.4),
                        'slice_thickness_mm':3., 'original_height_width':(448,512)})


def test_retains_three_planes_and_caps():
    data = [record(i, ('Sagittal','Coronal','Axial')[i%3], i%2, (i+1)%2) for i in range(15)]
    picked, excluded = select_series(data, max_series=8)
    assert len(picked) == 8 and len(excluded) == 7
    assert {p.plane for p in picked} == {'Sagittal','Coronal','Axial'}
    assert set(excluded).isdisjoint({p.series_uid for p in picked})


def test_inference_permutation_seed_invariant():
    data = [record(i, ('Sagittal','Coronal','Axial')[i%3]) for i in range(15)]
    original = [r.series_uid for r in select_series(data)[0]]
    for seed in range(5):
        assert [r.series_uid for r in select_series(list(reversed(data)), seed=seed)[0]] == original


def test_independent_fluid_and_fat_unknown():
    row = {'StudyInstanceUID':'a', 'SeriesInstanceUID':'b',
           'Anatomical_Plane':'Coronal', 'Fluid_Sensitive':'1', 'Fat_Suppression':'0'}
    r = SeriesRecord.from_row(row, '/tmp/train_series')
    assert (r.fluid,r.fat,r.plane) == (1,0,'Coronal')
    r2 = SeriesRecord.from_row({**row, 'Fat_Suppression':float('nan'), 'Fluid_Sensitive':'No'},'/tmp')
    assert (r2.fluid,r2.fat) == (0,2)


def test_pad_masks_and_metadata():
    cfg = PreprocessConfig(size=16,slices=3)
    records = [record(0, 'Coronal', 1, 0)]
    out = assemble_study('study', records, {records[0].series_uid:fake_tensor(cfg)},
                         max_series=4,cfg=cfg)
    assert out.images.shape == (4,3,3,16,16)
    assert out.series_mask.tolist() == [True,False,False,False]
    assert out.slice_mask[1:].sum() == 0 and not np.any(out.images[1:])
    assert out.plane.tolist() == [1,3,3,3]
    assert out.fluid.tolist() == [1,2,2,2] and out.fat.tolist() == [0,2,2,2]
    assert out.geometry_mask[0].tolist() == [True,True,True,False,True,True]
    assert not out.geometry_mask[1:].any()
    assert set(out.inference_inputs()) == {'images','slice_mask','series_mask','plane','fluid','fat',
                                            'geometry_features','geometry_mask','study_ids'}


def test_training_reproducible_with_seed_and_plane_coverage():
    data = [record(i, ('Sagittal','Coronal','Axial')[i%3], count=28) for i in range(30)]
    s1 = select_series(data,training=True,seed=12)
    assert s1 == select_series(list(reversed(data)),training=True,seed=12)
    assert {r.plane for r in s1[0]} == {'Sagittal','Coronal','Axial'}
    variations = {tuple(r.series_uid for r in select_series(data,training=True,seed=k,randomness=1)[0]) for k in range(12)}
    assert len(variations) > 1


def test_invalid_duplicate_and_empty():
    assert select_series([]) == ([],[])
    with pytest.raises(ValueError): select_series([record(0),record(0)])
    with pytest.raises(ValueError): select_series([record(0),record(1,study='other')])
    with pytest.raises(ValueError): select_series([record(0)],max_series=0)


def test_preprocess_fallback_after_reader_failure():
    cfg=PreprocessConfig(size=8,slices=2)
    data=[record(0,'Sagittal',count=100),record(1,'Sagittal',count=10),record(2,'Coronal'),record(3,'Axial')]
    def reader(rec):
        if rec.series_uid == 'series-00':
            raise OSError('simulated corrupt DICOM')
        return rec
    out=preprocess_study('study',data,series_reader=reader,
                         preprocessor=lambda raw,cfg,**kw:fake_tensor(cfg),
                         preprocess_config=cfg,max_series=3)
    assert out.series_mask.sum() == 3
    assert 'series-00' in out.failures
    assert {data[int(x.split('-')[-1])].plane for x in out.selected_series_uids} == {'Sagittal','Coronal','Axial'}


def test_dataset_collate_label_blind():
    cfg=PreprocessConfig(size=8,slices=2)
    data=[record(0,study='a'),record(1,'Coronal',study='b')]
    ds=StudyBagDataset(data,preprocess_config=cfg,max_series=2,
                       series_reader=lambda r: None)
    # Override preprocessing with a controlled fake via monkeypatch.
    import oxalpha.data.dataset as d
    original=d.preprocess_series
    d.preprocess_series=lambda raw,cfg,**kw:fake_tensor(cfg)
    try:
        batch=collate_studies([ds[0],ds[1]])
    finally:
        d.preprocess_series=original
    assert batch['images'].shape == (2,2,2,3,8,8)
    assert batch['study_ids'] == ['a','b']
    assert not {'Report','labels','text','y','teacher_logits'} & batch.keys()
