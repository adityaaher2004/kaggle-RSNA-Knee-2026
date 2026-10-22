"""Regression tests for Q-IMG-02 / Q-014."""
import sys
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from oxalpha.data.dicom import DecodedSlice, SeriesReadResult, header_from_dataset
from oxalpha.data.preprocessing import (PreprocessConfig, PreprocessError, normalize_intensity,
    sample_slice_indices, physical_slice_positions, letterbox, preprocess_series)


def make_series(n=6, h=16, w=32, *, positions=None, signed=False):
    slices = []
    for i in range(n):
        img = np.arange(h*w, dtype=np.float32).reshape(h,w) + i*10
        if signed: img -= 1000
        d = SimpleNamespace(Rows=h, Columns=w, PixelSpacing=[.3,.5],
                            ImageOrientationPatient=[1,0,0,0,1,0],
                            ImagePositionPatient=[0,0,float(i if positions is None else positions[i])],
                            InstanceNumber=i+1)
        header = header_from_dataset(d, f'{i}.dcm')
        mask = np.ones((h,w),dtype=bool)
        mask[:,0] = False
        img[:,0] = 0.
        slices.append(DecodedSlice(header,img,mask))
    return SeriesReadResult(slices)


def test_zero_is_valid_and_padding_not_in_percentiles():
    vol=np.array([[[0,10,999],[5,20,999]]],dtype=np.float32)
    valid=np.array([[[True,True,False],[True,True,False]]])
    out, flags, (lo,hi)=normalize_intensity(vol,valid,low=0,high=100)
    assert not flags and lo == 0 and hi == 20
    assert out[0,0,0]==0 and out[0,1,1]==1
    assert np.all(out[~valid]==0)


def test_constant_and_no_valid_rejected():
    s=make_series()
    for slice_ in s.slices:
        slice_.image[:]=3.
    with pytest.raises(PreprocessError,match='contrast'): preprocess_series(s,PreprocessConfig(size=32,slices=4))
    v=np.zeros((1,2,2),dtype='f4')
    out,flags,_=normalize_intensity(v,np.zeros_like(v,bool))
    assert 'empty_valid_volume' in flags and not out.any()


def test_physical_positions_and_nonuniform_sampling():
    s=make_series(6,positions=[0,1,2,3,4,30])
    positions=physical_slice_positions(s)
    assert np.array_equal(positions,[0,1,2,3,4,30])
    idx,mask=sample_slice_indices(6,3,positions)
    assert np.array_equal(idx,[0,4,5]) and mask.all()
    fallback,_=sample_slice_indices(6,3)
    assert not np.array_equal(idx,fallback)


def test_short_series_padding_and_valid_masks():
    cfg=PreprocessConfig(size=32,slices=7,imagenet_normalize=False)
    t=preprocess_series(make_series(3),cfg)
    assert t.images.shape==(7,3,32,32)
    assert np.array_equal(t.selected_indices,[0,1,2,-1,-1,-1,-1])
    assert t.slice_mask.tolist()==[True]*3+[False]*4
    assert not t.images[3:].any() and not t.spatial_mask[3:].any()


def test_two_point_five_d_uses_unsampled_neighbors():
    s=make_series(5,4,4)
    for i,x in enumerate(s.slices): x.image[:]=i*100 + np.arange(16).reshape(4,4)
    cfg=PreprocessConfig(size=4,slices=3,low_percentile=0, high_percentile=100,imagenet_normalize=False)
    t=preprocess_series(s,cfg)
    assert np.array_equal(t.selected_indices,[0,2,4])
    lo, hi = t.metadata['intensity_percentiles']
    for channel, neighbor in enumerate([1,2,3]):
        expected = np.clip((s.slices[neighbor].image - lo) / (hi-lo), 0, 1)
        assert np.allclose(t.images[1,channel,:,1:], expected[:,1:], atol=1e-6)
        assert np.all(t.images[1,channel,:,0] == 0)  # pixel-padding mask


def test_letterbox_preserves_anatomy_and_phantom_aspect():
    im=np.ones((10,20),dtype=np.float32)
    m=np.ones_like(im,bool)
    out,mask,placement=letterbox(im,m,40)
    assert placement==(10,0,20,40)
    assert mask.sum()==800
    assert np.all(out[:10]==0) and np.all(out[10:30]==1)
    assert np.all(out[30:]==0)


def test_valid_spatial_mask_does_not_mark_padding():
    t=preprocess_series(make_series(4),PreprocessConfig(size=64,slices=3))
    assert t.spatial_mask.shape==(3,64,64)
    assert not t.spatial_mask[:,:,0].any()  # letterbox horizontal margin
    assert t.spatial_mask.any()
    assert t.images.dtype==np.float32 and np.isfinite(t.images).all()


def test_deterministic_inference_and_seeded_train_only_augmentation():
    s=make_series(6)
    cfg=PreprocessConfig(size=32,slices=4,noise_std=.01,slice_dropout=.3)
    a=preprocess_series(s,cfg,training=False,seed=1)
    b=preprocess_series(s,cfg,training=False,seed=2)
    assert np.array_equal(a.images,b.images)
    c=preprocess_series(s,cfg,training=True,seed=12)
    d=preprocess_series(s,cfg,training=True,seed=12)
    e=preprocess_series(s,cfg,training=True,seed=13)
    assert np.array_equal(c.images,d.images)
    assert not np.array_equal(c.images,e.images)
    assert c.slice_mask.any()


def test_no_mirror_and_preserve_metadata_and_inplane_semantics():
    s=make_series(3)
    t=preprocess_series(s,PreprocessConfig(size=32,slices=3))
    assert t.metadata['canonical_inplane_orientation']=='unchanged'
    assert t.metadata['full_fov_mm']==(4.8,16.0)
    assert t.metadata['pixel_spacing_mm']==(.3,.5)


def test_invalid_config_and_index_contract():
    with pytest.raises(ValueError): PreprocessConfig(low_percentile=50,high_percentile=40)
    with pytest.raises(ValueError): sample_slice_indices(5,3,np.arange(5)[::-1])
    idx,mask=sample_slice_indices(5,1)
    assert idx.tolist()==[2] and mask.tolist()==[True]
