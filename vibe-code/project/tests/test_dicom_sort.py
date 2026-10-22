"""SPEC Q-IMG-01 / Q-013: deterministic physical order and QA flags."""
import sys
from pathlib import Path
from dataclasses import replace
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from oxalpha.data.dicom import DecodedSlice, SliceHeader, SeriesDecodeError, order_slices


def sl(z, *, inst=None, path=None, iop=(1,0,0,0,1,0), spacing=(0.3,0.3)):
    h = SliceHeader(path=path or f'{z}.dcm', sop_instance_uid=f'uid-{z}',
        series_instance_uid='series', study_instance_uid='study', rows=2,columns=2,
        pixel_spacing=spacing, orientation=iop, position=(0.,0.,float(z)),
        instance_number=inst, slice_location=float(z), laterality=None,
        image_laterality=None, thickness_mm=3.0, spacing_between_slices_mm=4.0,
        manufacturer=None, magnetic_field_strength_t=None,transfer_syntax_uid=None,
        pixel_representation=1,rescale_slope=1.,rescale_intercept=0.,
        pixel_padding_value=None,pixel_padding_range_limit=None)
    return DecodedSlice(h,np.full((2,2),z,dtype=np.float32),np.ones((2,2),dtype=bool))


def test_physical_order_ignores_filename_and_instance():
    s = [sl(10, inst=1,path='a'),sl(0,inst=3,path='z'),sl(5,inst=2,path='b')]
    order = order_slices(s)
    assert [x.image[0,0] for x in order.slices] == [0,5,10]
    assert [x.image[0,0] for x in order_slices(list(reversed(s))).slices] == [0,5,10]


def test_reversed_normal_order_is_canonical():
    flipped = (1,0,0,0,-1,0)
    s = [sl(z,iop=flipped) for z in [10,0,5]]
    result = order_slices(s)
    assert [v.image[0,0] for v in result.slices] == [0,5,10]
    assert 'reversed_normal' in result.flags


def test_duplicates_and_irregular_gaps():
    s = [sl(0),sl(.02,path='duplicate'),sl(2),sl(15)]
    r = order_slices(s)
    assert r.duplicate_positions == 1
    assert len(r.slices) == 3
    assert 'irregular_spacing' in r.flags


def test_inconsistent_geometry_rejected_with_flag():
    s = [sl(0),sl(1),sl(2, spacing=(0.5,0.5))]
    r = order_slices(s)
    assert len(r.slices) == 2
    assert r.rejected_geometry == 1
    assert 'inconsistent_geometry' in r.flags


def test_fallback_without_position():
    s = [sl(2, inst=2),sl(1,inst=1)]
    s = [replace(x, header=replace(x.header,position=None)) for x in s]
    r = order_slices(s)
    assert 'ordering_unreliable' in r.flags
    assert [x.header.instance_number for x in r.slices] == [1,2]


def test_oblique_flag():
    t = np.deg2rad(35)
    iop = (1,0,0,0,float(np.cos(t)),float(np.sin(t)))
    s = [sl(0,iop=iop),sl(1,iop=iop)]
    assert 'oblique' in order_slices(s).flags


def test_mixed_series_rejected():
    s = [sl(0),sl(1)]
    s[1] = replace(s[1],header=replace(s[1].header,series_instance_uid='other'))
    with pytest.raises(SeriesDecodeError,match='mixed'):
        order_slices(s)
