import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
spec = importlib.util.spec_from_file_location('smoke_studies', ROOT / 'scripts' / 'smoke_studies.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
from oxalpha.data.dataset import SeriesRecord, assemble_study
from oxalpha.data.preprocessing import PreprocessConfig, SeriesTensor


def sample(cfg=None):
    cfg = cfg or PreprocessConfig(size=8, slices=2)
    rec = SeriesRecord('study', 'series', Path('/x'), 'Sagittal', 1, 0)
    arr = SeriesTensor(np.ones((2, 3, 8, 8), np.float32), np.ones(2, bool),
                       np.ones((2, 8, 8), bool), np.arange(2), metadata={})
    study = assemble_study('study', [rec], {'series': arr}, max_series=2, cfg=cfg)
    return cfg, study


def test_shapes_masks_and_provenance():
    cfg, study = sample()
    assert study.selected_slice_indices == {'series': [0, 1]}
    info = mod.validate_study(study.inference_inputs(), cfg=cfg, max_series=2,
                              expected_planes={'Sagittal'})
    assert info['valid_slices'] == 2
    assert set(study.inference_inputs()) == set(mod.FIELDS) | {'study_ids'}


def test_padding_nonzero_rejected():
    cfg, study = sample()
    x = study.inference_inputs()
    x['images'][1,0,0,0,0] = 1
    with pytest.raises(ValueError, match='padding'):
        mod.validate_study(x, cfg=cfg, max_series=2)


def test_missing_plane_detected():
    cfg, study = sample()
    with pytest.raises(ValueError, match='lost available'):
        mod.validate_study(study.inference_inputs(), cfg=cfg, max_series=2,
                           expected_planes={'Sagittal','Coronal'})


def test_cache_roundtrip_and_checksum(tmp_path):
    cfg, study = sample()
    inputs = study.inference_inputs()
    key = mod.cache_key('study', 'a'*64, cfg, 2)
    mod.write_cache(tmp_path, key, inputs, source_sha='a'*64, cfg=cfg,
                    selected=['series'], indices={'series':[0,1]})
    cached, manifest = mod.read_cache(tmp_path, key)
    assert cached['study_ids'] == 'study'
    assert manifest['selected_slice_indices'] == {'series':[0,1]}
    for f in mod.FIELDS:
        np.testing.assert_array_equal(cached[f], inputs[f])
    with (tmp_path / (key+'.json')).open('r+') as f:
        meta = json.load(f); meta['tensor_sha256'] = '0'*64
        f.seek(0); json.dump(meta, f); f.truncate()
    with pytest.raises(ValueError, match='checksum'):
        mod.read_cache(tmp_path, key)


def test_legacy_npz_rejected(tmp_path):
    np.savez(tmp_path/'old.npz', images=np.zeros((1,)))
    assert mod.read_cache(tmp_path, 'old') is None
    (tmp_path/'old.json').write_text(json.dumps({'format': 'legacy', 'key': 'old'}))
    with pytest.raises(ValueError, match='legacy'):
        mod.read_cache(tmp_path,'old')


def test_source_digest_invalidates_on_pixel_change(tmp_path):
    path = tmp_path / 's' / 'x'; path.mkdir(parents=True)
    file = path / 'a.dcm'; file.write_bytes(b'first')
    rec = SeriesRecord('s','x',path)
    first = mod.source_digest([rec]); file.write_bytes(b'second')
    assert mod.source_digest([rec]) != first


def test_metadata_loading(tmp_path):
    (tmp_path/'train_series.csv').write_text('StudyInstanceUID,SeriesInstanceUID,Anatomical_Plane,Fluid_Sensitive,Fat_Suppression\ns,x,Sagittal,1,0\n')
    groups = mod.csv_records(tmp_path, 'train')
    assert groups['s'][0].fluid == 1 and groups['s'][0].fat == 0
    assert groups['s'][0].path == tmp_path/'train_series'/'s'/'x'


def test_reject_missing_study(tmp_path):
    with pytest.raises(FileNotFoundError):
        mod.csv_records(tmp_path,'train')


def test_cli_smoke_and_cache_hits(tmp_path, monkeypatch):
    (tmp_path/'train_series.csv').write_text('StudyInstanceUID,SeriesInstanceUID,Anatomical_Plane\ns,x,Sagittal\n')
    dicom_dir = tmp_path / 'train_series'/'s'/'x'; dicom_dir.mkdir(parents=True)
    (dicom_dir/'a.dcm').write_bytes(b'fixture')
    cfg, tensor = sample()
    # Use exact tensor from the real contract with fake decoding to test CLI I/O.
    monkeypatch.setattr(mod, 'preprocess_study', lambda *args, **kwargs: tensor)
    cmd = ['--data-root',str(tmp_path),'--out-dir',str(tmp_path/'out'),
           '--size','8','--slices','2','--max-series','2','--cache']
    assert mod.main(cmd) == 0
    assert mod.main(cmd) == 0
    report=json.loads((tmp_path/'out'/'q017_summary.json').read_text())
    assert report['cache_hits'] == 1 and report['ok'] == 1
    assert len((tmp_path/'out'/'q017_results.jsonl').read_text().splitlines()) == 1
