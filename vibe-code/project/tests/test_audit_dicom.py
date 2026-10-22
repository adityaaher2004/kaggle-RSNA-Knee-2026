"""Q-016 synthetic audit tests: no DICOM dependency required for mocked reads."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'audit_dicom.py'
spec = importlib.util.spec_from_file_location('audit_dicom', SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_geometry_detects_duplicates_and_irregular_gaps():
    def h(z):
        return SimpleNamespace(rows=256, columns=256, pixel_spacing=(0.5, 0.5),
            orientation=(1,0,0,0,1,0), position=(0,0,z))
    flags, dup, median = audit._geometry([h(z) for z in (0, 1, 1.03, 2, 15)])
    assert dup == 1
    assert 'irregular_spacing' in flags
    assert median == 1


def test_geometry_fallback_orientation():
    h = SimpleNamespace(rows=10, columns=10, pixel_spacing=None, orientation=None, position=None)
    flags, dup, median = audit._geometry([h, h])
    assert 'ordering_unreliable' in flags and dup == 0 and median is None


def test_audit_one_real_directory_mocked_reader(tmp_path):
    folder = tmp_path / 'study' / 'series'
    folder.mkdir(parents=True)
    for i in range(3):
        (folder / f'{i}.dcm').write_bytes(b'x')
    def reader(path):
        z = float(path.stem)
        return SimpleNamespace(study_instance_uid='study', series_instance_uid='series',
            sop_instance_uid=f'sop{path.stem}', rows=100, columns=200,
            pixel_spacing=(0.7, 0.7), orientation=(1,0,0,0,1,0), position=(0,0,z),
            manufacturer='test', transfer_syntax_uid='1.2', magnetic_field_strength_t=3.0,
            path=str(path))
    out = audit.audit_one({'StudyInstanceUID':'study','SeriesInstanceUID':'series'}, tmp_path,
                              decode=True, decode_limit=2, read_header_fn=reader,
                              read_slice_fn=lambda p: object())
    assert out['status'] == 'ok'
    assert out['n_headers_ok'] == 3 and out['n_decoded'] == 2
    assert out['median_gap_mm'] == 1 and out['flags'] == []
    assert 'PatientID' not in str(out)


def test_missing_series_and_summary(tmp_path):
    out = audit.audit_one({'StudyInstanceUID':'s','SeriesInstanceUID':'x'}, tmp_path)
    assert out['status'] == 'missing'
    report = audit.summarize([out], 1)
    assert report['status_counts']['missing'] == 1
    assert report['header_failure_rate'] is None


def test_checkpoint_last_write_wins(tmp_path):
    path = tmp_path / 'manifest.jsonl'
    rows = [{'StudyInstanceUID':'s','SeriesInstanceUID':'x','status':'failed'},
            {'StudyInstanceUID':'s','SeriesInstanceUID':'x','status':'ok'}]
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    assert audit.checkpoint_rows(path)['s/x']['status'] == 'ok'


def test_csv_schema_and_duplicates(tmp_path):
    f = tmp_path / 'train_series.csv'
    f.write_text('StudyInstanceUID,SeriesInstanceUID\ns,x\ns,x\n')
    import pytest
    with pytest.raises(ValueError, match='duplicate'):
        audit.discover_series(tmp_path, 'train')


def test_cli_resume_missing_series(tmp_path):
    data = tmp_path / 'data'
    data.mkdir()
    (data / 'train_series.csv').write_text('StudyInstanceUID,SeriesInstanceUID\ns,x\n')
    work = tmp_path / 'work'
    args = ['--data-root', str(data), '--out-dir', str(work), '--splits', 'train', '--no-parquet']
    assert audit.main(args) == 0
    assert audit.main(args) == 0
    rows = audit.checkpoint_rows(work / 'manifest_train.jsonl')
    assert len(rows) == 1 and rows['s/x']['status'] == 'missing'
    summary = json.loads((work / 'audit_train_summary.json').read_text())
    assert summary['complete'] and summary['audited_series'] == 1
