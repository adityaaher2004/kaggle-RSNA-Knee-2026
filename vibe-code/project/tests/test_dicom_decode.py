"""SPEC Q-IMG-01 / Q-012: native pixel decoding tests."""
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from oxalpha.data.dicom import DicomReadError, decode_dataset, header_from_dataset, read_header


class FakeDataset:
    def __init__(self, array, **attrs):
        self.Rows, self.Columns = array.shape
        self.PixelData = b'fake-pixel-payload'
        self.PixelRepresentation = int(np.issubdtype(array.dtype, np.signedinteger))
        self.pixel_array = array
        self.file_meta = SimpleNamespace(TransferSyntaxUID='1.2.840.10008.1.2.1')
        self.__dict__.update(attrs)


def test_signed_pixels_and_rescale():
    a = np.array([[-100, -2], [5, 200]], dtype=np.int16)
    result = decode_dataset(FakeDataset(a, RescaleSlope=2.5, RescaleIntercept=-10.))
    np.testing.assert_array_equal(result.image, a.astype(np.float32)*2.5 - 10)
    assert result.image[0, 0] == -260
    assert result.valid_mask.all()


def test_unsigned_and_padding_raw_space():
    a = np.array([[0, 1], [65535, 3]], dtype=np.uint16)
    out = decode_dataset(FakeDataset(a, PixelPaddingValue=65535, RescaleSlope=2., RescaleIntercept=5.))
    assert out.image[1, 0] == 0
    assert not out.valid_mask[1, 0]
    assert out.image[0, 0] == 5


def test_padding_range_and_unchanged_source():
    a = np.array([[-3, -2, -1], [0, 1, 2]], dtype=np.int16)
    out = decode_dataset(FakeDataset(a, PixelPaddingValue=-3, PixelPaddingRangeLimit=-1))
    assert out.valid_mask.sum() == 3
    assert np.array_equal(a[0], [-3,-2,-1])


def test_missing_pixels_and_multiframe_rejected():
    a = np.ones((2,2), dtype=np.uint16)
    ds = FakeDataset(a)
    ds.PixelData = None
    with pytest.raises(DicomReadError, match='missing PixelData'):
        decode_dataset(ds)
    ds.PixelData = b'yes'
    ds.NumberOfFrames = 2
    with pytest.raises(DicomReadError, match='multiframe'):
        decode_dataset(ds)


def test_invalid_geometry_and_decode_failure():
    ds = FakeDataset(np.zeros((2,2),dtype=np.uint16))
    ds.Rows = 0
    with pytest.raises(DicomReadError):
        header_from_dataset(ds)
    ds.Rows = 2
    ds.pixel_array = np.zeros((3,2),dtype=np.uint16)
    with pytest.raises(DicomReadError, match='unexpected pixel shape'):
        decode_dataset(ds)


def test_missing_codec_is_actionable():
    ds = FakeDataset(np.ones((2,2),dtype=np.uint16))
    del ds.pixel_array
    with pytest.raises(DicomReadError, match='codec'):
        decode_dataset(ds)


def test_real_uncompressed_dicom_roundtrip(tmp_path):
    pydicom = pytest.importorskip('pydicom')
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid
    a = np.array([[-3, 4], [20, -100]], dtype=np.int16)
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = MRImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.ImplementationClassUID = generate_uid()
    ds = FileDataset(str(tmp_path/'slice.dcm'), {}, file_meta=meta, preamble=b'\x00'*128)
    ds.SOPClassUID = MRImageStorage
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Rows = ds.Columns = 2
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = 'MONOCHROME2'
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1
    ds.PixelData = a.tobytes()
    path = tmp_path/'slice.dcm'
    ds.save_as(str(path), enforce_file_format=True)
    from oxalpha.data.dicom import read_slice
    assert read_header(path).rows == 2
    np.testing.assert_array_equal(read_slice(path).image, a.astype('float32'))


def test_optional_real_compressed_fixture():
    import os
    path = os.environ.get('RSNA_COMPRESSED_DICOM_FIXTURE')
    if not path:
        pytest.skip('Provide RSNA_COMPRESSED_DICOM_FIXTURE on Quartz for actual JPEG/J2K codec validation')
    from oxalpha.data.dicom import read_slice
    result = read_slice(path)
    assert result.image.ndim == 2 and result.image.size > 0
    assert np.isfinite(result.image).all()
