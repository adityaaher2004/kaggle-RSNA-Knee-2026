"""SPEC: S3.1

Read DICOM headers (stop_before_pixels) and pixels with decoder fallback chain pylibjpeg -> gdcm -> SimpleITK; RescaleSlope/Intercept; header dict incl. TransferSyntaxUID and decoder used.

API (planned): read_headers(series_dir) -> list[dict]; read_pixels(path) -> np.ndarray[H,W] float32
"""

raise NotImplementedError("stub -- see specs.txt S3.1")
