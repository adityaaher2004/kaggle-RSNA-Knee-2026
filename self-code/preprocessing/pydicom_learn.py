import matplotlib.pyplot as plt

from pydicom import examples
from pydicom import dcmread
from pydicom.data import get_testdata_file
from pydicom.pixels import pixel_array

from pathlib import Path

# Get an example dataset as a FileDataset instance
# ds = examples.ct

file_path = Path("C:/Users/adity/Downloads/RSNA/DATA/series_data/train_series/1.2.826.0.1.3680043.8.498.10004945927472656027199792075652399585/1.2.826.0.1.3680043.8.498.72137122913943091996840828871969421711.dcm")

ds = dcmread("test.dcm")
print(ds.file_meta)

# Convert the pixel data to an ndarray
arr = pixel_array(ds)

# Display the pixel data using matplotlib
plt.imshow(arr, cmap="gray")
plt.show()

