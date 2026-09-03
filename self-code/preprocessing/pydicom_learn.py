import matplotlib.pyplot as plt

from pydicom import examples
from pydicom import dcmread
from pydicom.data import get_testdata_file
from pydicom.pixels import pixel_array

from pathlib import Path

# Get an example dataset as a FileDataset instance
# ds = examples.ct

ds = dcmread("series2test4.dcm")
# print(str(ds))

for k, v in ds.items():
    if k != "Pixel Datassssssss":
        print(k)
        # print(v)
    print("-")
# Convert the pixel data to an ndarray
arr = pixel_array(ds)

# Display the pixel data using matplotlib
# plt.imshow(arr, cmap="gray")
# plt.show()

