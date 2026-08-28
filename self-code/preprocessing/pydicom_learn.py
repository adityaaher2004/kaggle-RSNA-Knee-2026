import matplotlib.pyplot as plt

from pydicom import examples
from pydicom.pixels import pixel_array

# Get an example dataset as a FileDataset instance
ds = examples.ct

# Convert the pixel data to an ndarray
arr = pixel_array(ds)
assert arr.shape == (128, 128)
assert str(arr.dtype) == "int16"

# Display the pixel data using matplotlib
plt.imshow(arr, cmap="gray")
plt.show()

