import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import mask_functions as msk
from skimage.feature import canny

def filter_sino(low_sino, high_sino):
    """

    :param low_sino: ndarray
            The low energy sinogram of the data (lower energy bin)
    :param high_sino: ndarray
            The high energy sinogram to extract the metal artifact curves from
    :return: A (hopefully) metal artifact corrected sinogram
    """

    # Shape of the sinogram
    shp = np.shape(low_sino)

    # Extract the metal curve's pixels from the sinogram
    # Could be easier to do it in the high-energy or low energy and then just copy the same thing in the other sinogram
    pixels = [5, 5, 6, 2]  # This will be the pixels

    # Create masks of the phantom without the metal and the metal only
    exclude_metal = np.ones(shp)
    metal_only = np.zeros(shp)

    exclude_metal[pixels] = 0
    metal_only[pixels] = 1

    # Get only the metal from the high sinogram, set everything else to 0
    # Set the metal to zero in the low sinogram
    high_sino = np.multiply(high_sino, metal_only)
    low_sino = np.multiply(low_sino, exclude_metal)

    # Add the two sinograms together
    final_sino = low_sino + high_sino

    return final_sino

data = np.load(r'D:\OneDrive - University of Victoria\Research\LDA Data\21-05-12_CT_metal\metal_in\Data\data_corr.npy')
data = data[:, 15, :, 6]

data2 = medfilt(data, 3)
data2 = canny(data2, sigma=2, low_threshold=10, high_threshold=12)
data2[0] = data2[1]
data2[-1] = data2[-2]
# print(data2[180])
data2 = np.array(data2, dtype='int')
for i, ang in enumerate(data2):
    print(len(np.argwhere(ang == 1)))
    true_counter = 0
    for j, val in enumerate(ang[1:]):
        if val == 1:
            if true_counter == 0:
                true_counter = j
            elif true_counter == j-1:
                true_counter = j-1

            else:
                data2[i, true_counter:j] = 1
                true_counter = 0

# background_mask = msk.square_ROI(data2)
#
# background = np.nanmean(data2*background_mask)
#
# data[data > background*1.1] = 0

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].imshow(data)
ax[1].imshow(data2)
plt.show()