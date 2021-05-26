from scipy.signal import medfilt
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import savemat
from skimage.feature import canny


def find_metal_traces(sinogram):
    """
    This function takes a 2D sinogram and finds the metal traces within it and returns an array of 1's and 0's with
    1's where the metal trace occurs
    :param sinogram: ndarray
            Must be a 2D numpy array
    :return: ndarray
            The array with the metal traces as 1 and everywhere as 0
    """
    # Apply a median filter to the data with a kernel width of 3
    sinogram = medfilt(sinogram, 3)

    # Find the minimum and maximum values in the sinogram (minus the air)
    minimum = np.min(sinogram[:, 40:-40])
    maximum = np.max(sinogram[:, 40:-40])
    rng = maximum - minimum
    low = minimum + rng/6
    high = maximum - rng/1.5

    # Find the edges of the metal traces in the sinogram
    edge_data = canny(sinogram, sigma=2, low_threshold=low, high_threshold=high)

    # Carry the found edges to the very edge of the sinogram
    edge_data[0] = edge_data[1]
    edge_data[-1] = edge_data[-2]

    # Switch from boolean to int array and copy the data to be able to look at the edges
    edge_data = np.array(edge_data, dtype='int')
    edge_copy = np.copy(edge_data)

    # Fill in the metal traces with 1's
    # Looks for the first instance of an edge in each row and then fills in the pixels until the next edge is found
    for i, ang in enumerate(edge_data):
        edge1 = 0
        edge2 = 0
        j = 0
        while j < len(ang):
            val = ang[j]
            if val == 1:
                if edge1 == 0 and edge2 == 0:
                    edge1 = j
                elif edge1 != 0 and edge2 == 0:
                    if j == edge1 + 1:
                        # if i == 468:
                        #     print(edge1)
                        edge1 = j
                    else:
                        # This handles if the edge in the particular row is multiple pixels in length
                        while val == 1:
                            edge2 = j
                            j += 1
                            val = ang[j]
                        # if i == 468:
                        #     print(edge1, edge2)
                        j -= 1

                        edge_data[i, edge1:edge2] = 1

                        # This handles if there is a corner in this row, therefore sets the corner as the stop and start
                        # of values to fill in

                        # If the current row is the last row
                        if i == len(edge_data) - 1:
                            if np.sum(edge_copy[i - 1, j - 3:j + 3]) == 0:
                                edge1 = j
                            else:
                                edge1 = 0

                        # If the current row in the first row
                        elif i == 0:
                            if np.sum(edge_copy[i + 1, j - 3:j + 3]) == 0:
                                edge1 = j
                            else:
                                edge1 = 0
                        # All other rows
                        else:
                            if np.sum(edge_copy[i - 1, j - 3:j + 3]) == 0 or np.sum(edge_copy[i + 1, j - 3:j + 3]) == 0:
                                edge1 = j
                                # if i == 468:
                                #     print(edge1)
                                #     print(np.sum(edge_copy[i - 1, j - 2:j + 2]))
                                #     print(np.sum(edge_copy[i + 1, j - 3:j + 3]))
                            else:
                                edge1 = 0
                                # if i == 468:
                                #     print(edge1)
                        edge2 = 0
                        j+=1
            # if i == 468:
            #     print(edge1, edge2)
            j += 1

    return edge_data, edge_copy


def correct_sino(low_sino, high_sino):
    """
    This function subtracts the metal traces out of the low energy sinogram and replaces it with the high energy metal
    traces
    :param low_sino: ndarray
            The low energy sinogram of the data (lower energy bin)
    :param high_sino: ndarray
            The high energy sinogram to extract the metal artifact curves from
    :return: A (hopefully) metal artifact corrected sinogram
    """

    # Extract the metal curve's pixels from the sinogram
    shp = np.shape(high_sino)
    metal_only = np.zeros(shp)

    # Go through all of the slices and find the traces of the metal
    for i in range(24):
        metal_only[:, i] = find_metal_traces(low_sino[:, i])

    # Invert the metal_only array to get an array of the rest of the phantom without the metal
    exclude_metal = np.array(np.invert(np.array(metal_only, dtype='bool')), dtype='int')

    # Get only the metal from the high sinogram, set everything else to 0
    # Set the metal to zero in the low sinogram
    high_sino = np.multiply(high_sino, metal_only)
    low_sino = np.multiply(low_sino, exclude_metal)

    # Add the two sinograms together
    final_sino = low_sino + high_sino

    return final_sino


data_tot = np.load(r'D:\OneDrive - University of Victoria\Research\LDA Data\21-05-12_CT_metal\metal_in\Data\data_corr.npy')

# data = correct_sino(data_tot[:, :, :, 6], data_tot[:, :, :, 3])

# fig = plt.figure(figsize=(10, 5))
# plt.imshow(data[:, 12, :])
# plt.show()
#
# np.save(r'D:\OneDrive - University of Victoria\Research\LDA Data\21-05-12_CT_metal_subtract_fullbin\metal_in\Data\data_corr_sub.npy', data)
# savemat(r'D:\OneDrive - University of Victoria\Research\LDA Data\21-05-12_CT_metal_subtract_fullbin\metal_in\Data\data_corr_sub.mat', {'data': data, 'label': 'metal_corr'}, do_compression=True)

for i in range(11, 12):
    data = data_tot[:, i, :, 0]

    datax, data2 = find_metal_traces(data)

    fig, ax = plt.subplots(1, 3, figsize=(12, 5))
    ax[0].imshow(data)
    ax[1].imshow(data2)
    ax[2].imshow(datax)
    plt.show()
    plt.savefig(r'D:\OneDrive - University of Victoria\Research\LDA Data\21-05-12_CT_metal_subtract_fullbin\metal_in\edge.png', dpi=500)
