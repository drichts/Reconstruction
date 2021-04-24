import os
import numpy as np
from datetime import datetime


def correct_dead_pixels(data, dead_pixel_mask):
    """
    This is to correct for known dead pixels. Takes the average of the eight surrounding pixels.
    Could implement a more sophisticated algorithm here if needed.
    :param data: 4D ndarray
                The data array in which to correct the pixels <captures, rows, columns, counter>
    :param dead_pixel_mask: 2D ndarray
                A data array with the same number of rows and columns as 'data'. Contains np.nan everywhere there
                is a known non-responsive pixel
    :return: The data array corrected for the dead pixels
    """
    # Find the dead pixels (i.e pixels = to nan in the DEAD_PIXEL_MASK)
    dead_pixels = np.array(np.argwhere(np.isnan(dead_pixel_mask)), dtype='int')

    data_shape = np.shape(data)
    while len(data_shape) < 4:
        if len(data_shape) == 3:
            data = np.expand_dims(data, axis=0)
        else:
            data = np.expand_dims(data, axis=3)
        data_shape = np.shape(data)

    for pixel in dead_pixels:
        for i in np.arange(data_shape[0]):
            for b in np.arange(data_shape[-1]):
                # Pixel is corrected in every counter and capture
                avg_val = get_average_pixel_value(data[i, :, :, b], pixel, dead_pixel_mask[:, :])
                data[i, pixel[0], pixel[1], b] = avg_val  # Set the new value in the 4D array

    return np.squeeze(data)


def get_average_pixel_value(img, pixel, dead_pixel_mask):
    """
    Averages the dead pixel using the 8 nearest neighbours
    Checks the dead pixel mask to make sure each of the neighbors is not another dead pixel

    :param img: 2D array
                The projection image

    :param pixel: tuple (row, column)
                The problem pixel (is a 2-tuple)

    :param dead_pixel_mask: 2D numpy array
                Mask with 1 at good pixel coordinates and np.nan at bad pixel coordinates

    :return: the average value of the surrounding pixels
    """
    shape = np.shape(img)
    row, col = pixel

    # Grabs each of the neighboring pixel values and sets to nan if they are bad pixels or
    # outside the bounds of the image
    if col == shape[1] - 1:
        n1 = np.nan
    else:
        n1 = img[row, col + 1] * dead_pixel_mask[row, col + 1]
    if col == 0:
        n2 = np.nan
    else:
        n2 = img[row, col - 1] * dead_pixel_mask[row, col - 1]
    if row == shape[0] - 1:
        n3 = np.nan
    else:
        n3 = img[row + 1, col] * dead_pixel_mask[row + 1, col]
    if row == 0:
        n4 = np.nan
    else:
        n4 = img[row - 1, col] * dead_pixel_mask[row - 1, col]
    if col == shape[1] - 1 or row == shape[0] - 1:
        n5 = np.nan
    else:
        n5 = img[row + 1, col + 1] * dead_pixel_mask[row + 1, col + 1]
    if col == 0 or row == shape[0] - 1:
        n6 = np.nan
    else:
        n6 = img[row + 1, col - 1] * dead_pixel_mask[row + 1, col - 1]
    if col == shape[1] - 1 or row == 0:
        n7 = np.nan
    else:
        n7 = img[row - 1, col + 1] * dead_pixel_mask[row - 1, col + 1]
    if col == 0 or row == 0:
        n8 = np.nan
    else:
        n8 = img[row - 1, col - 1] * dead_pixel_mask[row - 1, col - 1]

    # Takes the average of the neighboring pixels excluding nan values
    avg = np.nanmean(np.array([n1, n2, n3, n4, n5, n6, n7, n8]))

    return avg

if __name__ == "__main__":
    data = np.load(r'D:\OneDrive - University of Victoria\Research\LDA Data\21-04-14_CT_bin_width_5\metal_phantom\Data\data.npy')[:, :, :, 6]
    dpm = np.load(r'D:\OneDrive - University of Victoria\Research\LDA Data\dead_pixel_mask_4.npy')

    data_shape = np.array(np.shape(data))
    dead_pixels = np.array(np.argwhere(np.isnan(dpm)), dtype='int')

    data1 = np.copy(data)
    data2 = np.copy(data)
    data2 *= dpm
    data2 = np.pad(data2, 2)

    start = datetime.now().timestamp()
    for pixel in dead_pixels:
        for z in range(len(data)):
            # Pixel is corrected in every counter and capture
            avg_val = get_average_pixel_value(data1[z], pixel, dpm)
            data1[z, pixel[0], pixel[1]] = avg_val  # Set the new value in the 4D array
    print(datetime.now().timestamp()-start)
    print()


    start = datetime.now().timestamp()
    for pixel in dead_pixels:
        for z in range(len(data)):
            row = pixel[0]+2
            col = pixel[1]+2
            if row >= 24 and col >= 24:
                data2[row, col] = np.nanmean(data2[z+2, row-1:, col-1:])
            elif row >= 24:
                data2[row, col] = np.nanmean(data2[z+2, row-1:, col-1:col+3])
            elif col >= 24:
                data2[row, col] = np.nanmean(data2[z+2, row-1:row+3, col-1:])
            else:
                data2[row, col] = np.nanmean(data2[z + 2, row - 1:row+3, col - 1:col + 3])

    data2 = data2[2:-2, 2:-2, 2:-2]
    print(datetime.now().timestamp() - start)
    print()




