import _pickle as pickle
import numpy as np
from scipy.io import savemat, loadmat
import os
import matplotlib
# matplotlib.use('TKAgg')
import matplotlib.pyplot as plt


def save_object(obj, filepath):
    """This function takes an object and a filepath with filename and saves the object to that location"""
    if filepath[-3:] == 'pk1':
        filepath = filepath
    elif '.' in filepath[-4:]:
        filepath.replace(filepath[-3:], 'pk1')
    else:
        filepath = filepath + '.pk1'
    with open(filepath, 'wb') as output:  # Overwrites any existing file
        pickle.dump(obj, output)


def load_object(filepath):
    """This function loads the object at filepath."""
    if filepath[-3:] == 'pk1':
        filepath = filepath
    elif '.' in filepath[-4:]:
        filepath.replace(filepath[-3:], 'pk1')
    else:
        filepath = filepath + '.pk1'
    obj = []
    with open(filepath, 'rb') as openfile:
        while True:
            try:
                obj.append(pickle.load(openfile))
            except EOFError:
                break
    return obj[0]


def intensity_correction(data, air_data, dark_data):
    """
    This function corrects flatfield data to show images, -ln(I/I0), I is the intensity of the data, I0 is the
    intensity in an airscan
    :param data: The data to correct (must be the same shape as air_data)
    :param air_data: The airscan data (must be the same shape as data)
    :param dark_data: The darkscan data (must be the same shape as airdata)
    :return: The corrected data array
    """
    with np.errstate(invalid='ignore'):
        data = np.log(np.subtract(air_data, dark_data)) - np.log(np.subtract(data, dark_data))
    return data


def cnr(image, contrast_mask, background_mask):
    """
    This function calculates the CNR of an ROI given the image, the ROI mask, and the background mask
    It also gives the CNR error
    :param image: The image to be analyzed as a 2D numpy array
    :param contrast_mask: The mask of the contrast area as a 2D numpy array
    :param background_mask: The mask of the background as a 2D numpy array
    :return CNR, CNR_error: The CNR and error of the contrast area
    """
    # The mean signal within the contrast area
    mean_roi = np.nanmean(image * contrast_mask)
    std_roi = np.nanstd(image * contrast_mask)

    # Mean and std. dev. of the background
    bg = np.multiply(image, background_mask)
    mean_bg = np.nanmean(bg)
    std_bg = np.nanstd(bg)

    cnr_val = abs(mean_roi - mean_bg) / std_bg
    cnr_err = np.sqrt(std_roi ** 2 + std_bg ** 2) / std_bg

    return cnr_val, cnr_err


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
            # Pixel is corrected in every counter and capture
            avg_val = get_average_pixel_value(data[i, :, :, pixel[-1]], pixel[:-1], dead_pixel_mask[:, :, pixel[-1]])
            data[i, pixel[0], pixel[1], pixel[-1]] = avg_val  # Set the new value in the 4D array

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


def reshape(data):
    new_shape = (data.shape[0] // 2, 2, *data.shape[1:])
    data_sum = np.sum(np.reshape(data, new_shape), axis=1)
    np.save(r'D:\OneDrive - University of Victoria\Research\LDA Data\ct_180frames_1sproj_111220 - Synth\Data\data.npy', data_sum)
    return data_sum


def save_mat(path, data):
    savemat(path, {'data': data, 'label': 'central-ish sinogram'})




# air = np.load(r'D:\OneDrive - University of Victoria\Research\LDA Data\airscan_120kVP_1mA_1mmAl_3x8coll_360s_6frames\Data\data.npy')
# dark = np.load(r'D:\OneDrive - University of Victoria\Research\LDA Data\darkscan_360s_6frames\Data\data.npy')
# dpm = np.load(r'D:\OneDrive - University of Victoria\Research\LDA Data\mod1_deadpixelmask.npy')
# air = correct_dead_pixels(air, dpm)
# dark = np.sum(dark[1:], axis=0)
# dark = correct_dead_pixels(dark, dpm)
#
# plt.imshow(air[4, :, :, 6], vmin=1E5, vmax=1E9)
# plt.show()

# np.save(r'D:\OneDrive - University of Victoria\Research\LDA Data\airscan_120kVP_1mA_1mmAl_3x8coll_360s_6frames\Data\data_corr.npy', air)
# np.save(r'D:\OneDrive - University of Victoria\Research\LDA Data\darkscan_360s_6frames\Data\data_corr_300s.npy', dark)