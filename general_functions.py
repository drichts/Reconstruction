import _pickle as pickle
import numpy as np


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