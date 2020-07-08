import numpy as np
import os
import _pickle as pickle


class Analyze:
    def __init__(self, folder, load_dir, save_dir):
        self.folder = folder
        self.load_dir = os.path.join(load_dir, folder)
        self.save_dir = os.path.join(save_dir, folder)
        os.makedirs(save_dir, exist_ok=True)

    @staticmethod
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

    def save_object(self, filename):
        """This function takes an object and a filename and saves the object to the save directory"""
        if filename[-3:] == 'pk1':
            filename = filename
        elif '.' in filename[-4:]:
            filename.replace(filename[-3:], 'pk1')
        else:
            filename = filename + '.pk1'
        with open(self.save_dir + '/' + filename, 'wb') as output:
            pickle.dump(self, output)