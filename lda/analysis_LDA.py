import os
import general_functions as gen
from lda.parameters import *


class AnalyzeLDA:

    def __init__(self, folder, directory=DIRECTORY):
        self.folder = os.path.join(directory, folder)

        self.raw_data = os.path.join(folder, 'Data', 'data.npy')
        self.air_data = None
        self.dark_data = None

        self.corr_data = os.path.join(self.folder, 'Data', 'data_corr.npy')

        if not os.path.exists(self.corr_data):
            temp_data = self.intensity_correction(self.correct_dead_pixels())
            np.save(self.corr_data, self.reorganize(temp_data))

        self.bg_mask = None

    def intensity_correction(self, data):
        """
        This function corrects flatfield data to show images, -ln(I/I0), I is the intensity of the data, I0 is the
        intensity in an airscan
        """
        return gen.intensity_correction(data, np.load(self.air_data), np.load(self.dark_data))

    def cnr(self, image, contrast_mask):
        """
        This function calculates the CNR of an ROI given the image and a specific ROI
        :param image: The image to be analyzed as a 2D numpy array
        :param contrast_mask: The mask of the contrast area as a 2D numpy array
        :return CNR, CNR_error: The CNR and error of the contrast area
        """
        return gen.cnr(image, contrast_mask, np.load(self.bg_mask))

    def correct_dead_pixels(self):
        """
        This is to correct for known dead pixels. Takes the average of the eight surrounding pixels.
        Could implement a more sophisticated algorithm here if needed.
        :return: The data array corrected for the dead pixels
        """
        return gen.correct_dead_pixels(np.load(self.raw_data), DEAD_PIXEL_MASK)

    @staticmethod
    def reorganize(data):
        """Reorganizes the axes to be <counter, frame, row, column> """
        data_shape = np.shape(data)
        if len(data_shape) == 3:
            ax = (2, 0, 1)
        else:
            ax = (3, 0, 1, 2)
        return np.transpose(data, axes=ax)


class AnalyzeCT(AnalyzeLDA):

    def __init__(self, folder, num_proj, directory=r'D:\OneDrive - University of Victoria\Research\LDA Data'):
        super().__init__(folder, directory=directory)

        self.num_proj = num_proj

