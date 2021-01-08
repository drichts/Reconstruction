import general_functions as gen
import lda.ct_functions as ct
from scipy.io import savemat, loadmat
from lda.parameters import *


class AnalyzeLDA:

    def __init__(self, folder, directory=DIRECTORY):
        self.folder = os.path.join(directory, folder)

        self.raw_data = os.path.join(self.folder, 'Data', 'data.npy')
        self.air_data = AIR
        self.dark_data = DARK

        self.corr_data = os.path.join(self.folder, 'Data', 'data_corr.npy')

        # temp_data = self.sum_bins(np.load(self.raw_data), 1, 4)
        # np.save(self.raw_data, temp_data)
        # self.air_data = self.sum_bins(self.air_data, 1, 4)
        # print(np.shape(self.air_data))
        # self.dark_data = self.sum_bins(self.dark_data, 1, 4)
        # print(np.shape(self.dark_data))

        if not os.path.exists(self.corr_data):
            temp_data = self.intensity_correction(self.correct_dead_pixels())
            np.save(self.corr_data, temp_data)

        self.bg_mask = None

    def intensity_correction(self, data):
        """
        This function corrects flatfield data to show images, -ln(I/I0), I is the intensity of the data, I0 is the
        intensity in an airscan
        """
        return gen.intensity_correction(data, self.air_data, self.dark_data)

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
    def sum_bins(data, bin1, bin2):

        data_shape = np.array(np.shape(data))
        if len(data_shape) == 3:
            data = np.expand_dims(data, axis=0)
            data_shape = np.array(np.shape(data))

        # The new data will have the number of added bins - 1 new counters
        data_shape[-1] = data_shape[-1] - (bin2 - bin1)
        new_data = np.zeros(data_shape)

        # Set first bins to be the same
        new_data[:, :, :, 0:bin1] = data[:, :, :, 0:bin1]

        # Set the new bin to the sum of the original bins
        new_data[:, :, :, bin1] = np.sum(data[:, :, :, bin1:bin2 + 1], axis=3)

        # Keep the bins after the same
        new_data[:, :, :, bin1 + 1:] = data[:, :, :, bin2 + 1:]

        return np.squeeze(new_data)

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

        self.filt_data = os.path.join(self.folder, 'Data', 'proj_filt.npy')
        self.filt_data_mat = os.path.join(self.folder, 'Data', 'proj_filt.mat')

        temp_data = np.load(self.corr_data)
        # This will cut the projection down to the correct number if there are more than necessary
        if num_proj != len(temp_data):
            diff = abs(num_proj - len(temp_data))
            np.save(self.corr_data, temp_data[int(np.ceil(diff / 2)):len(temp_data) - diff // 2])

        if not os.path.exists(self.filt_data):
            temp_data = self.reorganize(np.load(self.corr_data))
            temp_data = ct.filtering(temp_data)
            np.save(self.filt_data, temp_data)
            savemat(self.filt_data_mat, {'data': temp_data, 'label': 'filt_data'})
