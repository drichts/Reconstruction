import general_functions as gen
import lda.ct_functions as ct
from scipy.io import savemat, loadmat
from lda.parameters import *


class AnalyzeLDA:

    def __init__(self, folder, reanalyze=False, directory=DIRECTORY):
        self.folder = os.path.join(directory, folder)
        self.air_folder = os.path.join(directory, AIR_FOLDER)
        self.dark_folder = os.path.join(directory, DARK_FOLDER)

        self.reanalyze = reanalyze
        self.correct_air_and_dark_scans()

        self.raw_data = os.path.join(self.folder, 'Data', 'data.npy')
        self.air_data = np.load(os.path.join(self.air_folder, 'Data', 'data_corr.npy')) / 60
        self.dark_data = np.load(os.path.join(self.dark_folder, 'Data', 'data.npy')) / 60

        self.corr_data = os.path.join(self.folder, 'Data', 'data_corr.npy')
        self.corr_data_mat = os.path.join(self.folder, 'Data', 'data_corr.mat')

        if not os.path.exists(self.corr_data) or reanalyze:
            temp_data = self.intensity_correction(self.correct_dead_pixels())
            print(len(np.argwhere(np.isnan(temp_data))))
            np.save(self.corr_data, temp_data)
            savemat(self.corr_data_mat, {'data': temp_data, 'label': 'corrected_data'})

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
        temp_data = gen.correct_dead_pixels(np.load(self.raw_data), DEAD_PIXEL_MASK)
        if len(np.shape(temp_data)) == 3:
            temp_data = np.expand_dims(temp_data, axis=0)
        data_nan = np.argwhere(np.isnan(temp_data))
        for nan_coords in data_nan:
            nan_coords = tuple(nan_coords)
            frame = nan_coords[0]
            img_bin = nan_coords[-1]
            pixel = nan_coords[-3:-1]
            temp_data[nan_coords] = gen.get_average_pixel_value(temp_data[frame, :, :, img_bin], pixel, np.ones((24, 576)))
        return temp_data

    def correct_air_and_dark_scans(self):
        """
        This corrects the air and dark data for known dead pixels.
        """
        airpath = os.path.join(self.air_folder, 'Data', 'data_corr.npy')
        darkpath = os.path.join(self.dark_folder, 'Data', 'data_corr.npy')

        raw_air = np.load(os.path.join(self.air_folder, 'Data', 'data.npy'))
        raw_dark = np.load(os.path.join(self.dark_folder, 'Data', 'data.npy'))

        if not os.path.exists(airpath) or self.reanalyze:
            temp_air = gen.correct_dead_pixels(raw_air, dead_pixel_mask=DEAD_PIXEL_MASK)
            air_nan = np.argwhere(np.isnan(temp_air))
            for nan_coords in air_nan:
                nan_coords = tuple(nan_coords)
                img_bin = nan_coords[-1]
                pixel = nan_coords[0:-1]
                temp_air[nan_coords] = gen.get_average_pixel_value(temp_air[:, :, img_bin], pixel, np.ones((24, 576)))
            np.save(airpath, temp_air)

        if not os.path.exists(darkpath) or self.reanalyze:
            temp_dark = gen.correct_dead_pixels(raw_dark, dead_pixel_mask=DEAD_PIXEL_MASK)
            dark_nan = np.argwhere(np.isnan(temp_dark))
            for nan_coords in dark_nan:
                nan_coords = tuple(nan_coords)
                img_bin = nan_coords[-1]
                pixel = nan_coords[0:-1]
                temp_dark[nan_coords] = gen.get_average_pixel_value(temp_dark[:, :, img_bin], pixel, np.ones((24, 576)))
            np.save(darkpath, temp_dark)

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

    def __init__(self, folder, num_proj, reanalyze=False, directory=DIRECTORY):
        super().__init__(folder, reanalyze=reanalyze, directory=directory)

        self.num_proj = num_proj

        self.filt_data = os.path.join(self.folder, 'Data', 'proj_filt.npy')
        self.filt_data_mat = os.path.join(self.folder, 'Data', 'proj_filt.mat')

        temp_data = np.load(self.corr_data)
        # This will cut the projection down to the correct number if there are more than necessary
        if num_proj != len(temp_data):
            diff = abs(num_proj - len(temp_data))
            new_data = temp_data[int(np.ceil(diff / 2)):len(temp_data) - diff // 2]
            np.save(self.corr_data, new_data)
            savemat(self.corr_data_mat, {'data': new_data, 'label': 'data_corr'})

