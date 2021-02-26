import general_functions as gen
import mask_functions as msk
from scipy.io import savemat, loadmat
from lda.parameters import *
from lda.get_corrected_array import pixel_corr


class ReconLDA:

    def __init__(self, folder, duration, airscan_time=60, reanalyze=False, directory=DIRECTORY):
        self.folder = os.path.join(directory, folder)
        self.air_folder = os.path.join(directory, AIR_FOLDER)
        self.dark_folder = os.path.join(directory, DARK_FOLDER)

        self.reanalyze = reanalyze
        self.correct_air_and_dark_scans()

        self.raw_data = os.path.join(self.folder, 'Data', 'data.npy')
        self.air_data = np.load(os.path.join(self.air_folder, 'Data', 'data_corr.npy')) / (airscan_time/duration)
        self.dark_data = np.load(os.path.join(self.dark_folder, 'Data', 'data_corr.npy')) / (airscan_time/duration)
        print(airscan_time/duration)

        self.corr_data = os.path.join(self.folder, 'Data', 'data_corr.npy')
        self.corr_data_mat = os.path.join(self.folder, 'Data', 'data_corr.mat')

        if not os.path.exists(self.corr_data) or reanalyze:
            temp_data = self.intensity_correction(self.correct_dead_pixels())
            print(len(np.argwhere(np.isnan(temp_data))))
            np.save(self.corr_data, temp_data)
            # savemat(self.corr_data_mat, {'data': temp_data, 'label': 'corrected_data'})

        self.bg_mask = None

    def intensity_correction(self, data):
        """
        This function corrects flatfield data to show images, -ln(I/I0), I is the intensity of the data, I0 is the
        intensity in an airscan
        """
        temp_data = gen.intensity_correction(data, self.air_data, self.dark_data)
        nan_coords = np.argwhere(np.isnan(temp_data))
        print(len(nan_coords))
        for coords in nan_coords:
            coords = tuple(coords)
            frame = coords[0]
            img_bin = coords[-1]
            pixel = coords[-3:-1]
            temp_data[coords] = gen.get_average_pixel_value(temp_data[frame, :, :, img_bin], pixel, np.ones((24, 576)))
        return temp_data

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
            np.save(airpath, temp_air)

        if not os.path.exists(darkpath) or self.reanalyze:
            temp_dark = gen.correct_dead_pixels(raw_dark, dead_pixel_mask=DEAD_PIXEL_MASK)
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


class ReconCT(ReconLDA):

    def __init__(self, folder, num_proj, duration=180, airscan_time=60, reanalyze=True, directory=DIRECTORY):
        super().__init__(folder, duration/num_proj, airscan_time=airscan_time, reanalyze=reanalyze, directory=directory)

        self.num_proj = num_proj

        temp_data = np.load(self.corr_data)
        # np.save(os.path.join(DIRECTORY, folder, 'Data', 'data_corr_before.npy'), temp_data)
        print(np.nanmedian(np.sum(temp_data, axis=0), axis=(0, 1)))
        # Correct for pixel non-uniformities
        temp_data = pixel_corr(temp_data)
        print(np.nanmedian(np.sum(temp_data, axis=0), axis=(0, 1)))

        # This will cut the projection down to the correct number if there are more than necessary
        if num_proj != len(temp_data):
            diff = abs(num_proj - len(temp_data))
            new_data = temp_data[int(np.ceil(diff / 2)):len(temp_data) - diff // 2]

            np.save(self.corr_data, new_data)
            savemat(self.corr_data_mat, {'data': new_data, 'label': 'data_corr'})


class AnalyzeCT:

    def __init__(self, folder, thresholds, contrast, water_slice=13, algorithm=None):
        self.folder = os.path.join(DIRECTORY, folder, 'phantom_scan', 'CT')
        self.thresholds = thresholds
        self.contrast = contrast
        self.water_slice = water_slice

        if algorithm:
            self.data = os.path.join(self.folder, f'{algorithm}CT.npy')
        else:
            self.data = os.path.join(self.folder, 'CT.npy')

            if not os.path.exists(self.data):
                mat_data = loadmat(os.path.join(self.folder, 'CT.mat'))['ct_img']
                np.save(self.data, mat_data)

        self.data_shape = np.array(np.shape(np.load(self.data)))

        if self.data_shape != [7, 24, 576, 576]:
            np.save(self.data, np.transpose(np.load(self.data), axes=(0, 3, 1, 2)))
            self.data_shape = np.array([7, 24, 576, 576])

        self.mask_water = self.get_water_masks()


    def get_water_masks(self):
        img = np.load(self.data)[6, self.water_slice]

        # Get the large vials filled with water
        masks = msk.phantom_ROIs(img, radius=8)

        # Sum all the inidivual vial masks together into one mask that grabs all water vials
        masks = np.nansum(masks, axis=0)
        masks[masks == 0] = np.nan

        return masks

    def get_contrast_masks(self):



    @staticmethod
    def norm_individual(image, water_value):
        """
        Normalize an individual slice
        :param image: The image to normalize
        :param water_value: The water value to normalize to 0 HU
        :return:
        """
        # Normalize to HU
        image = 1000 * np.divide((np.subtract(image, water_value)), water_value)
        # Get rid of any nan values
        image[np.isnan(image)] = -1000

        return image