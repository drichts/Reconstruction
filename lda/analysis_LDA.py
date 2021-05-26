import numpy as np
from scipy.io import savemat, loadmat
import matplotlib.pyplot as plt
from lda.parameters import *
from lda.get_corrected_array import pixel_corr
import general_functions as gen
import mask_functions as msk


class ReconLDA:

    def __init__(self, folder, duration, airscan_time=60, reanalyze=False, directory=DIRECTORY,
                 sub_folder='phantom_scan', air_folder='airscan_60s', dark_folder='darkscan_60s'):

        self.folder = os.path.join(directory, folder, sub_folder)
        self.air_folder = os.path.join(directory, folder, air_folder)
        self.dark_folder = os.path.join(directory, folder, dark_folder)

        self.reanalyze = reanalyze
        self.correct_air_and_dark_scans()

        self.raw_data = os.path.join(self.folder, 'Data', 'data.npy')
        self.air_data = np.load(os.path.join(self.air_folder, 'Data', 'data_corr.npy')) / (airscan_time/duration)
        self.dark_data = np.load(os.path.join(self.dark_folder, 'Data', 'data_corr.npy')) / (airscan_time/duration)
        print(airscan_time/duration)

        self.num_bins = np.shape(np.load(self.raw_data))
        if len(self.num_bins) == 3:
            self.num_bins = 1
        else:
            self.num_bins = self.num_bins[-1]

        self.corr_data = os.path.join(self.folder, 'Data', 'data_corr.npy')
        self.corr_data_mat = os.path.join(self.folder, 'Data', 'data_corr.mat')

        if not os.path.exists(self.corr_data) or reanalyze:
            temp_data = self.intensity_correction(self.correct_dead_pixels())
            print(len(np.argwhere(np.isnan(temp_data))))
            np.save(self.corr_data, temp_data)
            savemat(self.corr_data_mat, {'data': temp_data, 'label': 'corrected_data'}, do_compression=True)

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
        return np.squeeze(temp_data)

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

    def __init__(self, folder, num_proj=720, duration=180, airscan_time=60, top=False, corr_rings=True, reanalyze=True,
                 directory=DIRECTORY, sub_folder='phantom_scan', air_folder='airscan_60s', dark_folder='darkscan_60s'):
        super().__init__(folder, duration/num_proj, airscan_time=airscan_time, reanalyze=reanalyze, directory=directory,
                         sub_folder=sub_folder, air_folder=air_folder, dark_folder=dark_folder)

        self.num_proj = num_proj

        temp_data = np.load(self.corr_data)

        print(np.nanmedian(np.sum(temp_data, axis=0), axis=(0, 1)))

        # Correct for pixel non-uniformities
        if corr_rings:
            temp_data = pixel_corr(temp_data, num_bins=self.num_bins, top=top)
            print(np.nanmedian(np.sum(temp_data, axis=0), axis=(0, 1)))

        # This will cut the projection down to the correct number if there are more than necessary
        if num_proj != len(temp_data):
            diff = abs(num_proj - len(temp_data))
            new_data = temp_data[int(np.ceil(diff / 2)):len(temp_data) - diff // 2]

            np.save(self.corr_data, new_data)
            savemat(self.corr_data_mat, {'data': new_data, 'label': 'data_corr'}, do_compression=True)


class AnalyzeCT:

    def __init__(self, folder, water_slice=12, water_val=None, air_val=None, algorithm=None, reanalyze=False,
                 sub_folder='phantom_scan'):
        self.folder = os.path.join(DIRECTORY, folder, sub_folder)
        self.water_slice = water_slice
        self.reanalyze = reanalyze

        # Check if folder actually exists before creating new subfolders within it
        if os.path.exists(self.folder):
            print('Folder exists, analysis will continue.')
        else:
            raise Exception(f'Folder does not exist: {self.folder}')

        # Create the folder to house the normalized data, if necessary
        os.makedirs(os.path.join(self.folder, 'Norm CT'), exist_ok=True)

        # Assess where the raw CT data is, and set the path to the data
        if algorithm:
            self.data = os.path.join(self.folder, 'CT', f'{algorithm}_CT.npy')
            self.file_append = f'_{algorithm}'
        else:
            self.data = os.path.join(self.folder, 'CT', 'CT.npy')
            self.file_append = ''

            if os.path.exists(os.path.join(self.folder, 'CT', 'CT.mat')):
                mat_data = loadmat(os.path.join(self.folder, 'CT', 'CT.mat'))['ct_img']
                np.save(self.data, mat_data)
                del mat_data
            else:
                raise Exception(f'Data file does not exist: {self.data} \nand MAT data file does not exist.')

        # Check for the right water slice
        if not water_slice:
            data1 = np.load(self.data)
            for i in range(9, 14):
                fig, ax = plt.subplots(1, 3, figsize=(12, 6))
                ax[0].imshow(data1[0, i], cmap='gray', vmin=0, vmax=0.1)
                ax[0].set_title(f'{i}')
                ax[1].imshow(data1[1, i], cmap='gray', vmin=0, vmax=0.1)
                ax[2].imshow(data1[2, i], cmap='gray', vmin=0, vmax=0.1)
                plt.show()
                plt.pause(0.5)
                plt.close()

            self.water_slice = int(input("Enter the good slice: "))

        # Save only the regular CT data in the Norm CT folder, if not already there
        self.ct_path = os.path.join(self.folder, 'Norm CT', f'CT_norm{self.file_append}.npy')
        self.norm_data = np.load(self.data)
        self.num_bins = len(self.norm_data)
        print(self.num_bins)
        np.save(self.ct_path, self.norm_data)

        # Get the mask for the water vials
        self.water_path = os.path.join(self.folder, f'water_mask{self.file_append}.npy')
        self.water_mask = self.get_masks(mask_type=0, path=self.water_path, message_num=9)

        # Get the mask for the air ROI
        self.air_path = os.path.join(self.folder, f'air_mask{self.file_append}.npy')
        self.air_mask = self.get_masks(mask_type=1, path=self.air_path, message_num=8)

        # Get the background mask for the phantom ROIs
        # self.back_path = os.path.join(self.folder, f'back_mask{self.file_append}.npy')
        # self.back_mask = self.get_masks(mask_type=0, path=self.back_path, message_num=10)

        # Find the water and air values for each of the bins
        self.water_value = np.zeros(self.num_bins)
        self.air_value = np.zeros(self.num_bins)

        # Go through each bin and find the mean value in the water and air ROIs
        for i in range(self.num_bins):
            if water_val is not None:
                self.water_value[i] = water_val[i]
            else:
                self.water_value[i] = np.nanmean(self.norm_data[i, self.water_slice] * self.water_mask)
            if air_val is not None:
                self.air_value[i] = air_val[i]
            else:
                self.air_value[i] = np.nanmean(self.norm_data[i, self.water_slice] * self.air_mask)

        # Normalize the data
        self.normalize_HU()

    def get_masks(self, mask_type, path, message_num):
        """
        Allows the user to click on an image to define ROIs depending on the type of ROI desired
        :param mask_type: int
                0 or 1; 0 is for multiple circular ROIs that will be a single mask, 1 is for a single square ROI
        :param path: str
                The full path to save and load the mask from
        :param message_num: int
                The message to display on the top of the plot you click on
        :return: ndarray
                The desired mask array with 1's where inside the ROI(s) and nan everywhere else (2D)
        """

        # Check if the ROIs need to be reanalyzed or if they exist already
        if self.reanalyze or not os.path.exists(path):

            img = self.norm_data[-1, self.water_slice]

            # Whether to grab multiple circular ROIs or a single square ROI
            if mask_type == 0:
                # Click the vials or desired ROIs
                masks = msk.phantom_ROIs(img, radius=7, message_num=message_num)

                # Sum all the individual vial masks together into one mask that grabs all ROIs
                masks = np.nansum(masks, axis=0)
                masks[masks == 0] = np.nan
            else:
                masks = msk.square_ROI(img, message_num=8)

            # Save the data for later
            np.save(path, masks)

        else:
            masks = np.load(path)

        return masks

    def normalize_HU(self):
        """
        Normalize the EC bin to HU
        """
        # Normalize to HU
        for i in range(self.num_bins):
            self.norm_data[i] = 1000/(self.water_value[i] - self.air_value[i]) \
                                * np.subtract(self.norm_data[i], self.water_value[i])

        # Get rid of any nan values
        self.norm_data[np.isnan(self.norm_data)] = -1000
        np.save(self.ct_path, self.norm_data)


class AnalyzeOneKedge(AnalyzeCT):

    def __init__(self, folder, kedge_bins, high_conc_real, conc_vals=None, contrast='Au', water_slice=12,
                 water_val=None, air_val=None, algorithm=None, reanalyze=False, sub_folder='phantom_scan'):
        super().__init__(folder, water_slice=water_slice, water_val=water_val, air_val=air_val, algorithm=algorithm,
                         reanalyze=reanalyze, sub_folder=sub_folder)

        # The physical high concentration
        self.high_conc_real = high_conc_real

        # Load or create the contrast vial masks
        self.contrast_path = os.path.join(self.folder, f'contrast_masks{self.file_append}_{contrast}.npy')
        self.contrast_masks = self.get_contrast_masks(path=self.contrast_path)

        # Save only the K-edge subtracted data in the Norm CT folder, if not already there, or not done
        self.kedge_path = os.path.join(self.folder, 'Norm CT', f'K-edge{self.file_append}_{contrast}.npy')
        raw_data = np.load(self.data)

        self.kedge_data = raw_data[kedge_bins[1]] - raw_data[kedge_bins[0]]
        del raw_data
        np.save(self.kedge_path, self.kedge_data)

        # Normalize the K-edge data to the concentration values (in another image) given or the image itself
        # These values are the image values of the highest and lowest concentrations vials before normalization
        if conc_vals is not None:
            self.k_water_val = conc_vals[1]
            self.k_high_conc_val = conc_vals[0]
        else:
            self.k_water_val = np.nanmean(self.kedge_data[self.water_slice] * self.water_mask)
            # self.k_water_val = np.nanmean(self.kedge_data[self.water_slice] * self.contrast_masks[-1])  # This is for when the medium isn't water
            self.k_high_conc_val = np.nanmean(self.kedge_data[self.water_slice] * self.contrast_masks[0])

        # Normalize
        self.normalize_kedge()

    def get_contrast_masks(self, path):
        """
        Allows the user to click on an image to define ROIs for each of the contrast vials
        :param path: str
                The full path to save and load the mask from
        :return: ndarray
                The desired mask array with 1's where inside the ROI(s) and nan everywhere else (2D)
        """
        img = self.norm_data[-1, self.water_slice]

        if self.reanalyze or not os.path.exists(path):
            masks = msk.phantom_ROIs(img, radius=5, message_num=11)
            np.save(path, masks)
        else:
            masks = np.load(path)

        return masks

    def normalize_kedge(self):
        """
        Normalize the K-edge subtraction image linearly with concentration
        """

        # Normalize the K-edge data
        self.kedge_data = (self.kedge_data - self.k_water_val) * self.high_conc_real / \
                          (self.k_high_conc_val - self.k_water_val)
        np.save(self.kedge_path, self.kedge_data)


class AnalyzeTwoKedge(AnalyzeOneKedge):
    def __init__(self):
        pass
