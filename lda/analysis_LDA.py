import general_functions as gen
import mask_functions as msk
from scipy.io import savemat, loadmat
import matplotlib.pyplot as plt
from lda.parameters import *
from lda.get_corrected_array import pixel_corr


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

    def __init__(self, folder, num_proj, duration=180, airscan_time=60, reanalyze=True, directory=DIRECTORY,
                 sub_folder='phantom_scan', air_folder='airscan_60s', dark_folder='darkscan_60s'):
        super().__init__(folder, duration/num_proj, airscan_time=airscan_time, reanalyze=reanalyze, directory=directory,
                         sub_folder=sub_folder, air_folder=air_folder, dark_folder=dark_folder)

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

    def __init__(self, folder, water_slice=12, kedge_bins=None, high_conc=None, algorithm=None, reanalyze=True):
        self.folder = os.path.join(DIRECTORY, folder, 'phantom_scan')
        self.water_slice = water_slice
        self.kedge_bins = kedge_bins
        self.high_concentration = np.array(high_conc)

        # Create the folder to house the normalized data, if necessary
        os.makedirs(os.path.join(self.folder, 'Norm CT'), exist_ok=True)

        # Assess where the raw CT data is, and set the path to the data
        if algorithm:
            print('alg')
            self.data = os.path.join(self.folder, 'CT', f'{algorithm}_CT.npy')
            file_append = f'_{algorithm}'
        else:
            print('no alg')
            self.data = os.path.join(self.folder, 'CT', 'CT.npy')
            file_append = ''

            if not os.path.exists(self.data):
                print('redo')
                mat_data = loadmat(os.path.join(self.folder, 'CT', 'CT.mat'))['ct_img']
                np.save(self.data, mat_data)
                del mat_data

        # # Check for the right water slice
        # data1 = np.load(self.data)
        # for i in range(8, 18):
        #     fig, ax = plt.subplots(2, 3, figsize=(12, 6))
        #     ax[0, 0].imshow(data1[0, i], cmap='gray', vmin=0, vmax=0.2)
        #     ax[0, 0].set_title(f'{i}')
        #     ax[0, 1].imshow(data1[1, i], cmap='gray', vmin=0, vmax=0.2)
        #     ax[0, 2].imshow(data1[3, i], cmap='gray', vmin=0, vmax=0.2)
        #     ax[1, 0].imshow(data1[4, i], cmap='gray', vmin=0, vmax=0.2)
        #     ax[1, 1].imshow(data1[6, i], cmap='gray', vmin=0, vmax=0.2)
        #     plt.show()
        #     plt.pause(1)
        #     plt.close()
        #
        # self.water_slice = int(input("Enter the good slice: "))

        # Save only the regular CT data in the Norm CT folder, if not already there
        self.ct_path = os.path.join(self.folder, 'Norm CT', f'CT_norm{file_append}.npy')
        raw_data = np.load(self.data)
        self.norm_data = raw_data[-1]
        del raw_data
        np.save(self.ct_path, self.norm_data)

        # Get the mask for the water vials, or load it if already assessed
        self.water_path = os.path.join(self.folder, f'water_mask{file_append}.npy')
        if reanalyze or not os.path.exists(self.water_path):
            self.water_mask = self.get_water_masks()
            np.save(self.water_path, self.water_mask)
        else:
            self.water_mask = np.load(self.water_path)

        # Get the mask for the air value, or load if already assessed
        self.air_path = os.path.join(self.folder, f'air_mask{file_append}.npy')
        if reanalyze or not os.path.exists(self.air_path):
            self.air_mask = msk.square_ROI(self.norm_data[self.water_slice])
            np.save(self.air_path, self.air_mask)
        else:
            self.air_mask = np.load(self.air_path)

        # Get the background mask for the air value, or load if already assessed
        self.back_path = os.path.join(self.folder, f'back_mask{file_append}.npy')
        if reanalyze or not os.path.exists(self.back_path):
            self.back_mask = self.get_water_masks()
            np.save(self.back_path, self.back_mask)
        else:
            self.back_mask = np.load(self.back_path)

        # Get the masks for the contrast vials, or load them if already assessed

        if self.kedge_bins:
            if len(self.kedge_bins) > 2:
                self.contrast_path_1 = os.path.join(self.folder, f'contrast_masks{file_append}.npy')
                self.contrast_path_2 = os.path.join(self.folder, f'contrast_masks{file_append}_2.npy')

                if reanalyze or not os.path.exists(self.contrast_path_2):
                    self.contrast_masks_2 = self.get_contrast_masks()
                    np.save(self.contrast_path_2, self.contrast_masks_2)
                else:
                    self.contrast_masks_2 = np.load(self.contrast_path_2)

                # Save only the K-edge subtracted data in the Norm CT folder, if not already there, or not done
                self.kedge_path_2 = os.path.join(self.folder, 'Norm CT', f'K-edge{file_append}_2.npy')
                raw_data = np.load(self.data)
                self.kedge_data_2 = raw_data[kedge_bins[3]] - raw_data[kedge_bins[2]]
                del raw_data
                np.save(self.kedge_path_2, self.kedge_data_2)

                self.k_water_value_2 = np.nanmean(self.kedge_data_2[self.water_slice] * self.water_mask)
                self.k_edge_high_2 = np.nanmean(self.kedge_data_2[self.water_slice] * self.contrast_masks_2[0])

            else:
                self.contrast_path_1 = os.path.join(self.folder, f'contrast_masks{file_append}.npy')
                self.contrast_path_2 = None

            if reanalyze or not os.path.exists(self.contrast_path_1):
                self.contrast_masks_1 = self.get_contrast_masks()
                np.save(self.contrast_path_1, self.contrast_masks_1)
            else:
                self.contrast_masks_1 = np.load(self.contrast_path_1)

            # Save only the K-edge subtracted data in the Norm CT folder, if not already there, or not done
            self.kedge_path_1 = os.path.join(self.folder, 'Norm CT', f'K-edge{file_append}.npy')
            raw_data = np.load(self.data)
            self.kedge_data_1 = raw_data[kedge_bins[1]] - raw_data[kedge_bins[0]]
            del raw_data
            np.save(self.kedge_path_1, self.kedge_data_1)

            self.k_water_value_1 = np.nanmean(self.kedge_data_1[self.water_slice] * self.water_mask)
            self.k_edge_high_1 = np.nanmean(self.kedge_data_1[self.water_slice] * self.contrast_masks_1[0])

        self.water_value = np.nanmean(self.norm_data[self.water_slice] * self.water_mask)
        self.air_value = np.nanmean(self.norm_data[self.water_slice] * self.air_mask)

        # Create a file to see if data has been normalized already
        self.norm_check = os.path.join(self.folder, 'Norm CT', f'norm_check{file_append}.npy')
        if reanalyze or not os.path.exists(self.norm_check):
            self.normalize_HU()
            if self.kedge_bins:
                self.normalize_kedge()
                if len(self.kedge_bins) > 2:
                    self.normalize_kedge(k_num=2)
            np.save(self.norm_check, np.array([]))

    def get_water_masks(self):
        """
        :return:
        """
        img = self.norm_data[self.water_slice]

        # Get the large vials filled with water
        masks = msk.phantom_ROIs(img, radius=7)

        # Sum all the individual vial masks together into one mask that grabs all water vials
        masks = np.nansum(masks, axis=0)
        masks[masks == 0] = np.nan

        return masks

    def get_contrast_masks(self):
        """
        :return:
        """
        img = self.norm_data[self.water_slice]

        # Get the vials filled with contrast from highest to lowest concentration
        masks = msk.phantom_ROIs(img, radius=5)

        return masks

    def normalize_HU(self):
        """
        Normalize the EC bin to HU
        """
        # Normalize to HU
        self.norm_data = 1000/(self.water_value - self.air_value) * np.subtract(self.norm_data, self.water_value)

        # Get rid of any nan values
        self.norm_data[np.isnan(self.norm_data)] = -1000
        np.save(self.ct_path, self.norm_data)

    def normalize_kedge(self, k_num=1):
        """
        Normalize the K-edge subtraction image linearly with concentration
        """

        # Normalize the K-edge data
        if k_num == 1:
            self.kedge_data_1 = (self.kedge_data_1 - self.k_water_value_1) * self.high_concentration[0] / \
                                (self.k_edge_high_1 - self.k_water_value_1)
            np.save(self.kedge_path_1, self.kedge_data_1)
        else:
            self.kedge_data_2 = (self.kedge_data_2 - self.k_water_value_2) * self.high_concentration[1] / \
                                (self.k_edge_high_2 - self.k_water_value_2)
            np.save(self.kedge_path_2, self.kedge_data_2)
