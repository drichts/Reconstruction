import numpy as np
import matplotlib.pyplot as plt
import os
from redlen.redlen_analysis import RedlenAnalyze
import mask_functions as grm
import _pickle as pickle


class VisibilityError(Exception):
    pass


class AnalyzeUniformity(RedlenAnalyze):

    # def __new__(cls, folder, air_folder, filepath=None, test_num=1, mm='M20358_D32',
    #             load_dir=r'X:\TEST LOG\MINI MODULE\Canon', save_dir=r'C:\Users\10376\Documents\Phantom Data'):
    #     if filepath:
    #         with open(filepath, 'rb') as f:
    #             unpickler = pickle.Unpickler(f)
    #             inst = unpickler.load()
    #             f.close()
    #         if not isinstance(inst, cls):
    #             raise TypeError('Unpickled object is not of type {}'.format(cls))
    #         print('Loaded')
    #     else:
    #         inst = super(AnalyzeUniformity, cls).__new__(cls)
    #         print('New')
    #     return inst

    def __init__(self, folder, air_folder, test_num=1, mm='M20358_D32', load_dir=r'X:\TEST LOG\MINI MODULE\Canon',
                 save_dir=r'C:\Users\10376\Documents\Phantom Data'):

        super().__init__(folder, test_num, mm, 'UNIFORMITY', load_dir)
        self.thresholds = []
        self.pxp = np.array([1, 2, 3, 4, 6, 8, 12])
        self.frames = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 50, 100, 250, 500, 1000])

        self.air_data = RedlenAnalyze(air_folder, test_num, mm, 'UNIFORMITY', load_dir)

        self.save_dir = os.path.join(save_dir, 'Uniformity', folder)
        print(self.save_dir)
        os.makedirs(self.save_dir, exist_ok=True)
        self.filename = f'Run{test_num}Data.pk1'

        self.masks = []
        self.bg = []

        if 'Masks.npz' in os.listdir(self.save_dir):
            self.masks = np.load(os.path.join(self.save_dir, 'Masks.npz'), allow_pickle=True)['mask']
            self.bg = np.load(os.path.join(self.save_dir, 'Masks.npz'), allow_pickle=True)['bg']
        else:
            self.get_masks()

        # CNR vs. time, noise vs. time <pixels, time, bin, value or error (0 or 1)>
        self.cnr_time, self.noise_time = self.analyze_cnr_noise()
        # Reorganize to <pixels, bin, value or error, time>
        self.cnr_time = np.transpose(self.cnr_time, axes=(0, 2, 3, 1))
        self.noise_time = np.transpose(self.noise_time, axes=(0, 2, 3, 1))

        # Save the object
        self.save_object(self.filename)

    def get_masks(self):
        """This function gets the contrast mask and background mask for all pixel aggregations"""
        self.visible_bin = self.test_visibility()
        val = input('Is the phantom small? (y/n)')
        if val is 'y':
            self.small_phantom = True
        else:
            self.small_phantom = False

        for i, pixel in enumerate(self.pxp):
            if pixel == 1:
                tempdata = self.data_a0
                tempair = self.air_data.data_a0
            else:
                tempdata = self.sumpxp(self.data_a0, pixel)
                tempair = self.sumpxp(self.air_data.data_a0, pixel)
            image = self.intensity_correction(tempdata, tempair)
            image = np.sum(image, axis=1)
            tempmask, tempbg = self.choose_mask_types(image[self.visible_bin], pixel)

            self.masks.append(tempmask)
            self.bg.append(tempbg)

        np.savez(os.path.join(self.save_dir + '/Masks.npz'), mask=self.masks, bg=self.bg)

    def test_visibility(self):
        """
        This function will test if your contrast area is visible in the one of the energy bins
        :return: The bin to use
        """

        image = np.squeeze(self.intensity_correction(self.data_a0, self.air_data.data_a0))
        image = np.sum(image, axis=1)  # Sum all views

        bins = np.arange(len(image)-1, -1, -1)  # The bin options from EC down
        for b in bins:
            plt.imshow(image[b])
            plt.pause(1)
            plt.close()
            val1 = input('Was the contrast visible? (y/n)')
            if val1 is 'y':
                return b
        return VisibilityError('The phantom is not visible in any energy bin.')

    def choose_mask_types(self, image, pixels):
        """
        This function chooses the mask function to call based on if you are aggregating lots of pixels together and if
        the contrast area is small. If the contrast area is small, the contrast mask will chosen pixel by pixel,
        otherwise it will be selected by choosing the corners of a rectangle. The same is true for the background mask
        if you aggregate a lot of pixels, you need to choose the background mask pixel by pixel
        :param image: 2D ndarray
                    The image to mask
        :param pixels: int
                    The number of pixels being aggregated together
        :return:
        """
        # First get the contrast mask, if it is a small contrast area (<2mm) choose the ROI pixel by pixel
        # It will ask you after selecting if the ROI is acceptable, anything but y will allow you to redefine the ROI
        continue_flag = True
        if self.small_phantom or pixels > 5:
            while continue_flag:
                mask = grm.single_pixels_mask(image)
                val = input('Were the ROIs acceptable? (y/n)')
                if val is 'y':
                    continue_flag = False
        else:
            while continue_flag:
                mask = grm.square_ROI(image)
                val = input('Were the ROIs acceptable? (y/n)')
                if val is 'y':
                    continue_flag = False

        # Choose the background mask in the same way, if you are aggregating more pixels than 5,
        # you will select pixel by pixel
        continue_flag = True
        if pixels < 5:
            while continue_flag:
                bg = grm.square_ROI(image)
                val = input('Were the ROIs acceptable? (y/n)')
                if val is 'y':
                    continue_flag = False
        else:
            while continue_flag:
                bg = grm.single_pixels_mask(image)
                val = input('Were the ROIs acceptable? (y/n)')
                if val is 'y':
                    continue_flag = False

        return mask, bg

    def analyze_cnr_noise(self):
        """
        This function calculates the CNR and noise in each bin at each of the pixel aggregations values and at a number
        of different acquisition times
        :return:
        """
        cnr_vals = np.zeros([len(self.pxp), len(self.frames), self.num_bins, 2])
        noise = np.zeros([len(self.pxp), len(self.frames), self.num_bins, 2])

        # Aggregate the number of pixels in pxp squared
        for p, pix in enumerate(self.pxp):
            if pix == 1:
                data_pxp = np.squeeze(self.data_a0)
                air_pxp = np.squeeze(self.air_data.data_a0)
            else:
                data_pxp = np.squeeze(self.sumpxp(self.data_a0, pix))  # Aggregate the pixels
                air_pxp = np.squeeze(self.sumpxp(self.air_data.data_a0, pix))

            # Collect the frames aggregated over, the noise and cnr and save
            cnr_vals[p], noise[p] = self.avg_cnr_noise_over_all_frames(data_pxp, air_pxp, self.masks[p], self.bg[p])

        return cnr_vals, noise

    def avg_cnr_noise_over_frames(self, data, airdata, mask, bg_mask, frame):
        """
        This function will take the data and airscan and calculate the CNR every number of frames and then avg, will
        also give the CNR error as well. Also will calculate the avg noise
        :param data: 4D ndarray, <counters, views, rows, columns>
                    The phantom data
        :param airdata: 4D ndarray, <counters, views, rows, columns>
                    The airscan
        :param mask: 2D ndarray
                    The mask of the contrast area
        :param bg_mask: 2D ndarray
                    The mask of the background
        :param frame: int
                    The number of frames to avg together
        :return: four lists with the cnr, cnr error, noise, and std of the noise in each of the bins
                (usually 13 elements)
        """
        cnr_val = np.zeros([len(data), int(1000/frame)])
        cnr_err = np.zeros([len(data), int(1000/frame)])  # array for the cnr error
        noise = np.zeros([len(data), int(1000/frame)])  # Same for noise

        # Go over the data views in jumps of the number of frames
        for i, data_idx in enumerate(np.arange(0, 1001-frame, frame)):
            if frame == 1:
                tempdata = data[:, data_idx]  # Grab the next view
                tempair = airdata[:, data_idx]
            else:
                # Grab the sum of the next 'frames' views
                tempdata = np.sum(data[:, data_idx:data_idx + frame], axis=1)
                tempair = np.sum(airdata[:, data_idx:data_idx + frame], axis=1)

            corr_data = self.intensity_correction(tempdata, tempair)  # Correct for air

            # Go through each bin and calculate CNR
            for j, img in enumerate(corr_data):
                cnr_val[j, i], cnr_err[j, i] = self.cnr(img, mask, bg_mask)
                noise[j, i] = np.nanstd(img*bg_mask)  # Get noise as fraction of mean background

        # Average over the frames
        cnr_val = np.mean(cnr_val, axis=1)
        cnr_err = np.mean(cnr_err, axis=1)
        noise_std = np.std(noise, axis=1)
        noise = np.mean(noise, axis=1)

        return cnr_val, cnr_err, noise, noise_std

    def avg_cnr_noise_over_all_frames(self, data, airdata, mask, bg_mask):
        """
        This function will take the data and airscan and calculate the CNR and CNR error over all the frames in the list
        :param data: 4D ndarray, <counters, views, rows, columns>
                    The phantom data
        :param airdata: 4D ndarray, <counters, views, rows, columns>
                    The airscan
        :param mask: 2D ndarray
                    The mask of the contrast area
        :param bg_mask: 2D ndarray
                    The mask of the background
        :return:
        """
        cnr_frames = np.zeros([len(self.frames), len(data), 2])  # The CNR and CNR error over all frames in the list
        noise_frames = np.zeros([len(self.frames), len(data), 2])  # Same for noise

        for i in np.arange(len(self.frames)):
            # Calculate the CNR and error
            c, ce, n, ne = self.avg_cnr_noise_over_frames(data, airdata, mask, bg_mask, self.frames[i])
            cnr_frames[i, :, 0] = c   # The ith frames, set first column equal to cnr
            cnr_frames[i, :, 1] = ce  # The ith frames, set second column equal to cnr error
            noise_frames[i, :, 0] = n  # Same for noise, 1st column
            noise_frames[i, :, 1] = ne  # Noise error, 2nd column

        return cnr_frames, noise_frames

    def avg_contrast_over_frames(self, data, airdata, mask, bg_mask, frame):
        """
        This function will take the data and airscan and calculate the contrast as a fraction of mean background for
        every number of frames and then avg
        :param data: 4D ndarray, <counters, views, rows, columns>
                    The phantom data
        :param airdata: 4D ndarray, <counters, views, rows, columns>
                    The airscan
        :param mask: 2D ndarray
                    The mask of the contrast area
        :param bg_mask: 2D ndarray
                    The mask of the background
        :param frame: int
                    The number of frames to avg together
        :return: four lists with the cnr, cnr error, noise, and std of the noise in each of the bins
                    (usually 13 elements)
        """
        # Array to hold the cnr for the number of times it will be calculated
        contrast = np.zeros([len(data), int(1000/frame)])

        # Go over the data views in jumps of the number of frames
        for i, data_idx in enumerate(np.arange(0, 1001-frame, frame)):
            if frame == 1:
                tempdata = data[:, data_idx]  # Grab the next view
                tempair = airdata[:, data_idx]
            else:
                tempdata = np.sum(data[:, data_idx:data_idx + frame], axis=1)  # Grab the sum of the next 'frames' views
                tempair = np.sum(airdata[:, data_idx:data_idx + frame], axis=1)

            corr_data = self.intensity_correction(tempdata, tempair)  # Correct for air

            # Go through each bin and calculate CNR
            for j, img in enumerate(corr_data):
                background = np.nanmean(img*bg_mask)
                roi = np.nanmean(img*mask)
                contrast[j, i] = abs(background - roi)

        # Average over the frames
        contrast_err = np.std(contrast, axis=1)
        contrast = np.mean(contrast, axis=1)

        return contrast, contrast_err

    def avg_contrast_over_all_frames(self, data, airdata, mask, bg_mask):
        """
        This function will take the data and airscan and calculate the contrast and contrast error over all the frames
        in the list
        :param data: 4D ndarray, <counters, views, rows, columns>
                    The phantom data
        :param airdata: 4D ndarray, <counters, views, rows, columns>
                    The airscan
        :param mask: 2D ndarray
                    The mask of the contrast area
        :param bg_mask: 2D ndarray
                    The mask of the background
        :return:
        """
        # The contrast and contrast error over frames in the list
        contrast_frames = np.zeros([len(self.frames), len(data), 2])

        for i in np.arange(len(self.frames)):
            # Calculate the contrast
            c, ce = self.avg_contrast_over_frames(data, airdata, mask, bg_mask, self.frames[i])
            contrast_frames[i, :, 0] = c   # The ith frames, set first column equal to contrast
            contrast_frames[i, :, 1] = ce  # The ith frames, set second column equal to contrast error

        return contrast_frames

    def cnr_noise_vs_pixels(self):
        """ This function just transposes the cnr and noise data so that the values vs. pixels are the last two
            axes in the array"""
        cnr_pixel = np.transpose(self.cnr_time, axes=(0, 4, 2, 3, 1))
        noise_pixel = np.transpose(self.noise_time, axes=(0, 4, 2, 3, 1))
        return cnr_pixel, noise_pixel

    def mean_counts(self):
        """This function gets the mean counts within the airscan for all times in frames for all bins"""
        counts = np.zeros([self.num_bins, len(self.frames)])
        for j, frame in enumerate(self.frames):
            temp_cts = np.zeros([self.num_bins, int(1000 / frame)])  # Collect data for all bins
            # Get every frame number of frames and sum, over the entire view range
            for idx, data_idx in enumerate(np.arange(0, 1001 - frame, frame)):
                if frame == 1:
                    tempair = self.air_data.data_a0[:, data_idx]
                else:
                    tempair = np.sum(self.air_data.data_a0[:, data_idx:data_idx + frame], axis=1)

                for i in np.arange(self.num_bins):
                    temp_cts[i, idx] = np.nanmean(tempair[i]*self.bg[0])

            # Add temp_cts to the counts results, averaging over all different frames
            counts[:, j] = np.mean(temp_cts, axis=1)

        return counts

    def non_uniformity(self, pixel):
        """
        This function finds the relative difference between 2 pixels over time
        :param pixel: tuple
                    The pixel coordinates
        :return:
        """
        # The results of the test
        nu_res = np.zeros([self.num_bins, len(self.frames)])
        # Go through each frame in frames
        for j, frame in enumerate(self.frames):
            temp_nu = np.zeros([self.num_bins, int(1000/frame)])  # Collect data for all bins
            for idx, data_idx in enumerate(np.arange(0, 1001 - frame, frame)):
                if frame == 1:
                    tempair = self.air_data.data_a0[:, data_idx]
                else:
                    tempair = np.sum(self.air_data.data_a0[:, data_idx:data_idx + frame], axis=1)

                for i in np.arange(self.num_bins):
                    px_val = tempair[(i, *pixel)]
                    diff = np.abs(np.subtract(tempair[i], px_val))
                    temp_nu[i, idx] = np.nanmean(self.bg[0]*diff)/np.nanmean(self.bg[0]*tempair[i])

            # Add temp_nu to the non-uniformity results data
            nu_res[:, j] = np.mean(temp_nu, axis=1)

        return nu_res

    @staticmethod
    def intensity_correction(data, air_data):
        """
        This function corrects flatfield data to show images, -ln(I/I0), I is the intensity of the data, I0 is the
        intensity in an airscan
        :param data: The data to correct (must be the same shape as air_data)
        :param air_data: The airscan data (must be the same shape as data)
        :return: The corrected data array
        """
        return np.log(np.divide(air_data, data))

    @staticmethod
    def add_adj_bins(data, bins):
        """
        This function takes the adjacent bins given in bins and sums along the bin axis, can sum multiple bins
        :param data: 4D numpy array
                    Data array with shape <counters, views, rows, columns
        :param bins: 1D array
                    Bin numbers (as python indices, i.e the 1st bin would be 0) to sum
                    Form: [Starting bin, Ending bin]
                    Ex. for the 2nd through 5th bins, bins = [1, 4]
        :return: The summed data with the summed bins added together and the rest of the data intact
                    shape <counters, views, rows, columns>
        """
        data_shape = np.array(np.shape(data))
        data_shape[0] = data_shape[0] - (
                    bins[1] - bins[0])  # The new data will have the number of added bins - 1 new counters
        new_data = np.zeros(data_shape)

        new_data[0:bins[0]] = data[0:bins[0]]
        new_data[bins[0]] = np.sum(data[bins[0]:bins[-1] + 1], axis=0)
        new_data[bins[0] + 1:] = data[bins[1] + 1:]

        return new_data

