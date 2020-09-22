import numpy as np
import matplotlib.pyplot as plt
import os
from redlen.redlen_analysis import RedlenAnalyze
import mask_functions as grm
import _pickle as pickle


class VisibilityError(Exception):
    pass


class AnalyzeUniformity(RedlenAnalyze):

    def __init__(self, folder, air_folder, test_num=1, mm='M20358_Q20', load_dir=r'X:\TEST LOG\MINI MODULE\Canon',
                 save_dir=r'C:\Users\10376\Documents\Phantom Data'):

        super().__init__(folder, test_num, mm, 'UNIFORMITY', load_dir, save_dir)
        self.thresholds = []
        self.pxp = np.array([1, 2, 3, 4, 6])
        self.frames = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 50, 100, 250, 500, 1000])
        self.air_data = RedlenAnalyze(air_folder, test_num, mm, 'UNIFORMITY', load_dir, save_dir)

        self.masks = []
        self.bg = []
        self.small_phantom = False
        self.visible_bin = 12

        if 'Masks.npz' in os.listdir(self.save_dir):
            self.masks = np.load(os.path.join(self.save_dir, 'Masks.npz'), allow_pickle=True)['mask']
            self.bg = np.load(os.path.join(self.save_dir, 'Masks.npz'), allow_pickle=True)['bg']
        else:
            self.get_masks()

        self.cnr_time, self.noise_time = [], []
        self.counts = []
        self.rel_uniformity = []
        self.air_noise = []
        self.signal = []
        self.bg_signal = []
        self.contrast = []

    def get_masks(self):
        """This function gets the contrast mask and background mask for all pixel aggregations"""

        self.visible_bin = self.test_visibility()
        data = np.load(self.data_a0)
        airdata = np.load(self.air_data.data_a0)
        self.masks = []
        self.bg = []

        val = input('Is the phantom small? (y/n)')
        if val is 'y':
            self.small_phantom = True
        else:
            self.small_phantom = False

        for i, pixel in enumerate(self.pxp):
            if pixel == 1:
                tempdata = data
                tempair = airdata
            else:
                tempdata = self.sumpxp(data, pixel)
                tempair = self.sumpxp(airdata, pixel)

            image = self.intensity_correction(tempdata, tempair)
            image = np.sum(image, axis=1)
            tempmask, tempbg = self.choose_mask_types(image[self.visible_bin], pixel)

            self.masks.append(tempmask)
            self.bg.append(tempbg)

        np.savez(os.path.join(self.save_dir, 'Masks.npz'), mask=self.masks, bg=self.bg)

    def redo_masks(self, pixels=[1, 2, 3, 4, 6, 8, 12]):
        masks = np.load(os.path.join(self.save_dir, 'Masks.npz'), allow_pickle=True)
        self.visible_bin = self.test_visibility()

        val = input('Is the phantom small? (y/n)')
        if val is 'y':
            self.small_phantom = True
        else:
            self.small_phantom = False

        bg = masks['bg']
        masks = masks['mask']

        data = np.load(self.data_a0)
        airdata = np.load(self.air_data.data_a0)

        for pixel in pixels:
            idx = np.squeeze(np.argwhere(self.pxp == pixel))

            if pixel == 1:
                tempdata = data
                tempair = airdata
            else:
                tempdata = self.sumpxp(data, pixel)
                tempair = self.sumpxp(airdata, pixel)

            image = self.intensity_correction(tempdata, tempair)
            image = np.sum(image, axis=1)
            tempmask, tempbg = self.choose_mask_types(image[self.visible_bin], pixel)

            masks[idx] = tempmask
            bg[idx] = tempbg

        self.masks = masks
        self.bg = bg

        np.savez(os.path.join(self.save_dir, 'Masks.npz'), mask=self.masks, bg=self.bg)

    def test_visibility(self):
        """
        This function will test if your contrast area is visible in the one of the energy bins
        :return: The bin to use
        """
        data = np.load(self.data_a0)
        airdata = np.load(self.air_data.data_a0)

        image = np.squeeze(self.intensity_correction(data, airdata))
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

    def analyze_cnr_noise(self, pixels=None, redo=False):
        """
        This function calculates the CNR and noise in each bin at each of the pixel aggregations values and at a number
        of different acquisition times
        :return:
        """

        cnr_file = os.path.join(self.save_dir, f'TestNum{self.test_num}_cnr_time.npy')
        noise_file = os.path.join(self.save_dir, f'TestNum{self.test_num}_noise_time.npy')

        if os.path.exists(cnr_file) and os.path.exists(noise_file) and not redo:
            print('Loaded.')
            self.cnr_time = np.load(cnr_file)
            self.noise_time = np.load(noise_file)
        else:
            data = np.load(self.data_a0)
            airdata = np.load(self.air_data.data_a0)

            if pixels:
                print('Modifying...')
                self.cnr_time = np.load(cnr_file)
                self.noise_time = np.load(noise_file)
                pixels = np.array(pixels)
                p_idx = np.zeros(len(pixels), dtype=int)
                for j in np.arange(len(pixels)):
                    p_idx[j] = int(np.squeeze(np.argwhere(self.pxp == pixels[j])))

            else:
                print('Creating...')
                self.cnr_time = np.zeros([len(self.pxp), self.num_bins, 2, len(self.frames)])
                self.noise_time = np.zeros([len(self.pxp), self.num_bins, 2, len(self.frames)])
                pixels = self.pxp
                p_idx = np.arange(len(pixels))

            cnr_vals = np.zeros([len(pixels), len(self.frames), self.num_bins, 2])
            noise = np.zeros([len(pixels), len(self.frames), self.num_bins, 2])

            # Aggregate the number of pixels in pxp squared
            for p, pix in enumerate(pixels):
                if pix == 1:
                    data_pxp = np.squeeze(data)
                    air_pxp = np.squeeze(airdata)
                else:
                    data_pxp = np.squeeze(self.sumpxp(data, pix))  # Aggregate the pixels
                    air_pxp = np.squeeze(self.sumpxp(airdata, pix))

                # Collect the frames aggregated over, the noise and cnr and save
                cnr_vals[p], noise[p] = self.avg_cnr_noise_over_all_frames(data_pxp, air_pxp, self.masks[p_idx[p]],
                                                                           self.bg[p_idx[p]])

            # CNR vs. time, noise vs. time <pixels, time, bin, value or error (0 or 1)>
            # Reorganize to <pixels, bin, value or error, time>
            cnr_vals = np.transpose(cnr_vals, axes=(0, 2, 3, 1))
            noise = np.transpose(noise, axes=(0, 2, 3, 1))

            self.cnr_time[p_idx[0]:p_idx[-1]+1] = cnr_vals
            self.noise_time[p_idx[0]:p_idx[-1]+1] = noise
            np.save(cnr_file, self.cnr_time)
            np.save(noise_file, self.noise_time)

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
            tempair = np.divide(np.sum(airdata, axis=1), 1000/frame)
            if frame == 1:
                tempdata = data[:, data_idx]  # Grab the next view
                # tempair = airdata[:, data_idx]
            else:
                # Grab the sum of the next 'frames' views
                tempdata = np.sum(data[:, data_idx:data_idx + frame], axis=1)
                # tempair = np.sum(airdata[:, data_idx:data_idx + frame], axis=1)

            corr_data = self.intensity_correction(tempdata, tempair)  # Correct for air

            # Go through each bin and calculate CNR
            for j, img in enumerate(corr_data):
                cnr_val[j, i], cnr_err[j, i] = self.cnr(img, mask, bg_mask)
                noise[j, i] = np.nanstd(img*bg_mask)  # Get noise as fraction of mean background

        if 'BB4mm' in self.save_dir:
            if np.shape(data)[2] == 4:
                if frame == 25:
                    print('GOTIM')
                    cnr_val[8, 26] = np.nan

        # Average over the frames
        cnr_err = np.nanstd(cnr_val, axis=1)
        cnr_val = np.nanmean(cnr_val, axis=1)
        # cnr_err = np.nanmean(cnr_err, axis=1)
        noise_std = np.nanstd(noise, axis=1)
        noise = np.nanmean(noise, axis=1)

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

    def mean_signal_all_pixels(self, pixels=None, redo=False):
        signal_file = os.path.join(self.save_dir, f'TestNum{self.test_num}_signal.npy')

        if os.path.exists(signal_file) and not redo:
            print('Loaded.')
            self.signal = np.load(signal_file)
        else:
            data = np.load(self.data_a0)
            airdata = np.load(self.air_data.data_a0)

            if pixels:
                print('Modifying...')
                self.signal = np.load(signal_file)

                pixels = np.array(pixels)
                p_idx = np.zeros(len(pixels), dtype=int)
                for j in np.arange(len(pixels)):
                    p_idx[j] = int(np.squeeze(np.argwhere(self.pxp == pixels[j])))

            else:
                print('Creating...')
                self.signal = np.zeros([len(self.pxp), self.num_bins, 2, len(self.frames)])
                self.signal = np.zeros([len(self.pxp), self.num_bins, 2, len(self.frames)])
                pixels = self.pxp
                p_idx = np.arange(len(pixels))

            signal_vals = np.zeros([len(pixels), len(self.frames), self.num_bins, 2])

            # Aggregate the number of pixels in pxp squared
            for p, pix in enumerate(pixels):
                if pix == 1:
                    data_pxp = np.squeeze(data)
                    air_pxp = np.squeeze(airdata)
                else:
                    data_pxp = np.squeeze(self.sumpxp(data, pix))  # Aggregate the pixels
                    air_pxp = np.squeeze(self.sumpxp(airdata, pix))

                for f_idx, frame in enumerate(self.frames):
                    bg_signal = np.zeros([self.num_bins, int(1000 / frame)])
                    # Go over the data views in jumps of the number of frames
                    for i, data_idx in enumerate(np.arange(0, 1001 - frame, frame)):
                        tempair = np.divide(np.sum(air_pxp, axis=1), 1000/frame)
                        if frame == 1:
                            tempdata = data_pxp[:, data_idx]  # Grab the next view
                            # tempair = air_pxp[:, data_idx]
                        else:
                            # Grab the sum of the next 'frames' views
                            tempdata = np.sum(data_pxp[:, data_idx:data_idx + frame], axis=1)
                            # tempair = np.sum(air_pxp[:, data_idx:data_idx + frame], axis=1)

                        corr_data = self.intensity_correction(tempdata, tempair)  # Correct for air

                        # Go through each bin and calculate signal
                        for j, img in enumerate(corr_data):
                            bg_signal[j, i] = np.nanmean(img * self.bg[p_idx[p]])

                    # Average over the frames
                    signal_vals[p, f_idx, :, 0] = np.mean(bg_signal, axis=1)
                    signal_vals[p, f_idx, :, 1] = np.std(bg_signal, axis=1)

            # CNR vs. time, noise vs. time <pixels, time, bin, value or error (0 or 1)>
            # Reorganize to <pixels, bin, value or error, time>
            signal_vals = np.transpose(signal_vals, axes=(0, 2, 3, 1))

            self.signal[p_idx[0]:p_idx[-1] + 1] = signal_vals
            np.save(signal_file, self.signal)

    def avg_signal_over_frames(self, frame):
        """
        This function will take the data and airscan and calculate the signal of the contrast and the background for
        every number of frames and then avg
        :param frame: int
                    The number of frames to avg together
        :return: four lists with the signal, signal error, background signal, and bg signal error in each of the bins
                    (usually 13 elements)
        """
        data = np.load(self.data_a0)
        airdata = np.load(self.air_data.data_a0)

        # Array to hold the cnr for the number of times it will be calculated
        contrast_signal = np.zeros([self.num_bins, int(1000/frame)])
        bg_signal = np.zeros([self.num_bins, int(1000 / frame)])

        # Go over the data views in jumps of the number of frames
        for i, data_idx in enumerate(np.arange(0, 1001-frame, frame)):
            tempair = np.divide(np.sum(airdata, axis=1), 1000/frame)
            if frame == 1:
                tempdata = data[:, data_idx]  # Grab the next view
                # tempair = airdata[:, data_idx]
            else:
                # Grab the sum of the next 'frames' views
                tempdata = np.sum(data[:, data_idx:data_idx + frame], axis=1)
                # tempair = np.sum(airdata[:, data_idx:data_idx + frame], axis=1)

            corr_data = self.intensity_correction(tempdata, tempair)  # Correct for air

            # Go through each bin and calculate CNR
            for j, img in enumerate(corr_data):
                bg_signal[j, i] = np.nanmean(img*self.bg[0])
                contrast_signal[j, i] = np.nanmean(img*self.masks[0])

        # Average over the frames
        contrast_signal_err = np.std(contrast_signal, axis=1)
        contrast_signal = np.mean(contrast_signal, axis=1)

        bg_signal_err = np.std(bg_signal, axis=1)
        bg_signal = np.mean(bg_signal, axis=1)

        return contrast_signal, contrast_signal_err, bg_signal, bg_signal_err

    def avg_contrast_over_all_frames(self):
        """
        This function will take the data and airscan and calculate the contrast and contrast error over all the frames
        in the list, and the signal and background signal
        :return:
        """
        # The contrast and contrast error over frames in the list
        contrast_frames = np.zeros([self.num_bins, 2, len(self.frames)])
        signal_frames = np.zeros([self.num_bins, 2, len(self.frames)])
        bg_frames = np.zeros([self.num_bins, 2, len(self.frames)])

        for i in np.arange(len(self.frames)):
            # Calculate the contrast
            s, se, b, be = self.avg_signal_over_frames(self.frames[i])
            signal_frames[:, 0, i] = s
            signal_frames[:, 1, i] = se
            bg_frames[:, 0, i] = b
            bg_frames[:, 1, i] = be

        contrast_frames[:, 0] = np.abs(signal_frames[:, 0] - bg_frames[:, 0])
        contrast_frames[:, 1] = np.sqrt(signal_frames[:, 1]**2 + bg_frames[:, 1]**2)  # Error propagation

        self.contrast = contrast_frames
        self.bg_signal = bg_frames
        self.signal = signal_frames

    def cnr_noise_vs_pixels(self):
        """ This function just transposes the cnr and noise data so that the values vs. pixels are the last two
            axes in the array"""
        cnr_pixel = np.transpose(self.cnr_time, axes=(0, 4, 2, 3, 1))
        noise_pixel = np.transpose(self.noise_time, axes=(0, 4, 2, 3, 1))
        return cnr_pixel, noise_pixel

    def mean_counts(self, choice='data'):
        """This function gets the mean counts within the airscan for all times in frames for all bins"""

        if choice == 'data':
            data = np.load(self.data_a0)
        else:
            data = np.load(self.air_data.data_a0)
        counts = np.zeros([self.num_bins, len(self.frames)])
        for j, frame in enumerate(self.frames):
            temp_cts = np.zeros([self.num_bins, int(1000 / frame)])  # Collect data for all bins
            # Get every frame number of frames and sum, over the entire view range
            for idx, data_idx in enumerate(np.arange(0, 1001 - frame, frame)):
                if frame == 1:
                    tempair = data[:, data_idx]
                else:
                    tempair = np.sum(data[:, data_idx:data_idx + frame], axis=1)

                for i in np.arange(self.num_bins):
                    temp_cts[i, idx] = np.nanmean(tempair[i]*self.bg[0])

            # Add temp_cts to the counts results, averaging over all different frames
            counts[:, j] = np.mean(temp_cts, axis=1)
            self.counts = counts

        return counts

    def get_air_noise(self):
        """This function will get the noise in the airscan image at all bins and all times in self.frames"""
        airdata = np.load(self.air_data.data_a0)
        # The results of the test
        air_noise = np.zeros([self.num_bins, len(self.frames)])
        # Go through each frame in frames
        for j, frame in enumerate(self.frames):
            temp_an = np.zeros([self.num_bins, int(1000 / frame)])  # Collect data for all bins
            for idx, data_idx in enumerate(np.arange(0, 1001 - frame, frame)):
                if frame == 1:
                    tempair = airdata[:, data_idx]
                else:
                    tempair = np.sum(airdata[:, data_idx:data_idx + frame], axis=1)

                for i in np.arange(self.num_bins):
                    bg_img = self.bg[0]*tempair[i]
                    temp_an[i, idx] = np.nanstd(bg_img) / np.nanmean(bg_img)

            # Add temp_nu to the non-uniformity results data
            air_noise[:, j] = np.mean(temp_an, axis=1)
            self.air_noise = air_noise

        return air_noise

    def non_uniformity(self, pixel):
        """
        This function finds the relative difference between 2 pixels over time
        :param pixel: tuple
                    The pixel coordinates
        :return:
        """
        airdata = np.load(self.air_data.data_a0)
        # The results of the test
        nu_res = np.zeros([self.num_bins, len(self.frames)])
        # Go through each frame in frames
        for j, frame in enumerate(self.frames):
            temp_nu = np.zeros([self.num_bins, int(1000/frame)])  # Collect data for all bins
            for idx, data_idx in enumerate(np.arange(0, 1001 - frame, frame)):
                if frame == 1:
                    tempair = airdata[:, data_idx]
                else:
                    tempair = np.sum(airdata[:, data_idx:data_idx + frame], axis=1)

                for i in np.arange(self.num_bins):
                    px_val = tempair[(i, *pixel)]
                    diff = np.abs(np.subtract(tempair[i], px_val))
                    temp_nu[i, idx] = np.nanmean(self.bg[0]*diff)/np.nanmean(self.bg[0]*tempair[i])

            # Add temp_nu to the non-uniformity results data
            nu_res[:, j] = np.mean(temp_nu, axis=1)
            self.rel_uniformity = nu_res

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

