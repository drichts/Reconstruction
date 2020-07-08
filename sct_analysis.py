import numpy as np
from scipy.io import loadmat, whosmat
from obsolete import general_OS_functions as gof
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import mask_functions as grm
import glob
import os
from analysis import Analyze


class AnalyzePCCT(Analyze):
    def __init__(self, folder, load_dir=r'D:\Research\sCT Scan Data', save_dir=r'D:\Research\Python Data\Spectral CT'):
        self.folder = folder
        self.load_dir = os.path.join(load_dir, folder)
        self.save_dir = os.path.join(save_dir, folder)
        os.makedirs(save_dir, exist_ok=True)
        #self.mat_to_npy()

    def normal_analysis(self, slice_num='15', cc=False, reanalyze=False):
        """
        Function gets the ROIs for all the vials, normalizes the images, and calculates all k_edges
        :param slice_num: The slice number to look at for getting the masks
        :param cc: True if collecting cc data as well
        :param reanalyze: True if reanalyzing data using the same masks
        :return:
        """

        image = np.load(self.save_dir + '/RawSlices/Bin6_Slice' + slice_num + '.npy')
        continue_flag = True
        if reanalyze:
            continue_flag = False
            masks = np.load(self.save_dir + '/Vial_Masks.npy')
            phantom_mask = np.load(self.save_dir + '/Phantom_Mask.npy')

        while continue_flag:
            masks = grm.phantom_ROIs(image, radius=7)
            val = input('Were the ROIs acceptable? (y/n)')
            if val is 'y':
                continue_flag = False

        np.save(self.save_dir + '/Vial_Masks.npy', masks)

        self.normalize(0.2, cc=cc)

        continue_flag = True
        if reanalyze:
            continue_flag = False

        while continue_flag:
            phantom_mask = grm.entire_phantom(image)
            val = input('Were the ROIs acceptable? (y/n)')
            if val is 'y':
                continue_flag = False
        np.save(self.save_dir + '/Phantom_Mask.npy', phantom_mask)
        self.k_edge(4, 3)
        self.k_edge(3, 2)
        self.k_edge(2, 1)
        self.k_edge(1, 0)

        if cc:
            self.k_edge(11, 10)  # CC 4, 3
            self.k_edge(10, 9)  # CC 3, 2
            self.k_edge(9, 8)  # CC 2, 1
            self.k_edge(8, 7)  # CC 1, 0


    def mat_to_npy(self, old=False, cc=False):
        """
        This function takes the .mat files generated in Matlab and extracts each slice of each bin and saves it as an
        .npy file in the Python data folder
        :param old: This is just for reanalyzing the very first few scans I ever took, typically won't use it
        :param cc: True if you also want to collect the cc data
        :return: Nothing
        """
        path = self.load_dir
        save_path = os.path.join(self.save_dir, 'RawSlices')
        os.makedirs(save_path, exist_ok=True)

        if old:
            # This is just for the first couple scans I ever did, probably won't need this
            s0 = loadmat(path + '/binSEC0_multiplex_corrected2.mat')['Reconimg'] # bin 0
            s1 = loadmat(path + '/binSEC1_multiplex_corrected2.mat')['Reconimg']  # bin 1
            s2 = loadmat(path + '/binSEC2_multiplex_corrected2.mat')['Reconimg']  # bin 2
            s3 = loadmat(path + '/binSEC3_multiplex_corrected2.mat')['Reconimg']  # bin 3
            s4 = loadmat(path + '/binSEC4_multiplex_corrected2.mat')['Reconimg']  # bin 4
            s5 = loadmat(path + '/binSEC5_multiplex_corrected2.mat')['Reconimg']  # bin 5
            s6 = loadmat(path + '/binSEC6_multiplex_corrected2.mat')['Reconimg']  # bin 6 (summed bin)

        else:
            s0 = loadmat(path + '/data/binSEC1_test_corrected2_revisit.mat')['Reconimg']  # bin 0
            s1 = loadmat(path + '/data/binSEC2_test_corrected2_revisit.mat')['Reconimg']  # bin 1
            s2 = loadmat(path + '/data/binSEC3_test_corrected2_revisit.mat')['Reconimg']  # bin 2
            s3 = loadmat(path + '/data/binSEC4_test_corrected2_revisit.mat')['Reconimg']  # bin 3
            s4 = loadmat(path + '/data/binSEC5_test_corrected2_revisit.mat')['Reconimg']  # bin 4
            s5 = loadmat(path + '/data/binSEC6_test_corrected2_revisit.mat')['Reconimg']  # bin 5
            s6 = loadmat(path + '/data/binSEC13_test_corrected2_revisit.mat')['Reconimg']  # bin 6 (summed bin)

        # Save each slice separately
        for i in np.arange(24):
            bin0_slice = s0[:, :, i]
            bin1_slice = s1[:, :, i]
            bin2_slice = s2[:, :, i]
            bin3_slice = s3[:, :, i]
            bin4_slice = s4[:, :, i]
            bin5_slice = s5[:, :, i]
            bin6_slice = s6[:, :, i]

            np.save(save_path + '/Bin0_Slice' + str(i) + '.npy', bin0_slice)
            np.save(save_path + '/Bin1_Slice' + str(i) + '.npy', bin1_slice)
            np.save(save_path + '/Bin2_Slice' + str(i) + '.npy', bin2_slice)
            np.save(save_path + '/Bin3_Slice' + str(i) + '.npy', bin3_slice)
            np.save(save_path + '/Bin4_Slice' + str(i) + '.npy', bin4_slice)
            np.save(save_path + '/Bin5_Slice' + str(i) + '.npy', bin5_slice)
            np.save(save_path + '/Bin6_Slice' + str(i) + '.npy', bin6_slice)

        if cc:
            s7 = loadmat(path + '/data/binSEC7_test_corrected2_revisit.mat')['Reconimg']  # bin 7
            s8 = loadmat(path + '/data/binSEC8_test_corrected2_revisit.mat')['Reconimg']  # bin 8
            s9 = loadmat(path + '/data/binSEC9_test_corrected2_revisit.mat')['Reconimg']  # bin 9
            s10 = loadmat(path + '/data/binSEC10_test_corrected2_revisit.mat')['Reconimg']  # bin 10
            s11 = loadmat(path + '/data/binSEC11_test_corrected2_revisit.mat')['Reconimg']  # bin 11
            s12 = loadmat(path + '/data/binSEC12_test_corrected2_revisit.mat')['Reconimg']  # bin 12

            # Save each slice separately
            for i in np.arange(24):
                bin7_slice = s7[:, :, i]
                bin8_slice = s8[:, :, i]
                bin9_slice = s9[:, :, i]
                bin10_slice = s10[:, :, i]
                bin11_slice = s11[:, :, i]
                bin12_slice = s12[:, :, i]

                np.save(save_path + '/Bin7_Slice' + str(i) + '.npy', bin7_slice)
                np.save(save_path + '/Bin8_Slice' + str(i) + '.npy', bin8_slice)
                np.save(save_path + '/Bin9_Slice' + str(i) + '.npy', bin9_slice)
                np.save(save_path + '/Bin10_Slice' + str(i) + '.npy', bin10_slice)
                np.save(save_path + '/Bin11_Slice' + str(i) + '.npy', bin11_slice)
                np.save(save_path + '/Bin12_Slice' + str(i) + '.npy', bin12_slice)

        return

    def normalize(self, water=None, cc=False):
        """
        Normalizes the .npy matrices to HU based on the mean value in the water vial
        :param water: The water value to normalize to
        :param cc: True if wanting to collect cc data
        :return:
        """
        path = self.save_dir

        load_path = os.path.join(path, 'RawSlices')
        water_mask = np.load(load_path + '/Vial_Masks.npy')[0]
        if water:
            save_path = os.path.join(path, 'OneNormSlices')
        else:
            save_path = os.path.join(path, 'Slices')
        os.makedirs(save_path, exist_ok=True)

        if cc:
            num = 13
        else:
            num = 7

        for i in np.arange(num):
            for j in np.arange(24):
                # Load the specific slice
                file = 'Bin' + str(i) + '_Slice' + str(j) + '.npy'
                temp_img = np.load(load_path+file)
                if not water:
                    water = np.nanmean(temp_img*water_mask)
                temp = self.norm_individual(temp_img, water)  # Normalize the image to HU
                np.save(os.path.join(save_path, file), temp)  # Save the normalized matrices

    @staticmethod
    def norm_individual(image, water_value):
        """
        Normalize an individual slice
        :param image: The image to normalize
        :param water_value: The water value to normalize to 0 HU
        :return:
        """
        # Normalize to HU
        image = 1000*np.divide((np.subtract(image, water_value)), water_value)
        # Get rid of any nan values
        image[np.isnan(image)] = -1000

        return image

    ##
    def image_noise(self, method='water', BIN=7, SLICE=0):
        """
        Calculates the image noise of each slice in each bin in either the water ROI or the phantom ROI
        :param method: 'water' or 'phantom', whether you want the noise of the phantom or water
        :param BIN: optional, if you want a specific bins noise output
        :param SLICE: optional, if you want a specific slice in a bin output
        :return: noise array (7 x 24, bin x slice)
        """
        path = self.save_dir
        masks = np.load(path + '/Vial_Masks.npy')
        if method is 'water':
            bg = masks[0]  # Water ROI matrix
        else:
            bg = np.load(path + '/BackgroundMaskMatrix.npy')  # Background ROI matrix

        # Create a matrix to store the noise in each bin and each slice within that bin
        noise = np.empty([7, 24])
        # Calculate the noise in each slice
        for i in np.arange(7):
            for j in np.arange(24):
                # Load the specific slice
                temp = np.load(path + '/Slices/Bin' + str(i) + '_Slice' + str(j) + '.npy')
                noise[i, j] = np.nanstd(temp*bg)

        if BIN is not 7:
            if SLICE is 0:
                print('The noise for each slice in bin', BIN, 'is:\n', noise[BIN, :])
            else:
                print('The noise for each slice', SLICE, 'in bin', BIN, 'is:', noise[BIN, SLICE])

        np.save(path + '/Image_Noise_' + method, noise)
        return noise

    def k_edge(self, bin_high, bin_low):
        """
        This function will take all the slices of the two bins and subtract them from one another to get the K-edge
        images of each slice
        :param bin_high: integer of the higher bin (1-4)
        :param bin_low: integer of the lower bin (0-3)
        :return: Nothing, saves the files needed
        """
        path = self.save_dir

        path_k = path + '/K-Edge/'
        path_slices = path + '/RawSlices/'
        os.makedirs(path_slices, exist_ok=True)

        # Convert the bin numbers to strings
        bin_high = str(bin_high)
        bin_low = str(bin_low)

        for i in np.arange(24):
            slice_num = str(i)

            # Load each high and low bin slice
            image_high = np.load(path_slices + 'Bin' + bin_high + '_Slice' + slice_num + '.npy')
            image_low = np.load(path_slices + 'Bin' + bin_low + '_Slice' + slice_num + '.npy')

            # Create the K-edge image
            kedge_image = np.subtract(image_high, image_low)

            np.save(path_k + 'Bin' + bin_high + '-' + bin_low + '_Slice' + slice_num + '.npy', kedge_image)

    ##
    def get_ct_cnr(self, z, type_recon='water'):
        """
        Get the cnr for each of the vial ROIs in a specific slice
        :param z: Slice to look at
        :param type_recon:
        :return:
        """

        path = self.save_dir + '/Slices/'
        vials = np.load(self.save_dir + '/Vial_Masks.npy')
        back = np.load(self.save_dir + '/Phantom_Mask.npy')
        CNR = np.zeros(len(vials))
        CNR_err = np.zeros(len(vials))

        image = np.load(path + 'Bin6_Slice' + str(z) + '.npy')
        for i, vial in enumerate(vials):
            if type_recon is 'water':
                CNR[i], CNR_err[i] = self.cnr(image, vial, vials[0])
            else:
                CNR[i], CNR_err[i] = self.cnr(image, vial, back)

        return CNR, CNR_err

    ##
    def find_least_noise(self, low_slice, high_slice, directory='D:/Research/Python Data/Spectral CT/'):
        """
        Find the slice with the least noise in the summed bin
        :param folder: folder to examine
        :param low_slice:
        :param high_slice:
        :param directory: where the folder is located
        :return:
        """
        subfolder = '/Slices/'
        path = self.save_dir + subfolder
        vials = np.load(self.save_dir + '/Vial_Masks.npy')
        noise_vals = np.zeros(high_slice-low_slice+1)
        for i in np.arange(low_slice, high_slice+1):
            img = np.load(path + 'Bin6_Slice' + str(i) + '.npy')
            noise_vals[i-low_slice] = np.nanstd(vials[0]*img)

        idx = np.argmin(noise_vals)
        return idx+low_slice, noise_vals[idx]

    ##
    def open_ct_image(self, folder, b, z, show=True, directory='D:/Research/Python Data/Spectral CT/'):

        path = directory + folder + '/Slices/'

        img = np.load(path + 'Bin' + str(b) + '_Slice' + str(z) + '.npy')

        if show:
            plt.imshow(img, cmap='gray', vmin=-500, vmax=1000)

        return img

    ##
    def open_kedge_image(self, folder, b, z, show=True, colormap=3, directory='D:/Research/Python Data/Spectral CT/'):

        path = self.save_dir + '/K-Edge/'

        img = np.load(path + 'Bin' + b + '_Slice' + str(z) + '.npy')
        # Create the colormaps

        nbins = 100
        c1 = (1, 0, 1)
        c2 = (0, 1, 0)
        c3 = (1, 0.843, 0)
        c4 = (0, 0, 1)

        gray_val = 0
        gray_list = (gray_val, gray_val, gray_val)

        c1_rng = [gray_list, c1]
        cmap1 = colors.LinearSegmentedColormap.from_list('Purp', c1_rng, N=nbins)
        c2_rng = [gray_list, c2]
        cmap2 = colors.LinearSegmentedColormap.from_list('Gree', c2_rng, N=nbins)
        c3_rng = [gray_list, c3]
        cmap3 = colors.LinearSegmentedColormap.from_list('G78', c3_rng, N=nbins)
        c4_rng = [gray_list, c4]
        cmap4 = colors.LinearSegmentedColormap.from_list('Blu8', c4_rng, N=nbins)

        clr_maps = {1: cmap1, 2: cmap2, 3: cmap3, 4: cmap4}

        if show:
            plt.imshow(img, cmap=clr_maps[colormap], vmin=0, vmax=0.01)
            plt.show()

        return img

    ##
    def mean_ROI_value(self, image, vial):

        # Get the matrix with only the values in the ROI
        value = np.multiply(image, vial)

        # Calculate the mean value in the matrix
        mean_val = np.nanmean(value)

        return mean_val

    ##
    def find_norm_value(self, folder, good_slice, vial, edge, subtype=2, directory='D:/Research/Python Data/Spectral CT/'):
        """
        Find the mean value of the highest concentration and of water to normalize images to concentration
        :param folder:
        :param good_slice:
        :param vial:
        :param edge:
        :param subtype:
        :param directory:
        :return:
        """
        low_slice, high_slice = good_slice[0], good_slice[1]

        # Define subfolder with subtype
        subfolder = {1: '/Slices/',
                     2: '/K-Edge/'}

        # Define the specific K-edge
        bin_edge = {0: 'Bin1-0_',
                    1: 'Bin2-1_',
                    2: 'Bin3-2_',
                    3: 'Bin4-3_'}

        # Vial ROIs
        rois = np.load(directory + folder + '/Vial_Masks.npy')
        value_roi = rois[vial]  # Specific vial ROI
        water_roi = rois[0]

        # Go through all the good slices to find the mean value in each slice
        slice_values = np.empty(high_slice-low_slice)
        zero_values = np.empty(high_slice-low_slice)

        for z in np.arange(low_slice, high_slice):
            image = np.load(directory + folder + subfolder[subtype] + bin_edge[edge] + 'Slice' + str(z) + '.npy')
            slice_values[z-low_slice] = np.nanmean(image*value_roi)
            zero_values[z-low_slice] = np.nanmean(image*water_roi)

        # Get average value
        norm_value = np.mean(slice_values)
        water_value = np.mean(zero_values)

        return water_value, norm_value

    ##
    def linear_fit(self, zero_value, norm_value):
        """
        Find a linear fit between 0 and 5% concentration
        :param zero_value: the water value (0%) concentration
        :param norm_value: the 5% concentration value
        :return:
        """
        coeffs = np.polyfit([zero_value, norm_value], [0, 5], 1)

        return coeffs

    ##
    def norm_kedge(self, coeffs, edge):
        """
        Normalize the k-edge images and save in a new folder (Normed K-Edge)
        :param coeffs:
        :param edge: int
                    The K-edge image to look at, 0 = 1-0, 1 = 2-1, 2 = 3-2, 4 = 4-3
        :return:
        """
        # Define the specific K-edge
        bin_edge = {0: 'Bin1-0_',
                    1: 'Bin2-1_',
                    2: 'Bin3-2_',
                    3: 'Bin4-3_'}

        path = self.save_dir
        load_path = os.path.join(path, 'K-Edge')
        save_path = os.path.join(path, 'Normed K-Edge')
        os.makedirs(save_path, exist_ok=True)

        # The linear fit
        l_fit = np.poly1d(coeffs)

        # Normalize each slice and save it
        for z in np.arange(24):
            file = bin_edge[edge] + 'Slice' + str(z) + '.npy'

            # Load the image and normalize it to the norm_value
            image = np.load(load_path + file)

            # Norm between 0 and 1 and then multiply by norm value
            image = l_fit(image)
            # Save the new image in the new location
            np.save(save_path + file, image)
