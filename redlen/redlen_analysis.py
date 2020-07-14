from scipy.io import loadmat
import numpy as np
import os
from glob import glob
from analysis import Analyze


class SizeError(Exception):
    pass


class RedlenAnalyze(Analyze):

    def __init__(self, folder, test_num, mm, acquire_type, load_dir, save_dir):

        self.test_num = test_num
        substring = os.path.join('Raw Test Data', mm, acquire_type)
        self.load_dir = os.path.join(load_dir, mm, folder, substring)
        del substring

        self.save_dir = os.path.join(save_dir, acquire_type, folder)
        print(self.save_dir)
        os.makedirs(self.save_dir, exist_ok=True)
        self.filename = f'TestNum{test_num}.pk1'

        self.data_a0 = os.path.join(self.save_dir, f'TestNum{test_num}_DataA0.npy')
        self.data_a1 = os.path.join(self.save_dir, f'TestNum{test_num}_DataA1.npy')

        if not os.path.exists(self.data_a0):
            mat_a0_files = glob(os.path.join(self.load_dir, '*A0*'))
            mat_a1_files = glob(os.path.join(self.load_dir, '*A1*'))

            a0 = np.squeeze(self.mat_to_npy(mat_a0_files[test_num-1]))
            a1 = np.squeeze(self.mat_to_npy(mat_a1_files[test_num-1]))

            np.save(self.data_a0, a0)
            np.save(self.data_a1, a1)

        self.data_shape = np.shape(np.load(self.data_a0))
        self.num_bins = self.data_shape[0]

    @staticmethod
    def mat_to_npy(mat_path):
        """
        This function takes a .mat file from the detector and grabs the count data
        :param mat_path: The path to the .mat file
        :return: the data array as a numpy array
        """
        scan_data = loadmat(mat_path)
        just_data = np.array(scan_data['cc_struct']['data'][0][0][0][0][0])  # Get only the data, no headers, etc.

        return just_data

    def stitch_a0a1(self):
        """
        This function will take the counts from the two modules and assembles one numpy array
        [capture, bin, view, row, column]
        :param a0: The A0 data array
        :param a1: The A1 data array
        :return: The combined array
        """
        data = np.load(self.data_a0)
        data_shape = np.array(np.shape(data))  # Get the shape of the data files
        new_shape = list(data_shape)
        new_shape[-1] = 2*data_shape[-1]

        a0 = self.data_a0[..., :, :, ::-1]  # Flip horizontally
        a1 = self.data_a1[..., :, ::-1, :]  # Flip vertically
        both_mods = np.zeros(new_shape)

        both_mods[..., :, :, 0:36] = a0
        both_mods[..., :, :, 36:] = a1

        return both_mods

    @staticmethod
    def sumpxp(data, num_pixels):
        """
        This function takes a data array and sums nxn pixels along the row and column data
        :param data: 5D ndarray
                    The full data array <captures, counters, views, rows, columns>
        :return: The new data array with nxn pixels from the inital data summed together
        """
        dat_shape = np.array(np.shape(data))
        dat_shape[-2] = int(dat_shape[-2] / num_pixels)  # Reduce size by num_pixels in the row and column directions
        dat_shape[-1] = int(dat_shape[-1] / num_pixels)

        ndata = np.zeros(dat_shape)
        n = num_pixels
        for row in np.arange(dat_shape[-2]):
            for col in np.arange(dat_shape[-1]):
                # Get each 2x2 subarray over all of the first 2 axes
                if len(dat_shape) == 5:
                    temp = data[:, :, :, n * row:n * row + n, n * col:n * col + n]
                    ndata[:, :, :, row, col] = np.sum(temp, axis=(-2, -1))  # Sum over only the rows and columns
                elif len(dat_shape) == 4:
                    temp = data[:, :, n * row:n * row + n, n * col:n * col + n]
                    ndata[:, :, row, col] = np.sum(temp, axis=(-2, -1))  # Sum over only the rows and columns
                else:
                    return SizeError('Size of input array in sumpxp.py is not 4 or 5 dimensions.')
        return ndata

    # def stitch_MMs(self, test_type=3):
    #     """
    #     This function is meant to stitch together multiple MMs after the A0 and A1 modules have been combined
    #     :param folder: path for the test folder (i.e. Test03, or whatever you named it)
    #     :param subpath:
    #     :return:
    #     """
    #     tests = {1: '/DYNAMIC/',
    #              2: '/SPECTRUM/',
    #              3: '/UNIFORMITY/'}
    #     subpath = tests[test_type]
    #     subfolders = glob.glob(folder + '/Raw Test Data/*/')
    #
    #     mm0 = stitch_a0a1(subfolders[0] + subpath)
    #     data_shape = np.array(np.shape(mm0))  # Shape of the combined A0 A1 data matrix
    #     ax = len(data_shape) - 1  # Axis to concatenate along
    #
    #     for i, sub in enumerate(subfolders[1:]):
    #         file_path = sub + subpath
    #         curr_mm = stitch_a0a1(file_path)
    #         final_module = np.concatenate((mm0, curr_mm), axis=ax)
    #
    #     return final_module