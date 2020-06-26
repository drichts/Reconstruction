import numpy as np
from scipy.io import loadmat, whosmat
from obsolete import general_OS_functions as gof
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import generateROImask as grm
import glob
import os


class Analyze:
    def __init__(self, folder, load_dir=r'D:\Research\sCT Scan Data', save_dir=r'D:\Research\Python Data\Spectral CT'):
        self.folder = folder
        self.load_dir = os.path.join(load_dir, folder)
        self.save_dir = os.path.join(save_dir, folder)
        os.makedirs(save_dir, exist_ok=True)


    def mat_to_npy(self, old=False, cc=False):
        """
        This function takes the .mat files generated in Matlab and extracts each slice of each bin and saves it as an
        .npy file in the Python data folder
        :param folder: The specific scan folder desired
        :param load_directory: The directory where the specific folder with the .mat files is
        :param save_directory: The directory where we want to save the .npy files
        :param old: This is just for reanalyzing the very first few scans I ever took, typically won't use it
        :param cc: True if you also want to collect the cc data
        :return: Nothing
        """

        path = load_directory + folder
        save_path = save_directory + folder

        # Create the folder if necessary
        gof.create_folder(folder_name=folder, directory_path=save_directory)

        # Create the RawSlices folder within the folder
        gof.create_folder(folder_name='RawSlices', directory_path=save_path)

        # Update the save_path
        save_path = save_path + '/RawSlices/'

        if old:
            # This is just for the first couple scans I ever did, probably won't need this
            s0 = loadmat(self.load_dir + '/binSEC0_multiplex_corrected2.mat')['Reconimg']  # bin 0
            s1 = loadmat(self.load_dir + '/binSEC1_multiplex_corrected2.mat')['Reconimg']  # bin 1
            s2 = loadmat(self.load_dir + '/binSEC2_multiplex_corrected2.mat')['Reconimg']  # bin 2
            s3 = loadmat(self.load_dir + '/binSEC3_multiplex_corrected2.mat')['Reconimg']  # bin 3
            s4 = loadmat(self.load_dir + '/binSEC4_multiplex_corrected2.mat')['Reconimg']  # bin 4
            s5 = loadmat(self.load_dir + '/binSEC5_multiplex_corrected2.mat')['Reconimg']  # bin 5
            s6 = loadmat(self.load_dir + '/binSEC6_multiplex_corrected2.mat')['Reconimg']  # bin 6 (summed bin)

        else:
            s0 = loadmat(path + '/data/binSEC1_test_corrected2_revisit.mat')['Reconimg']  # bin 0
            s1 = loadmat(path + '/data/binSEC2_test_corrected2_revisit.mat')['Reconimg']  # bin 1
            s2 = loadmat(path + '/data/binSEC3_test_corrected2_revisit.mat')['Reconimg']  # bin 2
            s3 = loadmat(path + '/data/binSEC4_test_corrected2_revisit.mat')['Reconimg']  # bin 3
            s4 = loadmat(path + '/data/binSEC5_test_corrected2_revisit.mat')['Reconimg']  # bin 4
            s5 = loadmat(path + '/data/binSEC6_test_corrected2_revisit.mat')['Reconimg']  # bin 5
            s6 = loadmat(path + '/data/binSEC13_test_corrected2_revisit.mat')['Reconimg']  # bin 6 (summed bin)

        # Grab just the colormap matrices
        s0 = s0['Reconimg']
        s1 = s1['Reconimg']
        s2 = s2['Reconimg']
        s3 = s3['Reconimg']
        s4 = s4['Reconimg']
        s5 = s5['Reconimg']
        s6 = s6['Reconimg']

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
            s7 = loadmat(path + '/data/binSEC7_test_corrected2_revisit.mat')  # bin 7
            s8 = loadmat(path + '/data/binSEC8_test_corrected2_revisit.mat')  # bin 8
            s9 = loadmat(path + '/data/binSEC9_test_corrected2_revisit.mat')  # bin 9
            s10 = loadmat(path + '/data/binSEC10_test_corrected2_revisit.mat')  # bin 10
            s11 = loadmat(path + '/data/binSEC11_test_corrected2_revisit.mat')  # bin 11
            s12 = loadmat(path + '/data/binSEC12_test_corrected2_revisit.mat')  # bin 12

            # Grab just the colormap matrices
            s7 = s7['Reconimg']
            s8 = s8['Reconimg']
            s9 = s9['Reconimg']
            s10 = s10['Reconimg']
            s11 = s11['Reconimg']
            s12 = s12['Reconimg']

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

    def cnr(self, image, contrast_mask, background_mask):
        """
        This function calculates the CNR of an ROI given the image, the ROI mask, and the background mask
        It also gives the CNR error
        :param image: The image to be analyzed as a 2D numpy array
        :param contrast_mask: The mask of the contrast area as a 2D numpy array
        :param background_mask: The mask of the background as a 2D numpy array
        :return CNR, CNR_error: The CNR and error of the contrast area
        """
        # The mean signal within the contrast area
        mean_ROI = np.nanmean(image * contrast_mask)
        std_ROI = np.nanstd(image * contrast_mask)

        # Mean and std. dev. of the background
        bg = np.multiply(image, background_mask)
        mean_bg = np.nanmean(bg)
        std_bg = np.nanstd(bg)

        CNR = abs(mean_ROI - mean_bg) / std_bg
        CNR_err = np.sqrt(std_ROI ** 2 + std_bg ** 2) / std_bg

        return CNR, CNR_err
