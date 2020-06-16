import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import general_OS_functions as gof
import sCT_Analysis as sct
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from datetime import datetime as dt


def cnr_vs_pixels_multi_w(folder, num_tests, pxp=[1, 2, 3, 4, 6, 8, 10, 12], times=[1, 2, 10, 250],
                          directory='C:/Users/10376/Documents/Phantom Data/'):
    """
    This function takes the folder with all the 1w, 3w, etc folders and aggregates the data for CNR vs. the pixels that
    have been combined at certain time frames
    :param folder: string
                The folder with the w folders in it
    :param num_tests: int
                The number of tests run at each charge sharing width w
    :param pxp: list of ints
                The pixels that were combined in the tests
    :param times: list, ints
                The time frames to collect CNR vs pixel data
    :param directory: string
                The path to the folder
    :return:
    """
    path = directory + folder + '/'

    # Create the pixel Data folder strings
    pix_folders = []
    for p in pxp:
        pix_folders.append(str(p) + 'x' + str(p) + ' Data/')

    w_folders = glob(path + '*/')  # Get each of the different w folders to analyze

    # Go through each w folder and extract the CNR vs pixel data
    for w_idx, fold in enumerate(w_folders):

        gof.create_folder('CNR vs Pixel Data', fold)

        # Find the index locations of the time frames desired by times
        frames_name = glob(fold + pix_folders[0] + '/*1_Frames.npy')[0]
        frames = np.load(frames_name)
        time_locs = np.zeros(len(times))
        for i, t in enumerate(times):
            time_locs[i] = np.argwhere(frames == t)

        # Go through each test
        for test_num in np.arange(1, num_tests+1):
            cnr = np.zeros([len(times), 13, 2, len(pxp)])  # <times, bin, cnr and cnr error, pixels>
            # Go through pixel
            for p_idx, pixel in enumerate(pix_folders):
                cnr_path = fold + pixel + '*' + str(test_num) + '_Avg_CNR.npy'
                cnr_data = np.load(glob(cnr_path)[0])
                # Go through each time value wanted
                for idx, time_idx in enumerate(time_locs):
                    cnr[idx, :, :, p_idx] = cnr_data[int(time_idx)]  # Get the cnr, cnr error at the specified time

            # Save the data

            for ti, time_val in enumerate(times):
                np.save(fold + '/CNR vs Pixel Data/Test_Num_' + str(test_num) + '_Frame_' + str(time_val) + '.npy', cnr[ti])
        np.save(fold + '/CNR vs Pixel Data/Pixels_x-axis.npy', pxp)
