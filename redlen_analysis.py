from scipy.io import loadmat, whosmat
import numpy as np
import matplotlib.pyplot as plt
import glob

def mat_to_npy(mat_path):
    """
    This function takes a .mat file from the detector and grabs the count data
    :param mat_path: The path to the .mat file
    :return: the data array as a numpy array
    """

    scan_data = loadmat(mat_path)

    just_data = np.array(scan_data['cc_struct']['data'][0][0][0][0][0][0])  # Get only the data, no headers, etc.

    return np.squeeze(just_data)


def stitch_modules(num_modules, folder, tag):
    """
    This function will take the counts from all modules and assemble one numpy array
    [Bin, row, column]
    :param num_modules: The number of MMs in the detector
    :param folder: The folder containing the data files
    :param tag: A str that is contained by only the files corresponding to the specific run and modules
    :return:
    """
    folder = folder.replace("\\", "/")
    files = glob.glob(folder + '/*' + tag + '*.mat')

    data_shape = np.shape(mat_to_npy(files[0]))
    new_shape = np.array([data_shape[0], data_shape[1], data_shape[2]*num_modules])
    all_modules = np.zeros(new_shape)
    print(files)
    if num_modules is not len(files):
        print('Error: number of files and number of modules not equal')

    for i, file in enumerate(files):
        all_modules[:, :, i*data_shape[2]:(i+1)*data_shape[2]] = mat_to_npy(file)

    return all_modules


#%%

#path = 'D:/Research/sCT Scan Data/Al_2.0_10-17-19_1P/Air/air_9.88/Raw Test Data/M15691/UNIFORMITY/airscan_thresholds_D_M15691-A0_Run001_2019_10_17__10_58_38.mat'

#x = mat_to_npy(path)

x = stitch_modules(2, 'D:\Research\sCT Scan Data/Cu_0.5_10-17-19/Rot_9.88/Raw Test Data/M15691/UNIFORMITY', 'Run003')
