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
    mat_path = mat_path.replace("\\", "/")
    scan_data = loadmat(mat_path)

    just_data = np.array(scan_data['cc_struct']['data'][0][0][0][0][0][0])  # Get only the data, no headers, etc.

    return just_data


def stitch_A0A1(folder, tag=''):
    """
    This function will take the counts from all modules and assemble one numpy array
    [Bin, view, row, column] (view is optional depending if the files have multiple views)
    :param num_modules: The number of MMs in the detector
    :param folder: The folder containing the data files
    :param tag: A str that is contained by only the files corresponding to the specific run and modules, default is an
                empty string since normally there are only two files
    :return:
    """
    folder = folder.replace("\\", "/")
    files = glob.glob(folder + '/*' + tag + '*.mat')

    data_shape = np.array(np.shape(mat_to_npy(files[0])))
    data_len = len(data_shape)
    data_shape[data_len-1] = data_shape[data_len-1]*2
    both_mods = np.zeros(data_shape)

    for i, file in enumerate(files):
        both_mods[:, :, :, i*data_shape[data_len-1]:(i+1)*data_shape[data_len-1]] = mat_to_npy(file)

    return both_mods


def stitch_MMs(folder, test_type=3):
    """

    :param folder: path for the test folder (i.e. Test03, or whatever you named it)
    :param subpath:
    :return:
    """
    tests = {1: '/DYNAMIC/',
             2: '/SPECTRUM/',
             3: '/UNIFORMITY/'}
    subpath = tests[test_type]
    subfolders = glob.glob(path + '/Raw Test Data/*/')

    num_MMs = len(subfolders)
    data_shape = np.array(np.shape(mat_to_npy(glob.glob(subfolders + subpath)[0])))
    new_shape = data_shape
    l_shape = len(data_shape)
    new_shape[l_shape-1] = new_shape[l_shape-1]*num_MMs
    final_module = np.zeros(new_shape)

    for i, sub in enumerate(subfolders):
        curr_mm = stitch_A0A1(sub)
        final_module[:, :, :, i*new_shape[l_shape-1]:(i+1)*new_shape[l_shape-1]] = curr_mm

    return final_module


path = "C:/DEV TESTS/S0682/Test 03/Raw Test Data/"
files = glob.glob(path + '/*/')


#x = stitch_modules(2, 'D:\Research\sCT Scan Data/Cu_0.5_10-17-19/Rot_9.88/Raw Test Data/M15691/UNIFORMITY', 'Run003')