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

    files = glob.glob(folder + '/*' + tag + '*.mat')
    a0 = mat_to_npy(files[0])
    a1 = mat_to_npy(files[1])

    data_shape = np.array(np.shape(a0))  # Get the shape of the data files

    idx = len(data_shape)
    col = data_shape[idx-1]
    data_shape[idx-1] = col*2  # Update the column number to be two times as many

    both_mods = np.zeros(data_shape)  # Empty array to hold the data
    both_mods[:, :, :, 0:col] = a0
    both_mods[:, :, :, col:] = a1

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
    subfolders = glob.glob(folder + '/Raw Test Data/*/')
    num_MMs = len(subfolders)

    mm0 = stitch_A0A1(subfolders[0] + subpath)
    data_shape = np.array(np.shape(mm0))  # Shape of the combined A0 A1 data matrix
    idx = len(data_shape)
    col = data_shape[idx-1]  # Number of columns (Should be 72)

    new_shape = data_shape
    new_shape[idx-1] = col * num_MMs  # Update the number of columns for the number of MMs
    final_module = np.zeros(new_shape)
    final_module[:, :, :, 0:col] = mm0  # Put the first MM into the new data file

    for i, sub in enumerate(subfolders[1:]):
        file_path = sub + subpath
        curr_mm = stitch_A0A1(file_path)
        final_module[:, :, :, (i+1)*col:(i+2)*col] = curr_mm

    return final_module

#%%
path = r'X:\Devon_UVic\LDA Data\DM-general-04-28-20'
subfolders = glob.glob(path + '/Raw Test Data/*/')
#path = r'X:\TEST LOG\MINI MODULE\Canon\M20358_Q20\Test 84\Raw Test Data\M20358_Q20\UNIFORMITY'
x = stitch_MMs(path)
x0 = stitch_A0A1(subfolders[0] + '/UNIFORMITY/')
x1 = stitch_A0A1(subfolders[1] + '/UNIFORMITY/')
x2 = stitch_A0A1(subfolders[2] + '/UNIFORMITY/')
x3 = stitch_A0A1(subfolders[3] + '/UNIFORMITY/')


#%%
fig1 = plt.figure(figsize=(6, 3))
plt.imshow(x0[12, 3], vmin=0, vmax=1E2)
fig2 = plt.figure(figsize=(6, 3))
plt.imshow(x1[12, 3], vmin=0, vmax=1E2)
fig3 = plt.figure(figsize=(6, 3))
plt.imshow(x2[12, 3], vmin=0, vmax=1E2)
fig4 = plt.figure(figsize=(6, 3))
plt.imshow(x3[12, 3], vmin=0, vmax=1E2)
fig5 = plt.figure(figsize=(24, 3))
plt.imshow(x[12, 3], vmin=0, vmax=1E2)
plt.show()

#x = stitch_modules(2, 'D:\Research\sCT Scan Data/Cu_0.5_10-17-19/Rot_9.88/Raw Test Data/M15691/UNIFORMITY', 'Run003')