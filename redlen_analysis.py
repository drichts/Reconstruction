from scipy.io import loadmat, whosmat
import numpy as np
import matplotlib.pyplot as plt
import glob
import general_OS_functions as gof


def mat_to_npy(mat_path):
    """
    This function takes a .mat file from the detector and grabs the count data
    :param mat_path: The path to the .mat file
    :return: the data array as a numpy array
    """

    scan_data = loadmat(mat_path)

    just_data = np.array(scan_data['cc_struct']['data'][0][0][0][0][0])  # Get only the data, no headers, etc.

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

    ax = len(data_shape)-1

    both_mods = np.concatenate((a0, a1), axis=ax)

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

    mm0 = stitch_A0A1(subfolders[0] + subpath)
    data_shape = np.array(np.shape(mm0))  # Shape of the combined A0 A1 data matrix
    ax = len(data_shape)-1  # Axis to concatenate along

    for i, sub in enumerate(subfolders[1:]):
        file_path = sub + subpath
        curr_mm = stitch_A0A1(file_path)
        final_module = np.concatenate((mm0, curr_mm), axis=ax)

    return final_module


def get_data_and_save_A0A1(path, save_name, tag='', folder='Raw Data/', save_directory='C:/Users/10376/Documents/IEEE Abstract/'):
    """
    This function grabs the A0 and A1 modules, stitches them and saves them to the folder in save_directory
    :param path:
    :param save_name:
    :param tag:
    :param folder:
    :param save_directory:
    :return:
    """
    data = stitch_A0A1(path, tag=tag)  # Grab the test data and stitch it
    gof.create_folder(folder, save_directory)  # Create the folder within your save directory to save the data
    np.save(save_directory + folder + '/' + save_name + '.npy', data)  # Save the data
    return


def get_data_and_save(path, save_name, file='A0', folder='Spectra/', save_directory='C:/Users/10376/Documents/IEEE Abstract/Raw Data/'):

    files = glob.glob(path + '/*' + file + '*')
    print(files[0])
    data = mat_to_npy(files[0])  # Grab the test data
    gof.create_folder(folder, save_directory)  # Create the folder within your save directory to save the data
    np.save(save_directory + folder + '/' + save_name + '.npy', data)  # Save the data
    print(path, save_name)
    print()
    return data


#%%  Doodle

folder = r'X:\TEST LOG\MINI MODULE\Canon\M20358_Q20\acswindow'
subfolders = glob.glob(folder + '/*')
subfolders = subfolders[2:]
save_names = ['Am241_1w', 'Am241_4w', 'Co57_1w', 'Co57_4w', '2mA_1w', '5mA_1w', '10mA_1w', '25mA_1w',
              '2mA_4w', '5mA_4w', '10mA_4w', '25mA_4w']

for i, sf in enumerate(subfolders):
    get_data_and_save(sf+'/Raw Test Data/M20358_Q20/SPECTRUM/', save_names[i])

#get_data_and_save(path, save_name)
