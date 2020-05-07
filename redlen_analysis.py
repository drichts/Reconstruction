from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import glob
import general_OS_functions as gof


def get_uniformity_data(folder, load_directory, MM, air_path='none', save_directory='X:/Devon_UVic/'):
    """
    This performs all of the necessary work to get and save UNIFORMITY data in numpy arrays including correcting for air
    if necessary
    :param folder: Just the folder before 'Raw Test Data', the analyzed data will be saved within this folder name within
    the save directory
    :param load_directory: The directory leading to the folder
    :param MM: The MM name, e.g. M20358
    :param air_path: Full path to combined air data (A0A1) in .npy format
    :param save_directory: The directory to save the folder to
    """
    path = load_directory + '/' + folder + '/Raw Test Data/' + MM + '/UNIFORMITY/'
    files = glob.glob(path + '/*.mat')

    # Just in case there are the two extra files in the folder, skip over them
    if len(files) > 2:
        files = sorted(files, key=time_stamp)
        a0 = mat_to_npy(files[2])
        a1 = mat_to_npy(files[3])
    else:
        a0 = mat_to_npy(files[0])
        a1 = mat_to_npy(files[1])

    both_a = stitch_A0A1(a0, a1)

    gof.create_folder(folder, save_directory)
    save_path = save_directory + folder
    gof.create_folder('Raw Data', save_path)

    if air_path is not 'none':
        corrected_both = intensity_correction(both_a)
        gof.create_folder('Corrected Data', save_path)
        np.save(save_path + '/Corrected Data/full_view.npy', corrected_both)

    np.save(save_path + '/Raw Data/full_view.npy', both_a)
    np.save(save_path + '/Raw Data/a0.npy', a0)
    np.save(save_path + '/Raw Data/a1.npy', a1)


def get_spectrum_data(folder, load_directory, MM, which_data=1, save_directory='X:/Devon_UVic/'):
    """
    This performs all of the necessary work to get and save SPECTRUM data in numpy arrays and get the count data to plot
    spectra for cc and sec over energy range in AU
    :param folder: Just the folder before 'Raw Test Data', the analyzed data will be saved within this folder name within
    the save directory
    :param load_directory: The directory leading to the folder
    :param MM: The MM name, e.g. M20358
    :param which_data: which data to get the median count rate from (1: A0, 2: A1, 3: both A0 and A1)
    :param save_directory: The directory to save the folder to
    """
    modules = {1: 'A0',
               2: 'A1',
               3: 'both'}

    path = load_directory + folder + '/Raw Test Data/' + MM + '/UNIFORMITY/'
    files = glob.glob(path + '/*.mat')

    # Just in case there are the two extra files in the folder, skip over them
    if len(files) > 2:
        files = sorted(files, key=time_stamp)
        a0 = mat_to_npy(files[2])
        a1 = mat_to_npy(files[3])
    else:
        a0 = mat_to_npy(files[0])
        a1 = mat_to_npy(files[1])

    both_a = stitch_A0A1(a0, a1)

    if modules[which_data] is 'A0':
        cc_spect, sec_spect = get_spectrum(a0)
    elif modules[which_data] is 'A1':
        cc_spect, sec_spect = get_spectrum(a1)
    else:
        cc_spect, sec_spect = get_spectrum(both_a)

    gof.create_folder(folder, save_directory)
    save_path = save_directory + folder
    gof.create_folder('Raw Data', save_path)
    gof.create_folder('Spectra', save_path)

    np.save(save_path + '/Raw Data/full_view.npy', both_a)
    np.save(save_path + '/Raw Data/a0.npy', a0)
    np.save(save_path + '/Raw Data/a1.npy', a1)

    np.save(save_path + '/Spectra/sec.npy', sec_spect)
    np.save(save_path + '/Spectra/cc.npy', cc_spect)


def time_stamp(file):
    """
    This function is a key for the sorted function, it returns the time stamp of a .mat data file
    :param file: The file path
    :return: The timestamp HR_MN_SC (hour, minute, second)
    """
    return file[-12:-4]


def mat_to_npy(mat_path):
    """
    This function takes a .mat file from the detector and grabs the count data
    :param mat_path: The path to the .mat file
    :return: the data array as a numpy array
    """
    scan_data = loadmat(mat_path)
    just_data = np.array(scan_data['cc_struct']['data'][0][0][0][0][0])  # Get only the data, no headers, etc.

    return just_data


def stitch_A0A1(a0, a1):
    """
    This function will take the counts from the two modules and assembles one numpy array
    [time, bin, view, row, column]
    :param a0: The A0 data array
    :param a1: The A1 data array
    :return: The combined array
    """
    data_shape = np.array(np.shape(a0))  # Get the shape of the data files
    ax = len(data_shape)-1  # We are combining more column data, so get that axis
    both_mods = np.concatenate((a0, a1), axis=ax)

    return both_mods


def stitch_MMs(folder, test_type=3):
    """
    This function is meant to stitch together multiple MMs after the A0 and A1 modules have been combined
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


def get_spectrum(data):
    """
    This takes the given data matrix and outputs the counts (both sec and cc) over the energy range in AU
    :param data: The data matrix: form [time, bin, view, row, column])
    :return: Array of the counts at each AU value, sec counts and cc counts
    """
    length = len(data)  # The number of energy points in AU
    spectrum_sec = np.zeros(length)
    spectrum_cc = np.zeros(length)

    for i in np.arange(5):
        # This finds the median pixel in each capture
        temp_sec = np.squeeze(np.median(data[:, i, :, :, :], axis=[2, 3]))
        temp_cc = np.squeeze(np.median(data[:, i + 6, :, :, :], axis=[2, 3]))

        spectrum_sec = np.add(spectrum_sec, temp_sec)
        spectrum_cc = np.add(spectrum_cc, temp_cc)

    return spectrum_cc, spectrum_sec


def intensity_correction(data, air_data):
    """
    This function corrects flatfield data to show images, -ln(I/I0), I is the intensity of the data, I0 is the intensity
    in an airscan
    :param data: The data to correct (must be the same shape as air_data)
    :param air_data: The airscan data (must be the same shape as data)
    :return: The corrected data array
    """
    return np.log(np.divide(air_data, data))
