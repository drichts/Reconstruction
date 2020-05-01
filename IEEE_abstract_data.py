import numpy as np
import general_OS_functions as gof
import matplotlib.pyplot as plt
import seaborn as sns
import glob


def get_median_pixels(data, time=0, energybin=0, view=0):
    """
    Find the index of the pixel with the median value of the data
    :param data: The full data matrix to find the median
    :param time: The time point to look at, default = 0 since many only have 1 time point
    :param energybin: The energy bin to find the median within
    :param view: The view to find
    :return: The pixel(s) indices that contain the median value
    """
    matrix = data[time, energybin, view]  # Grab one capture of the corresponding bin and view
    med_idx = np.argwhere(matrix == (np.percentile(matrix, 50, interpolation='nearest')))

    return np.squeeze(med_idx)


def get_spectrum(data, pixel, energybin=0, view=0):
    """
    Gets the counts in each bin as the
    :param data:
    :param pixel:
    :param energybin:
    :param view:
    :return:
    """
    num_steps = np.arange(np.shape(data)[0])
    if len(np.shape(pixel)) is not 1:
        pixel = pixel[0]
    steps = np.squeeze(data[:, energybin, view, pixel[0], pixel[1]])
    return num_steps, steps


#%%  Open spectra data files, get the actual spectrum, and save

directory = 'C:/Users/10376/Documents/IEEE Abstract/'
load_folder = 'Raw Data/Spectra\\'
save_folder = 'Analysis Data/Spectra\\'

load_path = directory + load_folder
files = glob.glob(load_path + '\*')

for file in files:
    save_name = file.replace(load_path, "")
    save_path = directory + save_folder
    dat = np.load(file)

    pixelsec = get_median_pixels(dat)
    steps_sec, spectrum_sec = get_spectrum(dat, pixelsec)

    pixelcc = get_median_pixels(dat, energybin=6)
    steps_cc, spectrum_cc = get_spectrum(dat, pixelcc, energybin=6)

    np.save(save_path + '/Energy/SEC_' + save_name, steps_sec)
    np.save(save_path + '/Energy/CC_' + save_name, steps_cc)

    np.save(save_path + '/Spectra/SEC_' + save_name, spectrum_sec)
    np.save(save_path + '/Spectra/CC_' + save_name, spectrum_cc)


