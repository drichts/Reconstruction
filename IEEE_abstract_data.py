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
    return steps


#%%  Open spectra data files, get the actual spectrum, and save

def save_spectra():
    directory = 'C:/Users/10376/Documents/IEEE Abstract/'
    load_folder = 'Raw Data/Spectra\\'
    save_folder = 'Analysis Data/Spectra\\'

    load_path = directory + load_folder
    files = glob.glob(load_path + '\*')

    for file in files:
        save_name = file.replace(load_path, "")
        save_path = directory + save_folder
        dat = np.load(file)

        length = len(dat)
        spectrum_sec = np.zeros(length)
        spectrum_cc = np.zeros(length)

        for i in np.arange(5):

            # This section is for finding the median pixel in each capture
            temp_sec = np.squeeze(np.median(dat[:, i, :, :, :], axis=[2, 3]))
            temp_cc = np.squeeze(np.median(dat[:, i+6, :, :, :], axis=[2, 3]))

            spectrum_sec = np.add(spectrum_sec, temp_sec)
            spectrum_cc = np.add(spectrum_cc, temp_cc)

            np.save(save_path + '/SEC_' + save_name, spectrum_sec)
            np.save(save_path + '/CC_' + save_name, spectrum_cc)

    return

save_spectra()
#%% Flat field air

def flatfield_air():
    folder = 'C:/Users/10376/Documents/IEEE Abstract/Analysis Data/Flat Field/'

    print('1w           4w')
    w1 = np.load(folder + '/flatfield_1wA0.npy')
    w4 = np.load(folder + '/flatfield_4wA0.npy')

    w1 = np.squeeze(w1)
    w4 = np.squeeze(w4)


    for i in np.arange(5):
        print('Bin' + str(i))
        print(np.nanstd(w1[i+6])/np.mean(w1[i+6]))
        print(np.nanstd(w4[i+6])/np.mean(w4[i+6]))

flatfield_air()


#%% Flatfield phantoms (correct for air)

def flatfield_phantoms():
    directory = 'C:/Users/10376/Documents/IEEE Abstract/'
    load_folder = 'Raw Data/Flat Field\\'
    save_folder = 'Analysis Data/Flat Field/'

    load_path = directory + load_folder
    air1w = np.squeeze(np.load(load_path + '/flatfield_1w.npy'))
    air4w = np.squeeze(np.load(load_path + '/flatfield_4w.npy'))

    blue1w = np.squeeze(np.load(load_path + 'bluebelt_1w.npy'))
    blue4w = np.squeeze(np.load(load_path + 'bluebelt_4w.npy'))

    plexi1w = np.squeeze(np.load(load_path + 'plexiglass_1w.npy'))
    plexi4w = np.squeeze(np.load(load_path + 'plexiglass_4w.npy'))

    # Take average of the air fields
    #air1w = np.sum(air1w, axis=1)
    #air4w = np.sum(air4w, axis=1)
    #air1w = np.divide(air1w, 1000)
    #air4w = np.divide(air4w, 1000)

    blue1w = np.divide(blue1w, air1w)
    blue4w = np.divide(blue4w, air4w)

    plexi1w = np.divide(plexi1w, air1w)
    plexi4w = np.divide(plexi4w, air4w)


    #for i in np.arange(1000):
        #blue1w[:, i, :, :] = np.divide(blue1w[:, i, :, :], air1w)
        #blue4w[:, i, :, :] = np.divide(blue4w[:, i, :, :], air4w)

        #plexi1w[:, i, :, :] = np.divide(plexi1w[:, i, :, :], air1w)
        #plexi4w[:, i, :, :] = np.divide(plexi4w[:, i, :, :], air4w)

    blue1w = -np.log(blue1w)
    blue4w = -np.log(blue4w)

    plexi1w = -np.log(plexi1w)
    plexi4w = -np.log(plexi4w)

    np.save(directory + save_folder + 'bluebelt_1w.npy', blue1w)
    np.save(directory + save_folder + 'bluebelt_4w.npy', blue4w)

    np.save(directory + save_folder + 'plexiglass_1w.npy', plexi1w)
    np.save(directory + save_folder + 'plexiglass_4w.npy', plexi4w)
    #yyy = np.zeros([13, 1000, 24, 72])
    #yyy[:, 0, :, :] = np.divide(blue1w[:, 0, :, :], air1w)
    return #np.divide(blue1w[:, 0, :, :], air1w), yyy

flatfield_phantoms()
