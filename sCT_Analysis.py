import numpy as np
from scipy.io import loadmat, whosmat
import general_OS_functions as gof
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import generateROImask as grm
import glob


def main(folder, slice_num='15', directory='D:/Research/Python Data/Spectral CT/', cc=False, re_analyze=False):
    """
    Function takes the folder name, converts the .mat files to .npy files, gets the ROIs for all the vials, normalizes
    the images, and calculates all k_edges
    :param folder:
    :param slice_num:
    :param directory:
    :param cc: True if collecting cc data as well
    :return:
    """
    mat_to_npy(folder, cc=cc)
    image = np.load(directory + folder + '/RawSlices/Bin6_Slice' + slice_num + '.npy')
    continue_flag = True
    if re_analyze:
        continue_flag = False
        masks = np.load(directory + folder + '/Vial_Masks.npy')
        phantom_mask = np.load(directory + folder + '/Phantom_Mask.npy')

    while continue_flag:
        masks = grm.phantom_ROIs(image, radius=7)
        val = input('Were the ROIs acceptable? (y/n)')
        if val is 'y':
            continue_flag = False

    np.save(directory + folder + '/Vial_Masks.npy', masks)

    normalize(folder, cc=cc)

    continue_flag = True
    if re_analyze:
        continue_flag = False

    while continue_flag:
        phantom_mask = grm.entire_phantom(image)
        val = input('Were the ROIs acceptable? (y/n)')
        if val is 'y':
            continue_flag = False
    np.save(directory + folder + '/Phantom_Mask.npy', phantom_mask)
    k_edge(folder, 4, 3)
    k_edge(folder, 3, 2)
    k_edge(folder, 2, 1)
    k_edge(folder, 1, 0)

    if cc:
        k_edge(folder, 11, 10)  # CC 4, 3
        k_edge(folder, 10, 9)  # CC 3, 2
        k_edge(folder, 9, 8)  # CC 2, 1
        k_edge(folder, 8, 7)  # CC 1, 0


def mat_to_npy(folder, load_directory='D:/Research/sCT Scan Data/',
               save_directory='D:/Research/Python Data/Spectral CT/', old=False, cc=False):
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
        s0 = loadmat(path + '/binSEC0_multiplex_corrected2.mat')  # bin 0
        s1 = loadmat(path + '/binSEC1_multiplex_corrected2.mat')  # bin 1
        s2 = loadmat(path + '/binSEC2_multiplex_corrected2.mat')  # bin 2
        s3 = loadmat(path + '/binSEC3_multiplex_corrected2.mat')  # bin 3
        s4 = loadmat(path + '/binSEC4_multiplex_corrected2.mat')  # bin 4
        s5 = loadmat(path + '/binSEC5_multiplex_corrected2.mat')  # bin 5
        s6 = loadmat(path + '/binSEC6_multiplex_corrected2.mat')  # bin 6 (summed bin)

    else:
        s0 = loadmat(path + '/data/binSEC1_test_corrected2_revisit.mat')  # bin 0
        s1 = loadmat(path + '/data/binSEC2_test_corrected2_revisit.mat')  # bin 1
        s2 = loadmat(path + '/data/binSEC3_test_corrected2_revisit.mat')  # bin 2
        s3 = loadmat(path + '/data/binSEC4_test_corrected2_revisit.mat')  # bin 3
        s4 = loadmat(path + '/data/binSEC5_test_corrected2_revisit.mat')  # bin 4
        s5 = loadmat(path + '/data/binSEC6_test_corrected2_revisit.mat')  # bin 5
        s6 = loadmat(path + '/data/binSEC13_test_corrected2_revisit.mat')  # bin 6 (summed bin)

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


def normalize(folder, directory='D:/Research/Python Data/Spectral CT/', cc=False):
    """
    Normalizes the .npy matrices to HU based on the mean value in the water vial
    :param folder: The folder where the mask matrices live
    :param directory:
    :param cc: True if wanting to collect cc data
    :return:
    """
    path = directory+folder
    masks = np.load(path + '/Vial_Masks.npy')
    water_mask = masks[0]  # Water ROI matrix

    load_path = path + '/RawSlices/'
    save_path = path + '/Slices/'

    # Create the Slices folder within the save_path
    gof.create_folder(folder_name='Slices', directory_path=path)

    if cc:
        num = 13
    else:
        num = 7

    for i in np.arange(num):
        for j in np.arange(24):
            # Load the specific slice
            file = 'Bin' + str(i) + '_Slice' + str(j) + '.npy'
            temp = np.load(load_path+file)

            # Get the mean value in the water vial
            water = np.nanmean(temp * water_mask)

            # Normalize the image to HU
            temp = norm_individual(temp, water)

            # Save the normalized matrices
            np.save(save_path+file, temp)
    return


def norm_individual(image, water_value):
    """
    Normalize an individual slice
    :param image: The image to normalize
    :param water_value: The water value to normalize to 0 HU
    :return:
    """
    # Normalize to HU
    image = 1000*np.divide((np.subtract(image, water_value)), water_value)
    # Get rid of any nan values
    image[np.isnan(image)] = -1000

    return image


def image_noise(folder, method='water', directory='D:/Research/Python Data/Spectral CT/', BIN=7, SLICE=0):
    """
    Calculates the image noise of each slice in each bin in either the water ROI or the phantom ROI
    :param folder: folder where the masks live and the .npy Matrices folder is as well
    :param method: 'water' or 'phantom', whether you want the noise of the phantom or water
    :param directory:
    :param BIN: optional, if you want a specific bins noise output
    :param SLICE: optional, if you want a specific slice in a bin output
    :return: noise array (7 x 24, bin x slice)
    """
    path = directory+folder
    masks = np.load(path + '/Vial_Masks.npy')
    if method is 'water':
        bg = masks[0]  # Water ROI matrix
    else:
        bg = np.load(path + '/BackgroundMaskMatrix.npy')  # Background ROI matrix

    # Create a matrix to store the noise in each bin and each slice within that bin
    noise = np.empty([7, 24])
    # Calculate the noise in each slice
    for i in np.arange(7):
        for j in np.arange(24):
            # Load the specific slice
            temp = np.load(path + '/Slices/Bin' + str(i) + '_Slice' + str(j) + '.npy')
            noise[i, j] = np.nanstd(temp*bg)

    if BIN is not 7:
        if SLICE is 0:
            print('The noise for each slice in bin', BIN, 'is:\n', noise[BIN, :])
        else:
            print('The noise for each slice', SLICE, 'in bin', BIN, 'is:', noise[BIN, SLICE])

    np.save(path + '/Image_Noise_' + method, noise)
    return noise


def k_edge(folder, bin_high, bin_low, directory='D:/Research/Python Data/Spectral CT/'):
    """
    This function will take all the slices of the two bins and subtract them from one another to get the K-edge images
    of each slice
    :param folder: the current folder where the scan files live
    :param bin_high: integer of the higher bin (1-4)
    :param bin_low: integer of the lower bin (0-3)
    :param directory: shouldn't change, but the directory with all the different scan data
    :return: Nothing, saves the files needed
    """
    path = directory + folder

    path_K = path + '/K-Edge/'
    path_slices = path + '/RawSlices/'

    # Look for the K-Edge folder in the directory, and create it if it isn't there
    gof.create_folder(folder_name='K-Edge', directory_path=path)

    # Convert the bin numbers to strings
    bin_high = str(bin_high)
    bin_low = str(bin_low)

    for i in np.arange(24):
        slice_num = str(i)

        # Load each high and low bin slice
        image_high = np.load(path_slices + 'Bin' + bin_high + '_Slice' + slice_num + '.npy')
        image_low = np.load(path_slices + 'Bin' + bin_low + '_Slice' + slice_num + '.npy')

        # Create the K-edge image
        kedge_image = np.subtract(image_high, image_low)

        np.save(path_K + 'Bin' + bin_high + '-' + bin_low + '_Slice' + slice_num + '.npy', kedge_image)

    return


def airscan_flux(folder, load_directory='D:/Research/sCT Scan Data/',
                 save_directory='D:/Research/Python Data/Spectral CT/'):
    """
    This function calculates the average total flux in the airscan over the detector in each bin and returns an array
    with the flux in order of bin from SEC0-5, EC
    :param folder:
    :param load_directory:
    :param save_directory:
    :return:
    """
    # These are the bins corresponding to how the files are created in MATLAB
    matlab_bins = ['1', '2', '3', '4', '5', '6', '13']

    load_path = load_directory + folder + '/air_data/'
    save_path = save_directory + folder + '/'

    # Array to save the average total flux in each bin
    total_flux = np.empty(7)

    # Get the average total flux in each bin
    for i, mbin in enumerate(matlab_bins):
        files = glob.glob(load_path + '/Bin_' + mbin + '*.mat')

        flux = 0
        for file in files:
            data_name = whosmat(file)
            data = loadmat(file)
            data = data[data_name[0][0]]
            # Add the sum of the flux to the total
            flux += np.nanmean(data)

        # Take the average (half the number of files because there are two modules)
        flux = flux/(len(files)/2)
        total_flux[i] = flux

        # Save the total flux array
        np.save(save_path + 'Airscan_Flux.npy', total_flux)

    return total_flux


def total_image_noise_stats(image, folder, load=False, directory='D:/Research/Python Data/Spectral CT/'):
    """
    This function takes the 5 ROI for noise from the directory and calculates the std. dev. in each of the ROIs
    in the image
    :param image:
    :param folder:
    :param directory:
    :return:
    """
    if load:
        masks = np.load(directory + folder + '/Noise_Masks.npy')
    else:
        continue_flag = True
        while continue_flag:
            masks = grm.noise_ROIs(image)
            val = input('Were the ROIs acceptable? (y/n)')
            if val is 'y':
                continue_flag = False

        np.save(directory + folder + '/Noise_Masks.npy', masks)

    # Initialize an array to hold the std dev from each of the noise ROIs
    num_masks = len(masks)
    noise = np.empty(num_masks)
    for i in np.arange(num_masks):
        noise_ROI = masks[i]

        noise[i] = np.nanstd(noise_ROI*image)

    # Save the noise matrix to the folder
    mean_noise = np.mean(noise)
    std_noise = np.std(noise)

    return mean_noise, std_noise


def cnr(image, contrast_mask, background_mask):
    """
    This function calculates the CNR of an ROI given the image, the ROI mask, and the background mask
    It also gives the CNR error
    :param image: The image to be analyzed as a 2D numpy array
    :param contrast_mask: The mask of the contrast area as a 2D numpy array
    :param background_mask: The mask of the background as a 2D numpy array
    :return CNR, CNR_error: The CNR and error of the contrast area
    """
    # The mean signal within the contrast area
    mean_ROI = np.nanmean(image*contrast_mask)
    std_ROI = np.nanstd(image*contrast_mask)

    # Mean and std. dev. of the background
    bg = np.multiply(image, background_mask)
    mean_bg = np.nanmean(bg)
    std_bg = np.nanstd(bg)

    CNR = abs(mean_ROI - mean_bg) / std_bg
    CNR_err = np.sqrt(std_ROI**2 + std_bg**2) / std_bg

    return CNR, CNR_err


def get_ct_cnr(folder, z, type='water', directory='D:/Research/Python Data/Spectral CT/'):
    """
    Get the cnr for each of the vial ROIs in a specific slice
    :param folder:
    :param z: Slice to look at
    :param directory:
    :return:
    """

    path = directory + folder + '/Slices/'
    vials = np.load(directory + folder + '/Vial_Masks.npy')
    back = np.load(directory + folder + '/Phantom_Mask.npy')
    CNR = np.zeros(len(vials))
    CNR_err = np.zeros(len(vials))

    image = np.load(path + 'Bin6_Slice' + str(z) + '.npy')
    for i, vial in enumerate(vials):
        if type is 'water':
            CNR[i], CNR_err[i] = cnr(image, vial, vials[0])
        else:
            CNR[i], CNR_err[i] = cnr(image, vial, back)

    return CNR, CNR_err


def find_least_noise(folder, low_slice, high_slice, directory='D:/Research/Python Data/Spectral CT/'):
    """
    Find the slice with the least noise in the summed bin
    :param folder: folder to examine
    :param low_slice:
    :param high_slice:
    :param directory: where the folder is located
    :return:
    """
    subfolder = '/Slices/'
    path = directory + folder + subfolder
    vials = np.load(directory + folder + '/Vial_Masks.npy')
    noise_vals = np.zeros(high_slice-low_slice+1)
    for i in np.arange(low_slice, high_slice+1):
        img = np.load(path + 'Bin6_Slice' + str(i) + '.npy')
        noise_vals[i-low_slice] = np.nanstd(vials[0]*img)

    idx = np.argmin(noise_vals)
    return idx+low_slice, noise_vals[idx]


def open_ct_image(folder, b, z, show=True, directory='D:/Research/Python Data/Spectral CT/'):

    path = directory + folder + '/Slices/'

    img = np.load(path + 'Bin' + str(b) + '_Slice' + str(z) + '.npy')

    if show:
        plt.imshow(img, cmap='gray', vmin=-500, vmax=1000)

    return img


def open_kedge_image(folder, b, z, show=True, colormap=3, directory='D:/Research/Python Data/Spectral CT/'):

    path = directory + folder + '/K-Edge/'

    img = np.load(path + 'Bin' + b + '_Slice' + str(z) + '.npy')
    # Create the colormaps

    nbins = 100
    c1 = (1, 0, 1)
    c2 = (0, 1, 0)
    c3 = (1, 0.843, 0)
    c4 = (0, 0, 1)

    gray_val = 0
    gray_list = (gray_val, gray_val, gray_val)

    c1_rng = [gray_list, c1]
    cmap1 = colors.LinearSegmentedColormap.from_list('Purp', c1_rng, N=nbins)
    c2_rng = [gray_list, c2]
    cmap2 = colors.LinearSegmentedColormap.from_list('Gree', c2_rng, N=nbins)
    c3_rng = [gray_list, c3]
    cmap3 = colors.LinearSegmentedColormap.from_list('G78', c3_rng, N=nbins)
    c4_rng = [gray_list, c4]
    cmap4 = colors.LinearSegmentedColormap.from_list('Blu8', c4_rng, N=nbins)

    clr_maps = {1: cmap1, 2: cmap2, 3: cmap3, 4: cmap4}

    if show:
        plt.imshow(img, cmap=clr_maps[colormap], vmin=0, vmax=0.01)
        plt.show()

    return img


def mean_ROI_value(image, vial):

    # Get the matrix with only the values in the ROI
    value = np.multiply(image, vial)

    # Calculate the mean value in the matrix
    mean_val = np.nanmean(value)

    return mean_val


def find_norm_value(folder, good_slice, vial, edge, subtype=2, directory='D:/Research/Python Data/Spectral CT/'):
    """
    Find the mean value of the highest concentration and of water to normalize images to concentration
    :param folder:
    :param good_slice:
    :param vial:
    :param edge:
    :param subtype:
    :param directory:
    :return:
    """
    low_slice, high_slice = good_slice[0], good_slice[1]

    # Define subfolder with subtype
    subfolder = {1: '/Slices/',
                 2: '/K-Edge/'}

    # Define the specific K-edge
    bin_edge = {0: 'Bin1-0_',
                1: 'Bin2-1_',
                2: 'Bin3-2_',
                3: 'Bin4-3_'}

    # Vial ROIs
    rois = np.load(directory + folder + '/Vial_Masks.npy')
    value_roi = rois[vial]  # Specific vial ROI
    water_roi = rois[0]

    # Go through all the good slices to find the mean value in each slice
    slice_values = np.empty(high_slice-low_slice)
    zero_values = np.empty(high_slice-low_slice)

    for z in np.arange(low_slice, high_slice):
        image = np.load(directory + folder + subfolder[subtype] + bin_edge[edge] + 'Slice' + str(z) + '.npy')
        slice_values[z-low_slice] = np.nanmean(image*value_roi)
        zero_values[z-low_slice] = np.nanmean(image*water_roi)

    # Get average value
    norm_value = np.mean(slice_values)
    water_value = np.mean(zero_values)

    return water_value, norm_value


def linear_fit(zero_value, norm_value):
    """
    Find a linear fit between 0 and 5% concentration
    :param zero_value: the water value (0%) concentration
    :param norm_value: the 5% concentration value
    :return:
    """
    coeffs = np.polyfit([zero_value, norm_value], [0, 5], 1)

    return coeffs

def norm_kedge(folder, coeffs, edge, directory='D:/Research/Python Data/Spectral CT/'):
    """
    Normalize the k-edge images and save in a new folder (Normed K-Edge)
    :param folder:
    :param coeffs:
    :param edge: int
                The K-edge image to look at, 0 = 1-0, 1 = 2-1, 2 = 3-2, 4 = 4-3
    :param directory:
    :return:
    """
    # Define the specific K-edge
    bin_edge = {0: 'Bin1-0_',
                1: 'Bin2-1_',
                2: 'Bin3-2_',
                3: 'Bin4-3_'}

    path = directory + folder + '/'

    # Create the folder /Normed K-Edge/ if necessary
    gof.create_folder(folder_name='Normed K-Edge', directory_path=path)

    load_path = path + 'K-Edge/'
    save_path = path + 'Normed K-Edge/'

    # The linear fit
    l_fit = np.poly1d(coeffs)

    # Normalize each slice and save it
    for z in np.arange(24):
        file = bin_edge[edge] + 'Slice' + str(z) + '.npy'

        # Load the image and normalize it to the norm_value
        image = np.load(load_path + file)

        # Norm between 0 and 1 and then multiply by norm value
        image = l_fit(image)
        # Save the new image in the new location
        np.save(save_path + file, image)

    return

def sum_sec_cc(folder, sec=True, cc=False, directory='D:/Research/Python Data/Spectral CT/'):
    gof.create_folder(folder_name='CT Sum Slices', directory_path=directory+folder)
    if sec:
        for z in np.arange(24):
            s0 = np.load(directory + folder + '/Slices/Bin0_Slice' + str(z) + '.npy')
            s1 = np.load(directory + folder + '/Slices/Bin1_Slice' + str(z) + '.npy')
            s2 = np.load(directory + folder + '/Slices/Bin2_Slice' + str(z) + '.npy')
            s3 = np.load(directory + folder + '/Slices/Bin3_Slice' + str(z) + '.npy')
            s4 = np.load(directory + folder + '/Slices/Bin4_Slice' + str(z) + '.npy')

            s_sec = s0 + s1 + s2 + s3 + s4
            np.save(directory + folder + '/CT Sum Slices/SEC_' + str(z) + '.npy', s_sec)

    if cc:
        for z in np.arange(24):
            s7 = np.load(directory + folder + '/Slices/Bin7_Slice' + str(z) + '.npy')
            s8 = np.load(directory + folder + '/Slices/Bin8_Slice' + str(z) + '.npy')
            s9 = np.load(directory + folder + '/Slices/Bin9_Slice' + str(z) + '.npy')
            s10 = np.load(directory + folder + '/Slices/Bin10_Slice' + str(z) + '.npy')
            s11 = np.load(directory + folder + '/Slices/Bin11_Slice' + str(z) + '.npy')

            s_cc = s7 + s8 + s9 + s10 + s11
            np.save(directory + folder + '/CT Sum Slices/CC_' + str(z) + '.npy', s_cc)
    return

