from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import general_OS_functions as gof
import sCT_Analysis as sct
import generateROImask as grm


def testalign(folder, air_folder, num, directory='X:/TEST LOG/MINI MODULE/Canon/M20358_D32/'):
    """
    This function is to test the inital alignment of a phantom at Redlen
    :param folder: The folder name in the directory
    :param air_folder: The airscan folder name in the directory
    :param num: The file number to look at in Uniformity (only A0 files)
    :param directory: The file path where the test folders save to
    :return: shows an image of the phantom after air correction
    """
    datapath = directory + folder + '/Raw Test Data/M20358_D32/UNIFORMITY/*A0*'
    airpath = directory + air_folder + '/Raw Test Data/M20358_D32/UNIFORMITY/*A0*'

    datafiles = glob.glob(datapath)
    airfiles = glob.glob(airpath)

    # Grab the correct data files
    data = np.squeeze(mat_to_npy(datafiles[num]))
    air = np.squeeze(mat_to_npy(airfiles[num]))

    # Do the air correction
    corr = -1*np.log(np.divide(data, air))
    corr = np.sum(corr, axis=1)

    plt.imshow(corr[12])
    plt.pause(10)
    plt.close()


def npy_uniformity(folder, air_folder, small_contrast=False, w=1, boxname='M20358_D32', mm=0,
                   pxp=[1, 2, 3, 4, 6, 8, 12], directory='X:/TEST LOG/MINI MODULE/Canon/',
                   savedir='C:/Users/10376/Documents/Phantom Data/Polypropylene_06-12-20/', reanalyze=False):
    """
    This function does a lot of analysis: creates masks for the contrast area and background at each pixel aggregation,
    analyzes to calculate CNR of each bin at each of the pixel aggs and at a number of different time frames =
    :param folder: string
                The name of the folder where the .mat phantom data is
    :param air_folder: string
                The name of the folder where the .mat airscan is
    :param small_contrast: boolean, default=False
                If you are working with a small contrast agent, less than 2-3 mm, should be changed to True
    :param w: int, default is 1
                The charge sharing correction window width
    :param boxname: string, default is 'M20358_D32'
                The box name of the detector used
    :param mm: int, default is 0
                The ASIC data to choose, could be 0 or 1 (A0 or A1)
    :param pxp: list of ints, default is [1, 2, 3, 4, 6, 8, 10, 12]
                The number of pixels to aggregate
    :param directory: string
                The directory leading to the folder with box name in it first
    :param savedir: string
                The path to the folder where the data from this phantom should be saved
    :param reanalyze: boolean, default is False
                Whether you are re-analyzing the data or not
    :return:
    """
    # Modifiers for loading and saving data
    module = {0: '*A0*', 1: '*A1*'}
    save_mm = {0: 'a0_', 1: 'a1_'}

    gof.create_folder(folder, savedir)  # Create the folder in the save directory
    savedir = savedir + '/' + folder  # Update the save directory

    # The full data path to get either A0 or A1 files
    datapath = directory + '/' + boxname + '/' + folder + '/Raw Test Data/' + boxname + '/UNIFORMITY/' + module[mm]
    airpath = directory + '/' + boxname + '/' + air_folder + '/Raw Test Data/' + boxname + '/UNIFORMITY/' + module[mm]

    # Glob all data files with with A0 or A1
    datafiles = glob.glob(datapath)
    airfiles = glob.glob(airpath)
    print(datapath)
    print(airpath)

    for i in np.arange(len(datafiles)):

        filename = 'TestNumData_' + save_mm[mm] + str(i+1)  # Filename for this test's data
        airname = 'TestNumAir_' + save_mm[mm] + str(i+1)   # Filename for this test's air

        print(os.path.basename(datafiles[i]))
        print(os.path.basename(airfiles[i]))
        print()

        # Takes a long time to load a .mat file so if reanalyzing, load from the already saved .npy
        if reanalyze:
            data = np.expand_dims(np.load(savedir + '/1x1 Data/Raw Data/' + filename + '.npy'), axis=0)
            air = np.expand_dims(np.load(savedir + '/1x1 Data/Raw Data/' + airname + '.npy'), axis=0)
        else:
            data = mat_to_npy(datafiles[i])
            air = mat_to_npy(airfiles[i])

        # Aggregate the number of pixels in pxp squared
        for pix in pxp:
            if pix == 1:
                data_pxp = np.squeeze(data)
                air_pxp = np.squeeze(air)
            else:
                data_pxp = np.squeeze(sumpxp(data, pix))  # Aggregate the pixels
                air_pxp = np.squeeze(sumpxp(air, pix))

            pixel_name = str(pix) + 'x' + str(pix)
            gof.create_folder(pixel_name + ' Data', savedir)
            savepath = savedir + '/' + pixel_name + ' Data/'  # Update the save path

            gof.create_folder('Raw Data', savepath)  # Create folder for the raw data files (not corrected for air)
            np.save(savepath + 'Raw Data/' + filename + '.npy', data_pxp)  # Save the raw data
            np.save(savepath + 'Raw Data/' + airname + '.npy', air_pxp)  # Save the air data

            # Don't find new masks if reanalyzing for past the first file (all files will be of the same phantom)
            if w == 1:
                if reanalyze or i > 0:
                    mask = np.load(savepath + save_mm[mm] + 'Mask.npy')
                    bg = np.load(savepath + save_mm[mm] + 'Background.npy')
                else:
                    # Sum over all views to find background and contrast area
                    temp_data = np.sum(data_pxp[6:], axis=1)
                    temp_air = np.sum(air_pxp[6:], axis=1)
                    image = intensity_correction(temp_data, temp_air)  # Correct for air
                    # Test which bin shows the contrast area before defining the mask ROIs
                    ideal_bin = test_visibilty(image)
                    image = image[ideal_bin]

                    # Get a contrast and background mask for this pixel aggregation and save
                    mask, bg = choose_mask_types(image, small_contrast, pix)
                    np.save(savepath + save_mm[mm] + 'Mask.npy', mask)
                    np.save(savepath + save_mm[mm] + 'Background.npy', bg)
            else:
                if reanalyze or i > 0:
                    mask = np.load(savepath + save_mm[mm] + 'Mask.npy')
                    bg = np.load(savepath + save_mm[mm] + 'Background.npy')
                else:
                    maskpath = savepath
                    print(maskpath)
                    w_idx = maskpath.find(str(w) + 'w')
                    w_repl = maskpath[w_idx:w_idx+2]
                    maskpath = maskpath.replace(w_repl, '1w')
                    print(maskpath)
                    print()
                    mask = np.load(maskpath + save_mm[mm] + 'Mask.npy')
                    bg = np.load(maskpath + save_mm[mm] + 'Background.npy')
                    np.save(savepath + save_mm[mm] + 'Mask.npy', mask)
                    np.save(savepath + save_mm[mm] + 'Background.npy', bg)

            # Collect the frames aggregated over, the noise and cnr and save
            frames, cnr, noise = avg_cnr_noise_over_all_frames(data_pxp, air_pxp, mask, bg)
            np.save(savepath + filename + '_Frames.npy', frames)
            np.save(savepath + filename + '_Avg_CNR.npy', cnr)
            np.save(savepath + filename + '_Avg_noise.npy', noise)


def test_visibilty(image):
    """
    This function will test if your contrast area is visible in the current bin
    :param image: 3D ndarray
                The images with all bins
    :return: The bin to use
    """
    bins = np.roll(np.arange(len(image)), 1)  # The bin options from EC, first, second, etc.
    for b in bins:
        plt.imshow(image[b])
        plt.pause(1)
        plt.close()
        val = input('Was the contrast visible? (y/n)')
        if val is 'y':
            return b

    return np.nan


def choose_mask_types(image, small, pixels):
    """
    This function chooses the mask function to call based on if you are aggregating lots of pixels together and if
    the contrast area is small. If the contrast area is small, the contrast mask will chosen pixel by pixel, otherwise
    it will be selected by choosing the corners of a rectangle. The same is true for the background mask if you aggregate
    a lot of pixels, you need to choose the background mask pixel by pixel
    :param image: 2D ndarray
                The image to mask
    :param small: boolean
                True if the contrast area is small
    :param pixels: int
                The number of pixels being aggregated together
    :return:
    """
    # First get the contrast mask, if it is a small contrast area (<2mm) choose the ROI pixel by pixel
    # It will ask you after selecting if the ROI is acceptable, anything but y will allow you to redefine the ROI
    continue_flag = True
    if small or pixels > 5:
        while continue_flag:
            mask = grm.single_pixels_mask(image)
            val = input('Were the ROIs acceptable? (y/n)')
            if val is 'y':
                continue_flag = False
    else:
        while continue_flag:
            mask = grm.square_ROI(image)
            val = input('Were the ROIs acceptable? (y/n)')
            if val is 'y':
                continue_flag = False

    # Choose the background mask in the same way, if you are aggregating more pixels than 5, you will select pixel by pixel
    continue_flag = True
    if pixels < 5:
        while continue_flag:
            bg = grm.square_ROI(image)
            val = input('Were the ROIs acceptable? (y/n)')
            if val is 'y':
                continue_flag = False
    else:
        while continue_flag:
            bg = grm.single_pixels_mask(image)
            val = input('Were the ROIs acceptable? (y/n)')
            if val is 'y':
                continue_flag = False

    return mask, bg


def avg_cnr_noise_over_frames(data, airdata, mask, bg_mask, frames):
    """
    This function will take the data and airscan and calculate the CNR every number of frames and then avg, will also
    give the CNR error as well. Also will calculate the avg noise
    :param data: 4D ndarray, <counters, views, rows, columns>
                The phantom data
    :param airdata: 4D ndarray, <counters, views, rows, columns>
                The airscan
    :param frames: int
                The number of frames to avg together
    :param mask: 2D ndarray
                The mask of the contrast area
    :param bg_mask: 2D ndarray
                The mask of the background
    :return: four lists with the cnr, cnr error, noise, and std of the noise in each of the bins (usually 13 elements)
    """
    cnr = np.zeros([len(data), int(1000/frames)])  # Array to hold the cnr for the number of times it will be calculated
    cnr_err = np.zeros([len(data), int(1000/frames)])  # array for the cnr error
    noise = np.zeros([len(data), int(1000/frames)])  # Same for noise

    # Go over the data views in jumps of the number of frames
    for i, data_idx in enumerate(np.arange(0, 1001-frames, frames)):
        if frames == 1:
            tempdata = data[:, data_idx]  # Grab the next view
            tempair = airdata[:, data_idx]
        else:
            tempdata = np.sum(data[:, data_idx:data_idx + frames], axis=1)  # Grab the sum of the next 'frames' views
            tempair = np.sum(airdata[:, data_idx:data_idx + frames], axis=1)

        corr_data = intensity_correction(tempdata, tempair)  # Correct for air

        # Go through each bin and calculate CNR
        for j, img in enumerate(corr_data):
            cnr[j, i], cnr_err[j, i] = sct.cnr(img, mask, bg_mask)
            noise[j, i] = np.nanstd(img*bg_mask)  # Get noise as fraction of mean background

    # Average over the frames
    cnr = np.mean(cnr, axis=1)
    cnr_err = np.mean(cnr_err, axis=1)
    noise_std = np.std(noise, axis=1)
    noise = np.mean(noise, axis=1)

    return cnr, cnr_err, noise, noise_std


def avg_cnr_noise_over_all_frames(data, airdata, mask, bg_mask,
                                  frames=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 50, 100, 250, 500, 1000]):
    """
    This function will take the data and airscan and calculate the CNR and CNR error over all the frames in the list
    :param data: 4D ndarray, <counters, views, rows, columns>
                The phantom data
    :param airdata: 4D ndarray, <counters, views, rows, columns>
                The airscan
    :param mask: 2D ndarray
                The mask of the contrast area
    :param bg_mask: 2D ndarray
                The mask of the background
    :param frames: 1D ndarray
                List of the frames to avg over
    :return:
    """
    cnr_frames = np.zeros([len(frames), len(data), 2])  # The CNR and CNR error over all frames in the list
    noise_frames = np.zeros([len(frames), len(data), 2])  # Same for noise

    for i in np.arange(len(frames)):
        c, ce, n, ne = avg_cnr_noise_over_frames(data, airdata, mask, bg_mask, frames[i])  # Calculate the CNR and error
        cnr_frames[i, :, 0] = c   # The ith frames, set first column equal to cnr
        cnr_frames[i, :, 1] = ce  # The ith frames, set second column equal to cnr error
        noise_frames[i, :, 0] = n  # Same for noise, 1st column
        noise_frames[i, :, 1] = ne  # Noise error, 2nd column

    return frames, cnr_frames, noise_frames


def avg_contrast_over_frames(data, airdata, mask, bg_mask, frames):
    """
    This function will take the data and airscan and calculate the contrast as a fraction of mean background for every
    number of frames and then avg
    :param data: 4D ndarray, <counters, views, rows, columns>
                The phantom data
    :param airdata: 4D ndarray, <counters, views, rows, columns>
                The airscan
    :param frames: int
                The number of frames to avg together
    :param mask: 2D ndarray
                The mask of the contrast area
    :param bg_mask: 2D ndarray
                The mask of the background
    :return: four lists with the cnr, cnr error, noise, and std of the noise in each of the bins (usually 13 elements)
    """
    contrast = np.zeros([len(data), int(1000/frames)])  # Array to hold the cnr for the number of times it will be calculated

    # Go over the data views in jumps of the number of frames
    for i, data_idx in enumerate(np.arange(0, 1001-frames, frames)):
        if frames == 1:
            tempdata = data[:, data_idx]  # Grab the next view
            tempair = airdata[:, data_idx]
        else:
            tempdata = np.sum(data[:, data_idx:data_idx + frames], axis=1)  # Grab the sum of the next 'frames' views
            tempair = np.sum(airdata[:, data_idx:data_idx + frames], axis=1)

        corr_data = intensity_correction(tempdata, tempair)  # Correct for air

        # Go through each bin and calculate CNR
        for j, img in enumerate(corr_data):
            #background = np.nanmean(img*bg_mask)
            #contrast[j, i] = np.nanmean(img*mask) - background
            contrast[j, i] = np.nanstd(img*bg_mask)

    # Average over the frames
    contrast_err = np.std(contrast, axis=1)
    contrast = np.mean(contrast, axis=1)

    return contrast, contrast_err


def avg_contrast_over_all_frames(data, airdata, mask, bg_mask,
                                  frames=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 50, 100, 250, 500, 1000]):
    """
    This function will take the data and airscan and calculate the contrast and contrast error over all the frames in
    the list
    :param data: 4D ndarray, <counters, views, rows, columns>
                The phantom data
    :param airdata: 4D ndarray, <counters, views, rows, columns>
                The airscan
    :param mask: 2D ndarray
                The mask of the contrast area
    :param bg_mask: 2D ndarray
                The mask of the background
    :param frames: 1D ndarray
                List of the frames to avg over
    :return:
    """
    contrast_frames = np.zeros([len(frames), len(data), 2])  # The contrast and contrast error over frames in the list

    for i in np.arange(len(frames)):
        c, ce, = avg_contrast_over_frames(data, airdata, mask, bg_mask, frames[i])  # Calculate the contrast
        contrast_frames[i, :, 0] = c   # The ith frames, set first column equal to contrast
        contrast_frames[i, :, 1] = ce  # The ith frames, set second column equal to contrast error

    return contrast_frames


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


def polyprop_mult_energy(pxp=[12]):
    """
    Goes through the two folders holding all the energy threshold data and calculates the air corrected
    data and 3x3 data
    :return:
    """
    load_dir = r'X:\TEST LOG\MINI MODULE\Canon\M20358_Q20/'
    save_dir = r'C:\Users\10376\Documents\Phantom Data\Uniformity/Multiple Energy Thresholds/'

    save_folder = ['/1w/', '/3w/']  # Two folders in the save directory

    folders = ['multiple_energy_thresholds_1w', 'multiple_energy_thresholds_3w']
    air_folders = ['multiple_energy_thresholds_flatfield_1w', 'multiple_energy_thresholds_flatfield_3w']

    # Go through the two folders and the corresponding air folders
    for i in np.arange(2):
        files = glob.glob(load_dir + folders[i] + '/Raw Test Data/M20358_Q20/UNIFORMITY/*A0*')
        air_files = glob.glob(load_dir + air_folders[i] + '/Raw Test Data/M20358_Q20/UNIFORMITY/*A0*')

        # Go through each file and it's corresponding air file
        for j in np.arange(len(files)):
            print(files[j][-100:-25])
            print(air_files[j][-100:-25])

            data = mat_to_npy(files[j])
            air_data = mat_to_npy(air_files[j])

            for pix in pxp:
                datapxp = sumpxp(data, pix)
                air_datapxp = sumpxp(air_data, pix)
                corrpxp = intensity_correction(datapxp, air_datapxp)

                gof.create_folder(str(pix) + 'x' + str(pix) + ' Data', save_dir + save_folder[i])

                np.save(save_dir + save_folder[i] + str(pix) + 'x' + str(pix) + ' Data/Thresholds_' + str(j+1) + '.npy',
                        corrpxp)

            corr = intensity_correction(data, air_data)

            # Save the data, j corresponds to the NDT number in the filename and the threshold settings in my notes
            # See redlen notebook
            np.save(save_dir + save_folder[i] + 'Data/Thresholds_' + str(j+1) + '.npy', corr)


def sumpxp(data, num_pixels):
    """
    This function takes a data array and sums nxn pixels along the row and column data
    :param data: 4D array
                The full data array <captures, counters, views, rows, columns>
    :return: The new data array with nxn pixels from the inital data summed together
    """
    dat_shape = np.array(np.shape(data))
    dat_shape[3] = int(dat_shape[3] / num_pixels)  # Reduce size by num_pixels in the row and column directions
    dat_shape[4] = int(dat_shape[4] / num_pixels)

    ndata = np.zeros(dat_shape)
    n = num_pixels
    for row in np.arange(dat_shape[3]):
        for col in np.arange(dat_shape[4]):
            temp = data[:, :, :, n*row:n*row+n, n*col:n*col+n]  # Get each 2x2 subarray over all of the first 2 axes
            ndata[:, :, :, row, col] = np.sum(temp, axis=(3, 4))  # Sum over only the rows and columns

    return ndata

directory = r'C:\Users\10376\Documents\Phantom Data\Polypropylene_06-12-20/'
folders = ['polyprop_1w_deadtime_32ns/', 'polyprop_3w_deadtime_32ns/', 'polyprop_8w_deadtime_32ns/']

for folder in folders:
    path = directory + folder + '1x1 Data/'
    data = np.load(path + 'Raw Data/TestNumData_a0_1.npy')
    air = np.load(path + 'Raw Data/TestNumAir_a0_1.npy')
    mask = np.load(path + 'a0_Mask.npy')
    bg = np.load(path + 'a0_Background.npy')

    con = avg_contrast_over_all_frames(data, air, mask, bg)
    np.save(path + 'TestNumData_a0_1_Avg_noise.npy', con)