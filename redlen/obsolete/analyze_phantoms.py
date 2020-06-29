import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from obsolete import general_OS_functions as gof
import sct_analysis as sct




def get_CNR_over_1s_sum_adj_bin(folder, bins, corr3x3=False, CC=False,
                                directory='C:/Users/10376/Documents/Phantom Data/Uniformity/'):

    num_bins = 6 - (bins[1] - bins[0])
    if corr3x3:
        contrast_mask = np.load(directory + folder + '/3x3_a0_Mask.npy')
        bg_mask = np.load(directory + folder + '/3x3_a0_Background.npy')
    else:
        contrast_mask = np.load(directory + folder + '/a0_Mask.npy')
        bg_mask = np.load(directory + folder + '/a0_Background.npy')

    time_pts = np.arange(0.001, 1.001, 0.001)  # Time points from 0.001 s to 10 s by 0.001 s increments
    CNR_pts = np.zeros([10, num_bins, len(time_pts)])  # Collect CNR over 1 s for all 10 files

    for i in np.arange(1, 11):

        if corr3x3:
            total_data = np.zeros([num_bins, 8, 12])  # Holds the current data for all bins plus the sum of all bins
            add_data = np.load(directory + folder + '/3x3 Corrected Data/Run' + '{:03d}'.format(i) + '_a0.npy')
        else:
            total_data = np.zeros([num_bins, 24, 36])  # Holds the current data for all bins plus the sum of all bins
            add_data = np.load(directory + folder + '/Corrected Data/Run' + '{:03d}'.format(i) + '_a0.npy')

        add_data = np.squeeze(add_data)  # Squeeze out the single capture axis

        if CC:
            add_data = add_data[6:12]  # Grab just cc (or sec) bins
        else:
            add_data = add_data[0:6]  # Grab just sec bins

        add_data = add_adj_bins(add_data, bins)  # Add the appropriate bins together

        for j in np.arange(1000):
            single_frame = add_data[:, j]  # Get the next view data
            total_data[0:num_bins-1] = np.add(total_data[0:num_bins-1], single_frame[0:num_bins-1])  # Add to the current total data
            sum_single_frame = np.sum(single_frame, axis=0)  # Sum all bins to get summed cc (or sec)
            total_data[num_bins-1] = np.add(total_data[num_bins-1], sum_single_frame)  # Add to the total summed

            for k, img in enumerate(total_data):
                # Calculate the CNR (i-1 = file, k = bin, j = view/time point)
                CNR_pts[i-1, k, j], err = sct.cnr(img, contrast_mask, bg_mask)

    CNR_pts = np.mean(CNR_pts, axis=0)  # Average over all of the files

    return time_pts, CNR_pts


def get_CNR_over_time_data_1s(folder, corr3x3=False, CC=True,
                              directory='C:/Users/10376/Documents/Phantom Data/Uniformity/'):

    if corr3x3:
        contrast_mask = np.load(directory + folder + '/3x3_a0_Mask.npy')
        bg_mask = np.load(directory + folder + '/3x3_a0_Background.npy')
    else:
        contrast_mask = np.load(directory + folder + '/a0_Mask.npy')
        bg_mask = np.load(directory + folder + '/a0_Background.npy')

    time_pts = np.arange(0.001, 1.001, 0.001)  # Time points from 0.001 s to 10 s by 0.001 s increments
    CNR_pts = np.zeros([10, 6, len(time_pts)])  # Collect CNR over 1 s for all 10 files
    CNR_err = np.zeros([10, 6])

    for i in np.arange(1, 11):

        if corr3x3:
            total_data = np.zeros([6, 8, 12])  # Holds the current data for all bins plus the sum of all bins
            add_data = np.load(directory + folder + '/3x3 Corrected Data/Run' + '{:03d}'.format(i) + '_a0.npy')
        else:
            total_data = np.zeros([6, 24, 36])  # Holds the current data for all bins plus the sum of all bins
            add_data = np.load(directory + folder + '/Corrected Data/Run' + '{:03d}'.format(i) + '_a0.npy')

        add_data = np.squeeze(add_data)  # Squeeze out the single capture axis

        if CC:
            add_data = add_data[6:12]  # Grab just cc (or sec) bins
        else:
            add_data = add_data[0:6]  # Grab just sec bins

        for j in np.arange(1000):
            single_frame = add_data[:, j]  # Get the next view data
            total_data[0:5] = np.add(total_data[0:5], single_frame[0:5])  # Add to the current total data
            sum_single_frame = np.sum(single_frame, axis=0)  # Sum all bins to get summed cc (or sec)
            total_data[5] = np.add(total_data[5], sum_single_frame)  # Add to the total summed

            for k, img in enumerate(total_data):
                # Calculate the CNR (i-1 = file, k = bin, j = view/time point)
                CNR_pts[i-1, k, j], err = sct.cnr(img, contrast_mask, bg_mask)
                if j == 249:
                    CNR_err[i-1, k] = err

    CNR_pts = np.mean(CNR_pts, axis=0)  # Average over all of the files
    err_mean = np.mean(CNR_err, axis=0)
    err_std = np.std(CNR_err, axis=0)

    return time_pts, CNR_pts, err_mean, err_std


def get_CNR_over_time_energy_thresh(folder, threshold, pxp=1, CC=True, one_frame=True, frames=1000,
                            directory='C:/Users/10376/Documents/Phantom Data/Uniformity/Multiple Energy Thresholds/'):
    """

    :param folder:
    :param threshold:
    :param pxp:
    :param CC:
    :param one_frame:
    :param frames:
    :param directory:
    :return:
    """
    if pxp > 1:
        contrast_mask = np.load(directory + folder + '/' + str(pxp) + 'x' + str(pxp) + '_a0_Mask.npy')
        bg_mask = np.load(directory + folder + '/' + str(pxp) + 'x' + str(pxp) + '_a0_Background.npy')
    else:
        contrast_mask = np.load(directory + folder + '/a0_Mask.npy')
        bg_mask = np.load(directory + folder + '/a0_Background.npy')

    time_pts = np.arange(0.001, frames/1000 + 0.001, 0.001)  # Time points from 0.001 s to 10 s by 0.001 s increments
    CNR_pts = np.zeros([6, frames])  # Collect CNR over 1 s for all 10 files
    CNR_err = np.zeros([6, frames])  # Collect CNR error over 1 s for all 10 files

    if pxp > 1:
        r = int(24/pxp)
        c = int(36/pxp)
        total_data = np.zeros([6, r, c])  # Holds the current data for all bins plus the sum of all bins
        add_data = np.load(directory + folder + '/' + str(pxp) + 'x' + str(pxp) + ' Data/Thresholds_' + str(threshold) +
                           '.npy')
    else:
        total_data = np.zeros([6, 24, 36])  # Holds the current data for all bins plus the sum of all bins
        add_data = np.load(directory + folder + '/Data/Thresholds_' + str(threshold) + '.npy')

    add_data = np.squeeze(add_data)  # Squeeze out the single capture axis

    if CC:
        add_data = add_data[6:12]  # Grab just cc (or sec) bins
    else:
        add_data = add_data[0:6]  # Grab just sec bins

    for j in np.arange(frames):
        single_frame = add_data[:, j]  # Get the next view data
        if one_frame:
            total_data[0:5] = single_frame[0:5]  # switch between the two if looking to average over every frame
        else:
            total_data[0:5] = np.add(total_data[0:5], single_frame[0:5])  # Add to the current total data

        sum_single_frame = np.sum(single_frame, axis=0)  # Sum all bins to get summed cc (or sec)
        if one_frame:
            total_data[5] = sum_single_frame
        else:
            total_data[5] = np.add(total_data[5], sum_single_frame)  # Add to the total summed

        for k, img in enumerate(total_data):
            # Calculate the CNR (i-1 = file, k = bin, j = view/time point)
            CNR_pts[k, j], CNR_err[k, j] = sct.cnr(img, contrast_mask, bg_mask)

    if one_frame:
        CNR_err = np.std(CNR_pts, axis=1)
        CNR_pts = np.mean(CNR_pts, axis=1)

    return time_pts, CNR_pts, CNR_err


def get_CNR_single_adj_bin(folder, bins, threshold, pxp=1, CC=True, one_frame=True, frames=1000,
                                directory='C:/Users/10376/Documents/Phantom Data/Uniformity/'):
    """

    :param folder: The folder within the directory that contains the data
    :param bins: 1D array
                The range of bins to add together, i.e. [1, 4] adds the 1st through the 4th bin
    :param threshold: The test number as an integer
    :param pxp: int, default = 1
                The number of pixels aggregated together (on one side) 1 for 1x1, 2, for 2x2
    :param CC: boolean, default = True
                True if CC data, False if SEC data
    :param one_frame: boolean, default = True
                True if you want an average over the number of frames in frames
    :param frames: int, default = 1000
                The number of frames to go through
    :param directory:
    :return: return
    """
    if pxp > 1:
        contrast_mask = np.load(directory + folder + '/' + str(pxp) + 'x' + str(pxp) + '_a0_Mask.npy')
        bg_mask = np.load(directory + folder + '/' + str(pxp) + 'x' + str(pxp) + '_a0_Background.npy')
    else:
        contrast_mask = np.load(directory + folder + '/a0_Mask.npy')
        bg_mask = np.load(directory + folder + '/a0_Background.npy')

    time_pts = np.arange(0.001, frames/1000 + 0.001, 0.001)  # Time points from 0.001 s to 10 s by 0.001 s increments
    CNR_pts = np.zeros(frames)  # Collect CNR over 1 s for all 10 files

    if pxp > 1:
        r = int(24 / pxp)
        c = int(36 / pxp)
        total_data = np.zeros([6, r, c])  # Holds the current data for all bins plus the sum of all bins
        add_data = np.load(directory + folder + '/' + str(pxp) + 'x' + str(pxp) + ' Data/Thresholds_' + str(threshold) +
                           '.npy')
    else:
        total_data = np.zeros([6, 24, 36])  # Holds the current data for all bins plus the sum of all bins
        add_data = np.load(directory + folder + '/Data/Thresholds_' + str(threshold) + '.npy')

    add_data = np.squeeze(add_data)  # Squeeze out the single capture axis

    if CC:
        add_data = add_data[6:12]  # Grab just cc (or sec) bins
    else:
        add_data = add_data[0:6]  # Grab just sec bins

    add_data = add_adj_bins(add_data, bins)  # sum the desired bins together
    add_data = add_data[bins[0]]  # Get just the new summed bin

    for j in np.arange(frames):
        single_frame = add_data[j]  # Get the next view data
        if one_frame:
            total_data = single_frame
        else:
            total_data = np.add(total_data, single_frame)  # Add to the current total data
        # Calculate the CNR (i-1 = file, k = bin, j = view/time point)
        CNR_pts[j], err = sct.cnr(total_data, contrast_mask, bg_mask)

    if one_frame:
        CNR_err = np.std(CNR_pts)
        CNR_pts = np.mean(CNR_pts)
    else:
        CNR_err = err

    return time_pts, CNR_pts, CNR_err


def find_top_10(folder, sub, directory=r'C:\Users\10376\Documents\Phantom Data\Uniformity\Multiple Energy Thresholds/'):

    files = glob.glob(directory + folder + '/' + sub + '/Single Frame Avg/*')

    file_list = []
    num_list = np.zeros(len(files))

    for i, file in enumerate(files):
        file_list.append(file[-20:-4])
        num_list[i] = np.load(file)[0]

    high_files = []
    high_nums = np.zeros(15)
    for j in np.arange(15):
        idx = np.argmax(num_list)
        high_files.append(file_list[idx])
        high_nums[j] = num_list[idx]
        file_list = np.delete(file_list, idx)
        num_list = np.delete(num_list, idx)

    return high_files, high_nums


def plot_CNR_over_time_1s(time_pts, CNR_pts, CNR_err_mean, CC='CC', title='n/a', save=False,
                       directory='C:/Users/10376/Documents/Phantom Data/Uniformity/'):
    """

    :param time_pts:
    :param CNR_pts:
    :param CNR_err:
    :param title:
    :param save:
    :param directory:
    :return:
    """
    sns.set_style('whitegrid')
    fig, axes = plt.subplots(2, 3, figsize=(8, 6))
    ax1 = fig.add_subplot(111, frameon=False)
    ax1.grid(False)
    ax1.set_xticks([])
    ax1.set_yticks([])
    titles = ['20-30 keV', '30-50 keV', '50-70 keV', '70-90 keV', '90-120 keV', 'Sum ' + CC]

    max_CNR = np.max(CNR_pts) + 0.25
    for i, ax in enumerate(axes.flat):
        ax.plot(time_pts, CNR_pts[i], lw=1)
        ax.set_title(titles[i])
        ax.set_xlim([0, time_pts[-1]])
        ax.set_ylim([0, max_CNR])
        #ax.set_xlabel('CNR error at 0.25 s is \n' + '{:.2f}'.format(CNR_err_mean[i]))

    plt.subplots_adjust(left=0.12, bottom=0.2, right=0.96, top=0.88, wspace=0.31, hspace=0.55)
    ax1.set_xlabel('Acquisition Time (s)', fontsize=14, labelpad=45)
    ax1.set_ylabel('CNR', fontsize=14, labelpad=30)
    ax1.set_title(title, fontsize=15, pad=25)
    plt.show()

    if save:
        plt.savefig(directory + '/Plots/CNR ' + title + '.png', dpi=fig.dpi)
        plt.close()


def plot_CNR_adj_bins(time_pts, CNR_pts, plottitles, title='n/a', save=False,
                      directory='C:/Users/10376/Documents/Phantom Data/Uniformity/'):
    """

    :param time_pts:
    :param CNR_pts:
    :param title:
    :param save:
    :param directory:
    :return:
    """
    sns.set_style('whitegrid')
    fig, axes = plt.subplots(2, 3, figsize=(8, 6))
    ax1 = fig.add_subplot(111, frameon=False)
    ax1.grid(False)
    ax1.set_xticks([])
    ax1.set_yticks([])

    max_CNR = np.max(CNR_pts) + 0.25
    for i, ax in enumerate(axes.flat):
        if i > len(CNR_pts) - 1:
            break
        ax.plot(time_pts, CNR_pts[i][0:len(time_pts)], lw=1)
        ax.set_title(plottitles[i])
        #ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_ylim([0, max_CNR])

    plt.subplots_adjust(left=0.12, bottom=0.2, right=0.96, top=0.88, wspace=0.31, hspace=0.55)
    ax1.set_xlabel('Acquisition Time (s)', fontsize=14, labelpad=45)
    ax1.set_ylabel('CNR', fontsize=14, labelpad=30)
    ax1.set_title(title, fontsize=15, pad=25)
    plt.show()

    if save:
        plt.savefig(directory + '/Plots/' + title + '.png', dpi=fig.dpi)
        plt.close()


def get_single_view_avg_CNR(pxp=[1, 2, 3, 4, 6, 8, 12], frames=1, folders=['1w/', '3w/'], CC=True,
                            directory='C:/Users/10376/Documents/Phantom Data/Uniformity/Multiple Energy Thresholds/'):

    bintitles = [['20-30 keV', '30-50 keV', '50-70 keV', '70-90 keV', '90-120 keV', '20-120 keV'],
                 ['20-30 keV', '30-40 keV', '40-50 keV', '50-60 keV', '60-70 keV', '20-70 keV'],
                 ['20-35 keV', '35-50 keV', '50-65 keV', '65-80 keV', '80-90 keV', '20-90 keV'],
                 ['25-35 keV', '35-45 keV', '45-55 keV', '55-65 keV', '65-75 keV', '25-75 keV'],
                 ['25-40 keV', '40-55 keV', '55-70 keV', '70-80 keV', '80-95 keV', '25-95 keV'],
                 ['30-45 keV', '45-60 keV', '60-75 keV', '75-85 keV', '85-95 keV', '30-95 keV'],
                 ['20-30 keV', '30-70 keV', '70-85 keV', '85-100 keV', '100-120 keV', '20-120 keV']]

    nbt = [[20, 30, 50, 70, 90, 120],
           [20, 30, 40, 50, 60, 70],
           [20, 35, 50, 65, 80, 90],
           [25, 35, 45, 55, 65, 75],
           [25, 40, 55, 70, 80, 95],
           [30, 45, 60, 75, 85, 95],
           [20, 30, 70, 85, 100, 120]]

    if CC:
        bintype = 'CC'
    else:
        bintype = 'SEC'

    # The bin numbers to add together to create larger energy ranges in each test
    bins = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
    for folder in folders:
        for pix in pxp:
            # Create the folder to save the data and choose the savepath for the data
            if pix > 1 and frames == 1:
                gof.create_folder('Single Frame Avg ' + bintype, directory + folder + '/' + str(pix) + 'x' + str(pix) +
                                  ' Data/')
                save_path = directory + folder + '/' + str(pix) + 'x' + str(pix) + ' Data/Single Frame Avg ' + \
                            bintype + '/'

            elif pix > 1 and frames > 1:
                gof.create_folder(str(frames) + ' Frame Avg ' + bintype, directory + folder + '/' + str(pix) + 'x' +
                                  str(pix) + ' Data/')
                save_path = directory + folder + '/' + str(pix) + 'x' + str(pix) + ' Data/' + str(frames) + \
                            ' Frame Avg ' + bintype + '/'

            elif pix == 1 and frames == 1:
                gof.create_folder('Single Frame Avg ' + bintype, directory + folder + 'Data/')
                save_path = directory + folder + 'Data/Single Frame Avg ' + bintype + '/'

            else:
                gof.create_folder(str(frames) + ' Frame Avg ' + bintype, directory + folder + 'Data/')
                save_path = directory + folder + '/Data/' + str(frames) + ' Frame Avg ' + bintype + '/'

            # Go through each of the runs we collected
            for i in np.arange(1, 8):
                if frames == 1:
                    t, c, ce = get_CNR_over_time_energy_thresh(folder, i, pxp=pix, CC=CC, directory=directory)
                else:
                    t, c, ce = get_CNR_over_time_energy_thresh(folder, i, pxp=pix, CC=CC, one_frame=False,
                                                               frames=frames, directory=directory)
                    c = c[:, -1]  # Get the CNR of the last frame (which is what we want)
                    ce = ce[:, -1]

                # Save each of the bins under the proper name
                for j in np.arange(6):
                    np.save(save_path + str(i) + '_' + bintitles[i-1][j] + '.npy', np.array([c[j], ce[j]]))

                for b in bins:
                    if frames == 1:
                        tb, cb, ceb = get_CNR_single_adj_bin(folder, b, i, pxp=pix, CC=CC, directory=directory)
                    else:
                        tb, cb, ceb = get_CNR_single_adj_bin(folder, b, i, pxp=pix, CC=CC, one_frame=False,
                                                             frames=frames, directory=directory)
                        cb = cb[-1]  # Get the last value
                    ths = nbt[i-1]
                    name = str(i) + '_' + str(ths[b[0]]) + '-' + str(ths[b[1]+1]) + ' kev.npy'
                    np.save(save_path + name, np.array([cb, ceb]))
