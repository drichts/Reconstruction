import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import general_OS_functions as gof
import sCT_Analysis as sct

folders = ['m20358_q20_al_bluebelt_acryl_1w', 'm20358_q20_al_bluebelt_acryl_4w',
           'm20358_q20_al_bluebelt_fat_1w', 'm20358_q20_al_bluebelt_fat_4w',
           'm20358_q20_al_bluebelt_solidwater_1w', 'm20358_q20_al_bluebelt_solidwater_4w',
           'm20358_q20_al_polypropylene_1w', 'm20358_q20_al_polypropylene_4w']

air_folders = ['m20358_q20_al_air_1w', 'm20358_q20_al_air_4w']

a1_pixels = np.array([[0, 0], [0, 2], [0, 3], [23, 17], [16, 32], [13, 32], [8, 35], [11, 35], [12, 35], [14, 35],
                      [17, 35]])


def get_CNR_over_time_data_raw(folder, air_folder, directory='C:/Users/10376/Documents/Phantom Data/Uniformity/'):
    """
    This function takes a folder and collects the CNR is all CC bins at 0.001, 0.01, 0.1, 0.5, 1, 2, 3.... seconds
    :param folder:
    :param air_folder:
    :param directory:
    :return:
    """
    contrast_mask = np.load(directory + folder + '/a0_Mask.npy')
    bg_mask = np.load(directory + folder + '/a0_Background.npy')

    air = np.load(directory + air_folder + '/Raw Data/a0.npy')
    air = np.squeeze(air)

    data = np.load(directory + folder + '/Raw Data/Run001_a0.npy')
    data = np.squeeze(data)

    time_pts = np.array([0.001, 0.01, 0.1, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    CNR_pts = np.zeros([6, len(time_pts)])
    CNR_err = np.zeros([6, len(time_pts)])

    # The first fractions of a second results
    for i in np.arange(5):
        slices = int(time_pts[i]*1000)
        temp_data = np.sum(data[:, 0:slices, :, :], axis=1)
        temp_air = np.sum(air[:, 0:slices, :, :], axis=1)

        corrected_data = np.log(np.divide(temp_air, temp_data))

        for idx, b in enumerate(np.array([6, 7, 8, 9, 10, 12])):
            CNR_pts[idx, i], CNR_err[idx, i] = sct.cnr(corrected_data[b], contrast_mask, bg_mask)

    data = np.sum(data, axis=1)
    air = np.sum(air, axis=1)

    for j in np.arange(5, len(time_pts)):
        air = np.add(air, air)
        add_data = np.load(directory + folder + '/Raw Data/Run' + '{:03d}'.format(j-3) + '_a0.npy')
        add_data = np.squeeze(add_data)
        add_data = np.sum(add_data, axis=1)
        data = np.add(data, add_data)

        corrected_data = np.log(np.divide(air, data))

        for idx, b in enumerate(np.array([6, 7, 8, 9, 10, 12])):
            CNR_pts[idx, j], CNR_err[idx, j] = sct.cnr(corrected_data[b], contrast_mask, bg_mask)

    return time_pts, CNR_pts, CNR_err


def get_CNR_over_time_data_corrected_10s(folder, directory='C:/Users/10376/Documents/Phantom Data/Uniformity/'):

    contrast_mask = np.load(directory + folder + '/a0_Mask.npy')
    bg_mask = np.load(directory + folder + '/a0_Background.npy')

    time_pts = np.arange(0.001, 10.001, 0.001)  # Time points from 0.001 s to 10 s by 0.001 s increments
    CNR_pts = np.zeros([6, len(time_pts)])
    CNR_err = np.zeros([6, len(time_pts)])

    total_data = np.zeros([6, 24, 36])
    #random_order = np.array([10, 5, 6, 3, 9, 1, 4, 2, 8, 7])
    random_order = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
    for i in np.arange(1, 11):
        nn = random_order[i-1]
        add_data = np.load(directory + folder + '/Corrected Data/Run' + '{:03d}'.format(nn) + '_a0.npy')
        add_data = np.squeeze(add_data)
        add_data = add_data[6:11]

        temp_data = np.zeros([6, 24, 36])
        for j in np.arange(1000):
            single_frame = add_data[:, j]
            temp_data[0:5] = np.add(temp_data[0:5], single_frame)
            sumcc_single_frame = np.sum(single_frame, axis=0)
            temp_data[5] = np.add(temp_data[5], sumcc_single_frame)

            total_data = np.add(total_data, temp_data)

            for k, img in enumerate(total_data):
                CNR_pts[k, (i-1)*1000+j], blah = sct.cnr(img, contrast_mask, bg_mask)

    return time_pts, CNR_pts


def plot_CNR_over_time_10s(time_pts, CNR_pts, CNR_err=[], title='n/a', save=False,
                       directory='C:/Users/10376/Documents/Phantom Data/Uniformity/'):

    sns.set_style('whitegrid')
    fig, axes = plt.subplots(2, 3, figsize=(8, 6))
    ax1 = fig.add_subplot(111, frameon=False)
    ax1.grid(False)
    ax1.set_xticks([])
    ax1.set_yticks([])
    titles = ['20-30 keV', '30-50 keV', '50-70 keV', '70-90 keV', '90-120 keV', 'Sum CC']

    max_CNR = np.max(CNR_pts) + 0.25
    for i, ax in enumerate(axes.flat):
        ax.plot(time_pts, CNR_pts[i], lw=1)
        ax.set_title(titles[i])
        ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        ax.set_ylim([0, max_CNR])

    plt.subplots_adjust(left=0.12, bottom=0.11, right=0.96, top=0.88, wspace=0.31, hspace=0.44)
    ax1.set_xlabel('Acquisition Time (s)', fontsize=14, labelpad=25)
    ax1.set_ylabel('CNR', fontsize=14, labelpad=30)
    ax1.set_title(title, fontsize=15, pad=25)
    plt.show()

    if save:
        plt.savefig(directory + '/Plots/CNR_' + title + '.png', dpi=fig.dpi)
    plt.close()


def add_adj_bins(data, bins):
    """
    This function takes the adjacent bins given in bins and sums along the bin axis, can sum multiple bins
    :param data: 4D numpy array
                Data array with shape <counters, views, rows, columns
    :param bins: 1D array
                Bin numbers (as python indices, i.e the 1st bin would be 0) to sum
                Form: [Starting bin, Ending bin]
                Ex. for the 2nd through 5th bins, bins = [1, 4]
    :return: The summed data with the summed bins added together and the rest of the data intact
                shape <counters, views, rows, columns>
    """
    data_shape = np.array(np.shape(data))
    data_shape[0] = data_shape[0] - (bins[1] - bins[0])  # The new data will have the number of added bins - 1 new counters
    new_data = np.zeros(data_shape)

    new_data[0:bins[0]] = data[0:bins[0]]
    new_data[bins[0]] = np.sum(data[bins[0]:bins[-1]+1], axis=0)
    new_data[bins[0]+1:] = data[bins[1]+1:]

    return new_data


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


def get_CNR_over_time_data_corrected_1sec(folder, corr3x3=False, CC=True,
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



def plot_CNR_over_time_1s_multiple(time_pts, CNR_pts, CNR_err_mean, CC='CC', title='n/a', save=False,
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
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_ylim([0, max_CNR])
        ax.set_xlabel('CNR error at 0.25 s is \n' + '{:.2f}'.format(CNR_err_mean[i]))

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
        ax.plot(time_pts, CNR_pts[i], lw=1)
        ax.set_title(plottitles[i])
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_ylim([0, max_CNR])

    plt.subplots_adjust(left=0.12, bottom=0.2, right=0.96, top=0.88, wspace=0.31, hspace=0.55)
    ax1.set_xlabel('Acquisition Time (s)', fontsize=14, labelpad=45)
    ax1.set_ylabel('CNR', fontsize=14, labelpad=30)
    ax1.set_title(title, fontsize=15, pad=25)
    plt.show()

    if save:
        plt.savefig(directory + '/Plots/CNR ' + title + '.png', dpi=fig.dpi)
        plt.close()


t = 0
types = ['CC', 'SEC']
cc = 0
correction = ['330 um', '1 mm']

title = ['Bluebelt in Acrylic 1w ' + types[t] + ' ' + correction[cc], 'Bluebelt in Acrylic 4w ' + types[t] + ' ' + correction[cc],
         'Bluebelt in Fat 1w ' + types[t] + ' ' + correction[cc], 'Bluebelt in Fat 4w ' + types[t] + ' ' + correction[cc],
         'Bluebelt in Solid Water 1w ' + types[t] + ' ' + correction[cc], 'Bluebelt in Solid Water 4w ' + types[t] + ' ' + correction[cc],
         'Polypropylene in Acrylic 1w ' + types[t] + ' ' + correction[cc], 'Polypropylene in Acrylic 4w ' + types[t] + ' ' + correction[cc]]
bintitles = ['20-30 keV', '30-90 keV', '90-120 keV', 'Sum ' + types[t]]
bins = [1, 3]
#for nnn in np.arange(8):
nnn = 6
t, c = get_CNR_over_1s_sum_adj_bin(folders[nnn], bins, corr3x3=False, CC=True)
plot_CNR_adj_bins(t, c, bintitles, title=title[nnn], save=False)
#print(title[nnn])
#print()


#%%
#folders = ['m20358_q20_al_bluebelt_acryl_1w', 'm20358_q20_al_bluebelt_acryl_4w',
#           'm20358_q20_al_bluebelt_fat_1w', 'm20358_q20_al_bluebelt_fat_4w',
#           'm20358_q20_al_bluebelt_solidwater_1w', 'm20358_q20_al_bluebelt_solidwater_4w',
#           'm20358_q20_al_polypropylene_1w', 'm20358_q20_al_polypropylene_4w']
#plot_CNR_over_time(t, c)
#t, c, ce = get_CNR_over_time_data_raw(folders[6], air_folders[0])
#plot_CNR_over_time(t, c)

