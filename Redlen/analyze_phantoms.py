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

a1_pixels = np.array([[0, 0], [0, 2], [0, 3], [23, 17], [16, 32], [13, 32], [8, 35], [11, 35], [12, 35], [14, 35], [17, 35]])


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


def correct_dead_pixels(img, pixels):
    """
    Correct for non-responsive and anomalous pixels
    :param img: The image to be corrected as a 2D numpy array
    :param pixels: The dead pixels as an array of tuples
    :return img: The corrected image
    """
    for pixel in pixels:
        avg = get_average_pixel_value(img, pixel)
        img[pixel[0], pixel[1]] = avg

    return img


def get_average_pixel_value(img, pixel):
    """
    Averages the dead pixel using the 8 nearest neighbours
    :param img: the projection image
    :param pixel: the problem pixel (is a 2-tuple)
    :return:
    """

    shape = np.shape(img)
    row, col = pixel

    if col == shape[1]-1:
        n1 = np.nan
    else:
        n1 = img[row, col+1]
    if col == 0:
        n2 = np.nan
    else:
        n2 = img[row, col-1]
    if row == shape[0]-1:
        n3 = np.nan
    else:
        n3 = img[row+1, col]
    if row == 0:
        n4 = np.nan
    else:
        n4 = img[row-1, col]
    if col == shape[1]-1 or row == shape[0]-1:
        n5 = np.nan
    else:
        n5 = img[row+1, col+1]
    if col == 0 or row == shape[0]-1:
        n6 = np.nan
    else:
        n6 = img[row+1, col-1]
    if col == shape[1]-1 or row == 0:
        n7 = np.nan
    else:
        n7 = img[row-1, col+1]
    if col == 0 or row == 0:
        n8 = np.nan
    else:
        n8 = img[row-1, col-1]

    avg = np.nanmean(np.array([n1, n2, n3, n4, n5, n6, n7, n8]))

    return avg

t = 0
types = ['CC', 'SEC']
cc = 1
correction = ['330 um', '1 mm']

title = ['Bluebelt in Acrylic 1w ' + types[t] + ' ' + correction[cc], 'Bluebelt in Acrylic 4w ' + types[t] + ' ' + correction[cc],
         'Bluebelt in Fat 1w ' + types[t] + ' ' + correction[cc], 'Bluebelt in Fat 4w ' + types[t] + ' ' + correction[cc],
         'Bluebelt in Solid Water 1w ' + types[t] + ' ' + correction[cc], 'Bluebelt in Solid Water 4w ' + types[t] + ' ' + correction[cc],
         'Polypropylene in Acrylic 1w ' + types[t] + ' ' + correction[cc], 'Polypropylene in Acrylic 4w ' + types[t] + ' ' + correction[cc]]
#for nnn in np.arange(8):
nnn = 0
t, c, cem, ces = get_CNR_over_time_data_corrected_1sec(folders[nnn], corr3x3=True, CC=True)
plot_CNR_over_time_1s_multiple(t, c, CNR_err_mean=cem, CC='CC', title=title[nnn], save=False)
print(title[nnn])
print()


#%%
#folders = ['m20358_q20_al_bluebelt_acryl_1w', 'm20358_q20_al_bluebelt_acryl_4w',
#           'm20358_q20_al_bluebelt_fat_1w', 'm20358_q20_al_bluebelt_fat_4w',
#           'm20358_q20_al_bluebelt_solidwater_1w', 'm20358_q20_al_bluebelt_solidwater_4w',
#           'm20358_q20_al_polypropylene_1w', 'm20358_q20_al_polypropylene_4w']
#plot_CNR_over_time(t, c)
#t, c, ce = get_CNR_over_time_data_raw(folders[6], air_folders[0])
#plot_CNR_over_time(t, c)

