import numpy as np
import matplotlib.pyplot as plt
import glob
import general_OS_functions as gof
import sCT_Analysis as sct

folders = ['m20358_q20_al_bluebelt_acryl_1w', 'm20358_q20_al_bluebelt_acryl_4w',
           'm20358_q20_al_bluebelt_fat_1w', 'm20358_q20_al_bluebelt_fat_4w',
           'm20358_q20_al_bluebelt_solidwater_1w', 'm20358_q20_al_bluebelt_solidwater_4w',
           'm20358_q20_al_polypropylene_1w', 'm20358_q20_al_polypropylene_4w']

air_folders = ['m20358_q20_al_air_1w', 'm20358_q20_al_air_4w']

a1_pixels = np.array([[0, 0], [0, 2], [0, 3], [23, 17], [16, 32], [13, 32], [8, 35], [11, 35], [12, 35], [14, 35], [17, 35]])


def get_CNR_over_time_data(folder, air_folder, directory='C:/Users/10376/Documents/Phantom Data/Uniformity/'):

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


def plot_CNR_over_time(time_pts, CNR_pts, CNR_err, title='n/a', save=False):

    fig, axes = plt.subplots(2, 3, figsize=(8, 6))
    ax1 = fig.add_subplot(111, frameon=False)
    ax1.grid(False)
    ax1.set_xticks([])
    ax1.set_yticks([])
    titles = ['20-30 keV', '30-50 keV', '50-70 keV', '70-90 keV', '90-120 keV', '20-120 keV']

    for i, ax in enumerate(axes.flat):
        ax.scatter(time_pts, CNR_pts[i], lw=1)
        ax.set_title(titles[i])

    plt.subplots_adjust(left=0.12, bottom=0.11, right=0.96, top=0.88, wspace=0.31, hspace=0.44)
    ax1.set_xlabel('Acquisition Time (s)', fontsize=14, labelpad=25)
    ax1.set_ylabel('CNR', fontsize=14, labelpad=30)
    ax1.set_title(title, fontsize=15, pad=25)
    plt.show()


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


#t, c, ce = get_CNR_over_time_data(folders[6], air_folders[0])
#plot_CNR_over_time(t, c, ce)

#%%