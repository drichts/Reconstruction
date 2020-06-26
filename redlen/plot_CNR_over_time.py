import numpy as np
import sct_analysis as sct
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from datetime import datetime as dt


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


def get_CNR_over_time_shuffle(folder, threshold, pxp=1, CC=True, frames=1000,
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
    CNR_pts = np.zeros([20, 6, frames])  # Collect CNR over 1 s for all 20 iterations
    CNR_err = np.zeros([20, 6, frames])  # Collect CNR error over 1 s for all 20 iterations

    for i in np.arange(20):

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

        # Grab the number of frames desired starting from a random number
        start = int((1000-frames-1)*np.random.rand())
        add_data = add_data[:, start:start+frames]

        if CC:
            add_data = add_data[6:12]  # Grab just cc (or sec) bins
        else:
            add_data = add_data[0:6]  # Grab just sec bins

        # Grab the views in a random order
        shuffle_order = np.arange(frames)
        np.random.shuffle(shuffle_order)

        for j, sl in enumerate(shuffle_order):
            single_frame = add_data[:, sl]  # Get the next view data
            total_data[0:5] = np.add(total_data[0:5], single_frame[0:5])  # Add to the current total data
            sum_single_frame = np.sum(single_frame, axis=0)  # Sum all bins to get summed cc (or sec)
            total_data[5] = np.add(total_data[5], sum_single_frame)  # Add to the total summed

            for k, img in enumerate(total_data):
                # Calculate the CNR (i-1 = file, k = bin, j = view/time point)
                CNR_pts[i, k, j], CNR_err[i, k, j] = sct.cnr(img, contrast_mask, bg_mask)

    CNR_pts = np.mean(CNR_pts, axis=0)
    CNR_err = np.mean(CNR_err, axis=0)

    return time_pts, CNR_pts, CNR_err


def get_CNR_over_time_pixel_agg(pxp=[1, 2, 3, 4, 6, 8, 12], folders=['3w/'], CC=True,
                            directory='C:/Users/10376/Documents/Phantom Data/Uniformity/Multiple Energy Thresholds/'):

    for folder in folders:
        for pix in pxp:
            if pix == 1:
                save_path = directory + folder + 'Data/'
            else:
                save_path = directory + folder + str(pix) + 'x' + str(pix) + ' Data/'

            t, c, ce = get_CNR_over_time_shuffle(folder, 1, pxp=pix, CC=CC, frames=300, directory=directory)
            np.save(save_path + 'Thresh1_CNR_over_time_20randmean.npy', c)

            if folder == '3w/':
                plot_CNR_over_time_1s(t, c, ce, title=str(pix) + 'x' + str(pix) + ' 20 random mean', save=True,
                                      directory=directory)
