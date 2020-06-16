import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import general_OS_functions as gof
import sCT_Analysis as sct
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from datetime import datetime as dt
from scipy.interpolate import make_interp_spline as spline


def cnr_vs_time(folder, test_num, pixel, time, titles, ws=['1w', '3w', '8w'], CC='CC', save=False,
                directory='C:/Users/10376/Documents/Phantom Data/'):

    colors = ['black', 'red', 'blue', 'green']
    path = directory + folder + '/'
    w_folders = glob(path + '*w*/')  # Get each of the different w folders

    gof.create_folder('Plots', path) # Create the folder
    path = path + 'Plots/'
    gof.create_folder('CNR vs Time', path)
    path = path + 'CNR vs Time/'

    fig, axes = plt.subplots(2, 3, figsize=(8, 6), sharey=True)
    pixel_path = str(pixel) + 'x' + str(pixel) + ' Data/'
    gof.create_folder(pixel_path, path)
    path = path + pixel_path

    for c, fold in enumerate(w_folders):
        cnr_path = fold + pixel_path + '/*' + str(test_num) + '_Avg_CNR.npy'
        frames_path = fold + pixel_path + '/*' + str(test_num) + '_Frames.npy'

        cnr = np.load(glob(cnr_path)[0])
        frames = np.load(glob(frames_path)[0])
        full_frames = frames

        stop_idx = int(np.argwhere(frames == time) + 1)
        frames = frames[0:stop_idx]

        plot_cnr = np.zeros([len(frames), 6, 2])
        if CC == 'CC':
            plot_cnr[:, 0:5] = cnr[0:stop_idx, 6:11]  # Get only CC data up to the frame desired
        else:
            plot_cnr[:, 0:5] = cnr[0:stop_idx, 0:5]  # Get only SEC data up to the frame desired
        plot_cnr[:, 5] = cnr[0:stop_idx, 12]  # Get EC bin for summed bin

        for i, ax in enumerate(axes.flatten()):
            # Make some smooth data
            xnew = np.linspace(0, 1000, 1000)
            if i == 5:
                spl = spline(full_frames, cnr[:, 12, 0])
            elif CC == 'CC':
                spl = spline(full_frames, cnr[:, i+6, 0])
            else:
                spl = spline(full_frames, cnr[:, i, 0])
            cnr_smth = spl(xnew)

            ax.plot(xnew, cnr_smth, color=colors[c])
            #ax.errorbar(frames, plot_cnr[:, i, 0], yerr=plot_cnr[:, i, 1], fmt='none', color=colors[c])
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('CNR')
            ax.set_xlim([0, time])
            if i == 5:
                ax.set_title(titles[i] + ' Bin')
            else:
                ax.set_title(titles[i] + ' keV ' + CC)

    for i, ax in enumerate(axes.flatten()):
        ax.legend(ws)

    plt.subplots_adjust(hspace=0.4)
    plt.show()

    if save:
        plt.savefig(path + 'TestNum_' + str(test_num) + '.png', dpi=fig.dpi)
        plt.close()

titles_all = [['20-30', '30-50', '50-70', '70-90', '90-120', 'EC'],
              ['20-50', '50-70', '70-90', '90-100', '100-120', 'EC'],
              ['20-55', '55-70', '70-90', '90-100', '100-120', 'EC'],
              ['20-60', '60-75', '75-90', '90-100', '100-120', 'EC'],
              ['20-65', '65-80', '80-90', '90-100', '100-120', 'EC'],
              ['20-70', '70-80', '80-90', '90-100', '100-120', 'EC'],
              ['20-80', '80-90', '90-100', '100-110', '110-120', 'EC'],
              ['20-90', '90-100', '100-105', '105-110', '110-120', 'EC']]

for i in np.arange(1, 9):
    for j in [1, 2, 3, 4, 6, 8, 10, 12]:
        cnr_vs_time('Polypropylene_06-12-20', i, j, 250, titles_all[i-1], save=True)