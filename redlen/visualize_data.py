import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

        # Temporary
        #full_frames = full_frames[1:]
        #cnr = cnr[1:, :, :]
        #frames = frames[1:]
        #plot_cnr = plot_cnr[1:, :, :]

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

    ax1 = fig.add_subplot(111, frameon=False)
    ax1.grid(False)
    # Hide axes ticks
    ax1.set_xticks([])
    ax1.set_yticks([])
    blackpatch = mpatches.Patch(color=colors[0], label=ws[0])
    redpatch = mpatches.Patch(color=colors[1], label=ws[1])
    bluepatch = mpatches.Patch(color=colors[2], label=ws[2])
    leg = plt.legend(handles=[blackpatch, redpatch, bluepatch], loc='lower center', bbox_to_anchor=(0.5, -0.19), ncol=3,
                     fancybox=True)
    ax1.add_artist(leg)
    plt.subplots_adjust(hspace=0.45, bottom=0.17)

    if save:
        plt.savefig(path + 'TestNum_' + str(test_num) + '_' + CC + '.png', dpi=fig.dpi)
        plt.close()
    else:
        plt.show()


def noise_vs_time(folder, test_num, pixel, time, titles, ws=['1w', '3w', '8w'], CC='CC', save=False,
                directory='C:/Users/10376/Documents/Phantom Data/'):

    colors = ['black', 'red', 'blue', 'green']
    path = directory + folder + '/'
    w_folders = glob(path + '*w*/')  # Get each of the different w folders

    gof.create_folder('Plots', path) # Create the folder
    path = path + 'Plots/'
    gof.create_folder('Noise vs Time', path)
    path = path + 'Noise vs Time/'

    fig, axes = plt.subplots(2, 3, figsize=(8, 6), sharey=True)
    pixel_path = str(pixel) + 'x' + str(pixel) + ' Data/'
    gof.create_folder(pixel_path, path)
    path = path + pixel_path

    for c, fold in enumerate(w_folders):
        noise_path = fold + pixel_path + '/*' + str(test_num) + '_Avg_noise.npy'
        frames_path = fold + pixel_path + '/*' + str(test_num) + '_Frames.npy'

        noise = np.load(glob(noise_path)[0])
        frames = np.load(glob(frames_path)[0])
        full_frames = frames

        stop_idx = int(np.argwhere(frames == time) + 1)
        frames = frames[0:stop_idx]

        plot_noise = np.zeros([len(frames), 6, 2])
        if CC == 'CC':
            plot_noise[:, 0:5] = noise[0:stop_idx, 6:11]  # Get only CC data up to the frame desired
        else:
            plot_noise[:, 0:5] = noise[0:stop_idx, 0:5]  # Get only SEC data up to the frame desired
        plot_noise[:, 5] = noise[0:stop_idx, 12]  # Get EC bin for summed bin
        plot_noise = np.abs(plot_noise)
        noise = np.abs(noise)
        for i, ax in enumerate(axes.flatten()):
            # Make some smooth data
            xnew = np.linspace(0, 1000, 1000)
            if i == 5:
                spl = spline(full_frames, noise[:, 12, 0])
            elif CC == 'CC':
                spl = spline(full_frames, noise[:, i+6, 0])
            else:
                spl = spline(full_frames, noise[:, i, 0])
            cnr_smth = spl(xnew)

            ax.plot(xnew, cnr_smth, color=colors[c])
            #ax.errorbar(frames, plot_noise[:, i, 0], yerr=plot_noise[:, i, 1], fmt='none', color=colors[c])
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Noise')
            ax.set_xlim([0, time])
            if i == 5:
                ax.set_title(titles[i] + ' Bin')
            else:
                ax.set_title(titles[i] + ' keV ' + CC)

    ax1 = fig.add_subplot(111, frameon=False)
    ax1.grid(False)
    # Hide axes ticks
    ax1.set_xticks([])
    ax1.set_yticks([])
    blackpatch = mpatches.Patch(color=colors[0], label=ws[0])
    redpatch = mpatches.Patch(color=colors[1], label=ws[1])
    bluepatch = mpatches.Patch(color=colors[2], label=ws[2])
    leg = plt.legend(handles=[blackpatch, redpatch, bluepatch], loc='lower center', bbox_to_anchor=(0.5, -0.19), ncol=3,
                     fancybox=True)
    ax1.add_artist(leg)
    plt.subplots_adjust(hspace=0.45, bottom=0.17)

    if save:
        plt.savefig(path + 'TestNum_' + str(test_num) + '_' + CC + '.png', dpi=fig.dpi)
        plt.close()
    else:
        plt.show()


def contrast_vs_time(folder, test_num, pixel, time, titles, ws=['1w', '3w', '8w'], CC='CC', save=False,
                directory='C:/Users/10376/Documents/Phantom Data/'):

    colors = ['black', 'red', 'blue', 'green']
    path = directory + folder + '/'
    w_folders = glob(path + '*w*/')  # Get each of the different w folders

    gof.create_folder('Plots', path) # Create the folder
    path = path + 'Plots/'
    gof.create_folder('Contrast vs Time', path)
    path = path + 'Contrast vs Time/'

    fig, axes = plt.subplots(2, 3, figsize=(8, 6), sharey=True)
    pixel_path = str(pixel) + 'x' + str(pixel) + ' Data/'
    gof.create_folder(pixel_path, path)
    path = path + pixel_path

    for c, fold in enumerate(w_folders):
        contrast_path = fold + pixel_path + '/*' + str(test_num) + '_Avg_contrast.npy'
        frames_path = fold + pixel_path + '/*' + str(test_num) + '_Frames.npy'

        contrast = np.load(glob(contrast_path)[0])
        frames = np.load(glob(frames_path)[0])
        full_frames = frames

        stop_idx = int(np.argwhere(frames == time) + 1)
        frames = frames[0:stop_idx]

        plot_contrast = np.zeros([len(frames), 6, 2])
        if CC == 'CC':
            plot_contrast[:, 0:5] = contrast[0:stop_idx, 6:11]  # Get only CC data up to the frame desired
        else:
            plot_contrast[:, 0:5] = contrast[0:stop_idx, 0:5]  # Get only SEC data up to the frame desired
        plot_contrast[:, 5] = contrast[0:stop_idx, 12]  # Get EC bin for summed bin
        contrast = np.abs(contrast)
        for i, ax in enumerate(axes.flatten()):
            # Make some smooth data
            xnew = np.linspace(0, 1000, 1000)
            if i == 5:
                spl = spline(full_frames, contrast[:, 12, 0])
            elif CC == 'CC':
                spl = spline(full_frames, contrast[:, i+6, 0])
            else:
                spl = spline(full_frames, contrast[:, i, 0])
            contrast_smth = spl(xnew)

            ax.plot(xnew, contrast_smth, color=colors[c])
            #ax.errorbar(frames, plot_contrast[:, i, 0], yerr=plot_contrast[:, i, 1], fmt='none', color=colors[c])
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Contrast')
            ax.set_xlim([0, time])
            if i == 5:
                ax.set_title(titles[i] + ' Bin')
            else:
                ax.set_title(titles[i] + ' keV ' + CC)

    ax1 = fig.add_subplot(111, frameon=False)
    ax1.grid(False)
    # Hide axes ticks
    ax1.set_xticks([])
    ax1.set_yticks([])
    blackpatch = mpatches.Patch(color=colors[0], label=ws[0])
    redpatch = mpatches.Patch(color=colors[1], label=ws[1])
    bluepatch = mpatches.Patch(color=colors[2], label=ws[2])
    leg = plt.legend(handles=[blackpatch, redpatch, bluepatch], loc='lower center', bbox_to_anchor=(0.5, -0.19), ncol=3,
                     fancybox=True)
    ax1.add_artist(leg)
    plt.subplots_adjust(hspace=0.45, bottom=0.17)

    if save:
        plt.savefig(path + 'TestNum_' + str(test_num) + '_' + CC + '.png', dpi=fig.dpi)
        plt.close()
    else:
        plt.show()


def cnr_vs_pixels(folder, test_num, time, titles, ws=['1w', '3w', '8w'], CC='CC', save=False,
                directory='C:/Users/10376/Documents/Phantom Data/'):

    colors = ['black', 'red', 'blue', 'green']
    path = directory + folder + '/'
    w_folders = glob(path + '*w*/')  # Get each of the different w folders

    gof.create_folder('Plots', path) # Create the folder
    path = path + 'Plots/'
    gof.create_folder('CNR vs Pixels', path)
    path = path + 'CNR vs Pixels/'

    fig, axes = plt.subplots(2, 3, figsize=(8, 6), sharey=True)

    for c, fold in enumerate(w_folders):
        cnr_path = fold + 'CNR vs Pixel Data/Test_Num_' + str(test_num) + '_Frame_' + str(time) + '.npy'
        pixels_path = fold + 'CNR vs Pixel Data/Pixels_x-axis.npy'

        cnr = np.load(cnr_path)
        pixels = np.load(pixels_path)

        ten_pix = np.argwhere(pixels == 10)

        plot_cnr = np.zeros([6, 2, len(pixels)])
        if CC == 'CC':
            plot_cnr[0:5] = cnr[6:11]  # Get only CC data up to the frame desired
        else:
            plot_cnr[0:5] = cnr[0:5]  # Get only SEC data up to the frame desired
        plot_cnr[5] = cnr[12]  # Get EC bin for summed bin

        pixels = np.delete(pixels, ten_pix)
        for i, ax in enumerate(axes.flatten()):
            ax.errorbar(pixels, np.delete(plot_cnr[i, 0], ten_pix), yerr=np.delete(plot_cnr[i, 1], ten_pix), fmt='none', color=colors[c], capsize=4)
            ax.set_xlabel('Pixel Number')
            ax.set_ylabel('CNR')
            ax.set_xlim([0, np.max(pixels)+1])
            if c == 0:
                ax.set_ylim([0, np.max(plot_cnr[:, 0, 0:6]+5)])
            if i == 5:
                ax.set_title(titles[i] + ' Bin')
            else:
                ax.set_title(titles[i] + ' keV ' + CC)

    ax1 = fig.add_subplot(111, frameon=False)
    ax1.grid(False)
    # Hide axes ticks
    ax1.set_xticks([])
    ax1.set_yticks([])
    blackpatch = mpatches.Patch(color=colors[0], label=ws[0])
    redpatch = mpatches.Patch(color=colors[1], label=ws[1])
    bluepatch = mpatches.Patch(color=colors[2], label=ws[2])
    leg = plt.legend(handles=[blackpatch, redpatch, bluepatch], loc='lower center', bbox_to_anchor=(0.5, -0.19), ncol=3,
                     fancybox=True)
    ax1.add_artist(leg)
    plt.subplots_adjust(hspace=0.45, bottom=0.17)

    if save:
        plt.savefig(path + 'TestNum_' + str(test_num) + '_Time_' + str(time) + '_' + CC + '.png', dpi=fig.dpi)
        plt.close()
    else:
        plt.show()


def noise_vs_pixels(folder, test_num, time, titles, ws=['1w', '3w', '8w'], CC='CC', save=False,
                directory='C:/Users/10376/Documents/Phantom Data/'):

    colors = ['black', 'red', 'blue', 'green']
    path = directory + folder + '/'
    w_folders = glob(path + '*w*/')  # Get each of the different w folders

    gof.create_folder('Plots', path) # Create the folder
    path = path + 'Plots/'
    gof.create_folder('Noise vs Pixels', path)
    path = path + 'Noise vs Pixels/'

    fig, axes = plt.subplots(2, 3, figsize=(8, 6), sharey=True)

    for c, fold in enumerate(w_folders):
        noise_path = fold + 'Noise vs Pixel Data/Test_Num_' + str(test_num) + '_Frame_' + str(time) + '.npy'
        pixels_path = fold + 'Noise vs Pixel Data/Pixels_x-axis.npy'

        noise = np.load(noise_path)
        pixels = np.load(pixels_path)
        ten_pix = np.argwhere(pixels == 10)

        plot_noise = np.zeros([6, 2, len(pixels)])

        if CC == 'CC':
            plot_noise[0:5] = noise[6:11]  # Get only CC data up to the frame desired
        else:
            plot_noise[0:5] = noise[0:5]  # Get only SEC data up to the frame desired
        plot_noise[5] = noise[12]  # Get EC bin for summed bin

        pixels = np.delete(pixels, ten_pix)
        for i, ax in enumerate(axes.flatten()):
            ax.errorbar(pixels, np.delete(plot_noise[i, 0], ten_pix), np.delete(plot_noise[i, 1], ten_pix), fmt='none', color=colors[c], capsize=4)
            ax.set_xlabel('Pixel Number')
            ax.set_ylabel('Noise')
            ax.set_xlim([0, np.max(pixels)+1])
            #ax.set_ylim([0, 10])
            if i == 5:
                ax.set_title(titles[i] + ' Bin')
            else:
                ax.set_title(titles[i] + ' keV ' + CC)

    ax1 = fig.add_subplot(111, frameon=False)
    ax1.grid(False)
    # Hide axes ticks
    ax1.set_xticks([])
    ax1.set_yticks([])
    blackpatch = mpatches.Patch(color=colors[0], label=ws[0])
    redpatch = mpatches.Patch(color=colors[1], label=ws[1])
    bluepatch = mpatches.Patch(color=colors[2], label=ws[2])
    leg = plt.legend(handles=[blackpatch, redpatch, bluepatch], loc='lower center', bbox_to_anchor=(0.5, -0.19), ncol=3,
                     fancybox=True)
    ax1.add_artist(leg)
    plt.subplots_adjust(hspace=0.45, bottom=0.17)

    if save:
        plt.savefig(path + 'TestNum_' + str(test_num) + '_Time_' + str(time) + '_' + CC + '.png', dpi=fig.dpi)
        plt.close()
    else:
        plt.show()


def see_all_w_images(folder, test_num, pixel, bin_num, ws=['1w', '3w', '8w'],
                     directory='C:/Users/10376/Documents/Phantom Data/', save=False):

    path = directory + folder + '/'
    pixel_path = str(pixel) + 'x' + str(pixel) + ' Data/'
    w_folders = glob(path + '*w*/')  # Get each of the different w folders

    fig, axes = plt.subplots(1, len(w_folders), figsize=(8, 4))

    for c, fold in enumerate(w_folders):
        datapath = fold + '/' + pixel_path + 'Raw Data/TestNumData_a0_' + str(test_num) + '.npy'
        airpath = fold + '/' + pixel_path + 'Raw Data/TestNumAir_a0_' + str(test_num) + '.npy'

        data = np.squeeze(np.load(datapath))
        air = np.squeeze(np.load(airpath))

        data = np.sum(data, axis=1)
        air = np.sum(air, axis=1)

        corr = -1*np.log(np.divide(data, air))

        axes[c].imshow(corr[bin_num])
        #axes[c].imshow(np.load(fold + '/' + pixel_path + 'a0_Mask.npy'), alpha=0.3)
        #axes[c].imshow(np.load(fold + '/' + pixel_path + 'a0_Background.npy'), alpha=0.3)
        axes[c].set_title(ws[c])
        bg = np.load(fold + '/' + pixel_path + 'a0_Background.npy')
        print(np.mean(corr[bin_num]))
    gof.create_folder('Images', path)
    plt.show()
    if save:
        plt.savefig(path + 'Images/TestNum_' + str(test_num) + '_Bin_' + str(bin_num) + '_Pixel_' + str(pixel) + '.png', dpi=fig.dpi, transparent=True)
        plt.close()


titles_all = [['20-30', '30-50', '50-70', '70-90', '90-120', 'EC'],
              ['20-50', '50-70', '70-90', '90-100', '100-120', 'EC'],
              ['20-55', '55-70', '70-90', '90-100', '100-120', 'EC'],
              ['20-60', '60-75', '75-90', '90-100', '100-120', 'EC'],
              ['20-65', '65-80', '80-90', '90-100', '100-120', 'EC'],
              ['20-70', '70-80', '80-90', '90-100', '100-120', 'EC'],
              ['20-80', '80-90', '90-100', '100-110', '110-120', 'EC'],
              ['20-90', '90-100', '100-105', '105-110', '110-120', 'EC']]
folder = 'Polypropylene_06-12-20'
#for folder in ['Polypropylene_06-12-20']:#, 'Small_bluebelt_06-12-20']:
for i in np.arange(1, 2):
    #for j in [1, 2, 3, 4, 6]:
    #cnr_vs_time(folder, i, 1, 1000, titles_all[i-1], CC='SEC')#, save=True)
    #cnr_vs_time(folder, i, 1, 1000, titles_all[i-1], CC='CC')#, save=True)

    noise_vs_time(folder, i, 1, 100, titles_all[i-1], CC='SEC', save=True)
    noise_vs_time(folder, i, 1, 100, titles_all[i-1], CC='CC', save=True)

    #for k in [2, 10, 250]:
    #    cnr_vs_pixels(folder, i, k, titles_all[i-1], CC='SEC', save=True)
    #    cnr_vs_pixels(folder, i, k, titles_all[i - 1], CC='CC', save=True)

        #    noise_vs_pixels(folder, i, k, titles_all[i - 1], CC='SEC', save=True)
        #    noise_vs_pixels(folder, i, k, titles_all[i - 1], CC='CC', save=True)

        #for b in np.arange(12, 13):
        #    see_all_w_images(folder, 1, j, b, save=True)