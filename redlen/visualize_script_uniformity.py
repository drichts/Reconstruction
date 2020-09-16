import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import seaborn as sns
import os
from scipy.interpolate import make_interp_spline as spline
from redlen.uniformity_analysis import AnalyzeUniformity
from redlen.uniformity_analysis_add_bins import AddBinsUniformity
from redlen.visualize import VisualizeUniformity
from redlen.visualize_add_bins import AddBinsVisualize
from redlen.visualize_3windows import Visualize3Windows
from general_functions import load_object

titles_mult = [['20-30', '30-50', '50-70', '70-90', '90-120', 'TC'],
              ['20-30', '30-40', '40-50', '50-60', '60-70', 'TC'],
              ['20-35', '35-50', '50-65', '65-80', '80-90', 'TC'],
              ['25-35', '35-45', '45-55', '55-65', '65-75', 'TC'],
              ['25-40', '40-55', '55-70', '70-80', '80-95', 'TC'],
              ['30-45', '45-60', '60-75', '75-85', '85-95', 'TC'],
              ['20-30', '30-70', '70-85', '85-100', '100-120', 'TC']]
              #['20-90', '90-100', '100-105', '105-110', '110-120', 'TC']]

titles_energy_check = [['20-30', '30-50', '50-70', '70-90', '90-120', 'TC'],
                       ['20-50', '50-70', '70-100', '100-110', '110-120', 'TC'],
                       ['20-40', '40-50', '50-70', '70-80', '80-120', 'TC']]

titles_many = [['20-30', '30-50', '50-70', '70-90', '90-120', 'TC'],
               ['20-40', '40-60', '60-80', '80-100', '100-120', 'TC'],
               ['25-45', '45-65', '65-85', '85-105', '105-120', 'TC'],
               ['20-35', '35-55', '55-75', '75-95', '95-120', 'TC'],
               ['20-50', '50-80', '80-100', '100-110', '110-120', 'TC'],
               ['20-30', '30-60', '60-90', '90-100', '100-120', 'TC'],
               ['20-40', '40-70', '70-100', '100-110', '110-120', 'TC'],
               ['20-60', '60-70', '70-80', '80-90', '90-120', 'TC'],
               ['20-70', '70-80', '80-90', '90-100', '100-120', 'TC'],
               ['20-80', '80-90', '90-100', '100-110', '110-120', 'TC'],
               ['20-90', '90-100', '100-105', '105-110', '110-120', 'TC'],
               ['20-30', '30-70', '70-80', '80-90', '90-120', 'TC'],
               ['20-30', '30-80', '80-90', '90-100', '100-120', 'TC'],
               ['20-30', '30-90', '90-100', '100-110', '110-120', 'TC'],
               ['20-40', '40-80', '80-90', '90-100', '100-120', 'TC'],
               ['20-40', '40-90', '90-100', '100-110', '110-120', 'TC'],
               ['20-50', '50-90', '90-100', '100-110', '110-120', 'TC']]

folders = ['many_thresholds_BB4mm', 'many_thresholds_BB2mm',
           'many_thresholds_glass2mm', 'many_thresholds_glass1mm',
           'many_thresholds_steel07mm', 'many_thresholds_steel2mm',
           'many_thresholds_PP']
airfolder = 'many_thresholds_airscan'

conts = ['4 mm TPU', '2 mm TPU', '2 mm glass', '1 mm glass', '0.7 mm steel', '2 mm steel', '3 mm polypropylene']

folders2 = ['many_thresholds_BB4mm', 'many_thresholds_BB2mm', 'many_thresholds_BB1mm', 'many_thresholds_glass1mm']

folders3 = ['multiple_energy_thresholds_1w', 'multiple_energy_thresholds_3w']
airfolders3 = ['multiple_energy_thresholds_flatfield_1w', 'multiple_energy_thresholds_flatfield_3w']

u_folders = ['NDT_BB4mm', 'NDT_BB2mm',
             'NDT_glass2mm',
             'NDT_steel07mm', 'NDT_steel2mm']
u_airfolder = 'NDT_airscan'
u_directory = r'C:\Users\10376\Documents\Phantom Data\UVic'


def smooth_data(xpts, ypts, cnr_or_noise):
    xsmth = np.linspace(xpts[0], xpts[-1], 1000)
    if cnr_or_noise == 2:  # NOISE
        coeffs = np.polyfit(xpts, ypts, 2)
        p = np.poly1d(coeffs)
        ysmth = p(xsmth)
    else:  # CNR
        p = spline(xpts, ypts)
        ysmth = p(xsmth)
    return xsmth, ysmth


def see_ROIs(folder_num, pixel):
    a1 = AnalyzeUniformity(folders[folder_num], airfolder)
    a1.analyze_cnr_noise()
    a1.mean_signal_all_pixels()

    pixel = np.squeeze(np.argwhere(a1.pxp == pixel))
    mask = a1.masks[pixel]
    bg = a1.bg[pixel]

    #plt.imshow()

    plt.imshow(mask)
    plt.show()
    plt.pause(2)
    plt.close()

    plt.imshow(bg)
    plt.show()
    plt.pause(2)
    plt.close()


def run_cnr_noise_time_pixels():
    for folder in folders:
        for i in np.arange(1, 2):
            #a1 = AnalyzeUniformity(folder, u_airfolder, test_num=i, mm='M15691')
            a1 = AnalyzeUniformity(folder, airfolder, test_num=i)
            a1.analyze_cnr_noise()
            a1.mean_signal_all_pixels()
            v1 = VisualizeUniformity(a1)
            v1.titles = titles_many[i - 1]

            v1.blank_vs_time_six_bins(cnr_or_noise=0, end_time=25, save=True)
            v1.blank_vs_time_six_bins(cnr_or_noise=1, end_time=25, save=True)

            if i == 1:
                #
                # v1.blank_vs_pixels_six_bins(cnr_or_noise=0, time=2, save=True)
                # v1.blank_vs_pixels_six_bins(cnr_or_noise=0, time=5, save=True)
                # v1.blank_vs_pixels_six_bins(cnr_or_noise=0, time=10, save=True)
                v1.blank_vs_pixels_six_bins(cnr_or_noise=0, time=25, save=True)

                # v1.blank_vs_pixels_six_bins(cnr_or_noise=1, time=2, save=True)
                # v1.blank_vs_pixels_six_bins(cnr_or_noise=1, time=5, save=True)
                # v1.blank_vs_pixels_six_bins(cnr_or_noise=1, time=10, save=True)
                v1.blank_vs_pixels_six_bins(cnr_or_noise=1, time=25, save=True)


def cnrbinning_multiple_one_plot(fs=None, save=False):
    sns.set_style('whitegrid')
    path = r'C:\Users\10376\Documents\Phantom Data\Report\CNR Binning/'
    colors = ['mediumblue', 'orangered', 'mediumseagreen']
    folds = folders
    cs = conts
    if fs:
        folds = []
        cs = []
        for f in fs:
            folds.append(folders[f])
            cs.append(conts[f])
    fig, axes = plt.subplots(2, 3, figsize=(8, 6), sharey='all')
    patches = []
    for j, folder in enumerate(folds):
        patches.append(mlines.Line2D([0], [0], color=colors[j], lw=2, linestyle='-', label=cs[j]))
        titles = titles_many[0]
        cnr_vals = np.load(rf'C:\Users\10376\Documents\Phantom Data\UNIFORMITY\{folder}\TestNum1_cnr_time.npy')  # <pixels, bin, val, time>

        pixels = np.array([1, 2, 3, 4, 6])

        # Grab only the pixel values at the 25 ms and rearrange to <bin, val, pixels> and cut out 12x12 data
        cnr_vals = np.transpose(cnr_vals[:, :, :, 12], axes=(1, 2, 0))

        # Cut out overflow bins
        plot_cnr = np.zeros([11, 2, len(pixels)])
        plot_cnr[0:5] = cnr_vals[0:5]  # Get only SEC data up to the frame desired
        plot_cnr[5:10] = cnr_vals[6:11]  # Get only CC data up to the frame desired
        plot_cnr[10] = cnr_vals[12]  # Get EC bin for summed bin

        # Make some smooth data
        # pc_shape = np.shape(plot_cnr)
        # cnr_smth = np.zeros([pc_shape[0], 1000])
        # for idx, ypts in enumerate(plot_cnr[:, 0]):
        #     pxs_smth, cnr_smth[idx] = smooth_data(pixels, ypts, cnr_or_noise=0)

        for i, ax in enumerate(axes.flatten()):
            ax.set_yscale('log')
            if i < 5:
                # ax.plot(pxs_smth, cnr_smth[i + 5], color='k')
                ax.errorbar(pixels, plot_cnr[i + 5, 0], yerr=plot_cnr[i + 5, 1], capsize=3, color=colors[j])
                if j == 2:
                    print(plot_cnr[i + 5, 1])
                ax.set_title(titles[i] + ' keV')
            else:
                # ax.plot(pxs_smth, cnr_smth[-1], color='k')
                ax.errorbar(pixels, plot_cnr[-1, 0], yerr=plot_cnr[-1, 1], capsize=3, color=colors[j])
                ax.set_title(titles[i])
            ax.set_xlim([0, pixels[-1] + 1])
            ax.set_xticks([2, 4, 6])
            ax.set_xticklabels([r'2$\times$2', r'4$\times$4', r'6$\times$6'])
            #ax.set_ylim([1E0, 2E3])

    fig.text(0.5, 0.09, 'Binning', ha='center', fontsize=14)
    fig.text(0.06, 0.52, 'CNR', va='center', rotation='vertical', fontsize=14)
    leg = plt.legend(handles=patches, loc='lower center', bbox_to_anchor=(-0.7, -0.55), fancybox=True, shadow=False,
                     ncol=len(fs), fontsize=12)
    fig.add_artist(leg)
    plt.subplots_adjust(hspace=0.45, bottom=0.17)
    plt.show()

    if save:
        plt.savefig(path + 'three.png', dpi=fig.dpi)
        plt.close()
    else:
        plt.pause(5)
        plt.close()


def noisebinning_multiple_one_plot(fs=None, save=False):
    sns.set_style('whitegrid')
    path = r'C:\Users\10376\Documents\Phantom Data\Report\CNR Binning/'
    colors = ['mediumblue', 'orangered', 'mediumseagreen']
    folds = folders
    cs = conts
    if fs:
        folds = []
        cs = []
        for f in fs:
            folds.append(folders[f])
            cs.append(conts[f])
    fig, axes = plt.subplots(2, 3, figsize=(8, 6), sharey='all')
    patches = []
    for j, folder in enumerate(folds):
        patches.append(mlines.Line2D([0], [0], color=colors[j], lw=2, linestyle='-', label=cs[j]))
        titles = titles_many[0]
        noise_vals = np.load(rf'C:\Users\10376\Documents\Phantom Data\UNIFORMITY\{folder}\TestNum1_noise_time.npy')  # <pixels, bin, val, time>
        signal = np.load(rf'C:\Users\10376\Documents\Phantom Data\UNIFORMITY\{folder}\TestNum1_signal.npy')

        signal[:, :, 1, :] = np.power(np.divide(signal[:, :, 1, :], signal[:, :, 0, :]), 2)
        noise_vals[:, :, 1, :] = np.power(np.divide(noise_vals[:, :, 1, :], noise_vals[:, :, 0, :]), 2) # Square rel. error

        noise_vals[:, :, 0, :] = noise_vals[:, :, 0, :] / signal[:, :, 0, :]  # Get relative noise mean values
        noise_vals[:, :, 1, :] = np.multiply(noise_vals[:, :, 0, :],
                                           np.sqrt(np.add(noise_vals[:, :, 1, :], signal[:, :, 1, :])))

        noise_vals = noise_vals * 100

        pixels = np.array([1, 2, 3, 4, 6])

        # Grab only the pixel values at the 25 ms and rearrange to <bin, val, pixels> and cut out 12x12 data
        noise_vals = np.transpose(noise_vals[:, :, :, 12], axes=(1, 2, 0))

        # Cut out overflow bins
        plot_cnr = np.zeros([11, 2, len(pixels)])
        plot_cnr[0:5] = noise_vals[0:5]  # Get only SEC data up to the frame desired
        plot_cnr[5:10] = noise_vals[6:11]  # Get only CC data up to the frame desired
        plot_cnr[10] = noise_vals[12]  # Get EC bin for summed bin

        # Make some smooth data
        # pc_shape = np.shape(plot_cnr)
        # cnr_smth = np.zeros([pc_shape[0], 1000])
        # for idx, ypts in enumerate(plot_cnr[:, 0]):
        #     pxs_smth, cnr_smth[idx] = smooth_data(pixels, ypts, cnr_or_noise=0)

        for i, ax in enumerate(axes.flatten()):
            if i < 5:
                # ax.plot(pxs_smth, cnr_smth[i + 5], color='k')
                ax.errorbar(pixels, plot_cnr[i + 5, 0], yerr=plot_cnr[i + 5, 1], capsize=3, color=colors[j])
                ax.set_title(titles[i] + ' keV')
            else:
                # ax.plot(pxs_smth, cnr_smth[-1], color='k')
                ax.errorbar(pixels, plot_cnr[-1, 0], yerr=plot_cnr[-1, 1], capsize=3, color=colors[j])
                ax.set_title(titles[i])
            ax.set_xlim([0, pixels[-1] + 1])
            ax.set_xticks([2, 4, 6])
            ax.set_xticklabels([r'2$\times$2', r'4$\times$4', r'6$\times$6'])

    fig.text(0.5, 0.09, 'Binning', ha='center', fontsize=14)
    fig.text(0.05, 0.52, 'Relative noise (%)', va='center', rotation='vertical', fontsize=14)
    leg = plt.legend(handles=patches, loc='lower center', bbox_to_anchor=(-0.7, -0.55), fancybox=True, shadow=False,
                     ncol=len(fs), fontsize=12)
    fig.add_artist(leg)
    plt.subplots_adjust(hspace=0.45, bottom=0.17)
    plt.show()

    if save:
        plt.savefig(path + 'three_noise.png', dpi=fig.dpi)
        plt.close()
    else:
        plt.pause(5)
        plt.close()


def all_images_pixels(folder_num):
    a1 = AnalyzeUniformity(folders[folder_num], airfolder)
    data = np.load(a1.data_a0)
    airdata = np.load(a1.air_data.data_a0)
    save_folder = os.path.join(r'C:\Users\10376\Documents\Phantom Data\UNIFORMITY/', folders[folder_num], 'Figures',
                               'Pixel Images')
    os.makedirs(save_folder, exist_ok=True)

    for p in [1, 2, 3, 4, 6, 8, 12]:
        fig = plt.figure(figsize=(3, 3))
        title = f'{p}x{p} pixels'
        save_nm = f'/Pixel{p}.png'
        if p > 1:
            td = np.sum(a1.sumpxp(data, p), axis=1)
            ta = np.sum(a1.sumpxp(airdata, p), axis=1)
        else:
            td = np.sum(data, axis=1)
            ta = np.sum(airdata, axis=1)

        corr = a1.intensity_correction(td, ta)
        plt.imshow(corr[12])
        plt.title(title, fontsize=16)
        plt.xticks([])
        plt.yticks([])
        plt.show()
        plt.savefig(save_folder + save_nm, dpi=fig.dpi)
        plt.close()


def add_bins_one_figure():

    tests = np.array([14, 12, 12, 11])
    bins = np.array([[1, 3], [1, 2], [1, 2], [0, 3]])
    plot_titles = ['Blue belt', 'Glass', 'Steel', 'Polypropylene']

    fig, axes = plt.subplots(2, 2, figsize=(7, 6))
    ax = axes.flatten()

    for i, folder in enumerate([folders[0], folders[2], folders[5], folders[6]]):
        syn = AddBinsUniformity(folder, airfolder)
        phys = AnalyzeUniformity(folder, airfolder, test_num=tests[i])

        syn.add_adj_bins(bins[i])
        syn.analyze_cnr_noise()
        phys.analyze_cnr_noise()

        frames = phys.frames
        phys_cnr = phys.cnr_time[0, bins[i][0], :]
        syn_cnr = syn.cnr_time[0, bins[i][0], :]

        ypts_phys = np.concatenate((np.array([0]), phys_cnr[0]))
        ypts_syn = np.concatenate((np.array([0]), syn_cnr[0]))
        xpts = np.concatenate((np.array([0]), frames))

        xp, yp = smooth_data(xpts, ypts_phys, 0)
        xs, ys = smooth_data(xpts, ypts_syn, 0)

        ax[i].plot(xs, ys, color='r')
        ax[i].plot(xp, yp, color='b')

        ax[i].errorbar(frames, phys_cnr[0], yerr=phys_cnr[1], fmt='none', color='b', capsize=3)
        ax[i].errorbar(frames, syn_cnr[0], yerr=syn_cnr[1], fmt='none', color='r', capsize=3)

        ax[i].legend(['Synthetic bin', 'Physical bin'], fontsize=13)
        ax[i].set_xlim([0, 26])
        ax[i].tick_params(labelsize=13)
        ax[i].set_title(plot_titles[i], fontsize=14)

    fig.text(0.5, 0.02, 'Time (ms)', ha='center', fontsize=16)
    fig.text(0.02, 0.5, 'CNR', va='center', rotation='vertical', fontsize=16)
    plt.subplots_adjust(top=0.95, wspace=0.29, hspace=0.37)
    plt.plot()
    plt.savefig(r'C:\Users\10376\Documents\Phantom Data\Report\Ranges/Comparison.png', dpi=fig.dpi)


if __name__ == '__main__':
    #see_ROIs(0, 3)
    #run_cnr_noise_time_pixels()
    cnrbinning_multiple_one_plot(fs=[4, 5, 0], save=True)
    noisebinning_multiple_one_plot(fs=[4, 5, 0], save=True)
    #all_images_pixels(3)
    #all_images_pixels(3)
