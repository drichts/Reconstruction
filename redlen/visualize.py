import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from scipy.interpolate import make_interp_spline as spline
from redlen.uniformity_analysis import AnalyzeUniformity
from general_functions import load_object


class VisualizeUniformity:
    def __init__(self, AnalyzeUniformity):
        self.AnalyzeUniformity = AnalyzeUniformity
        self.save_dir = os.path.join(self.AnalyzeUniformity.save_dir, 'Figures')
        os.makedirs(self.save_dir, exist_ok=True)
        self.titles = ['20-30 keV', '30-40 keV', '40-50 keV', '50-60 keV', '60-70 keV', 'EC']

    def find_titles(self):
        """Create the plot titles from the energy threshold values"""
        thresholds = self.AnalyzeUniformity.thresholds
        num_files = len(thresholds)
        titles = np.empty([num_files, 6], dtype='str')
        for i, file in enumerate(thresholds):
            for j in range(5):
                titles[i, j] = f'{file[j]}-{file[j+1]} keV'
            titles[i, 5] = f'{file[0]}-{file[5]} keV (EC)'

        return titles

    @staticmethod
    def smooth_data(xpts, ypts, cnr_or_noise):
        xsmth = np.linspace(xpts[0], xpts[-1], 1000)
        if cnr_or_noise == 2:  # NOISE
            coeffs = np.polyfit(xpts, ypts, 3)
            p = np.poly1d(coeffs)
            ysmth = p(xsmth)
        else:  # CNR
            p = spline(xpts, ypts)
            ysmth = p(xsmth)
        return xsmth, ysmth

    def blank_vs_time_single_bin(self, bin_num, cnr_or_noise=0, pixel=1, end_time=25, save=False):
        """
                This function plots CNR or noise over time (up to the end_time) for both CC and SEC bins
                :param cnr_or_noise: int
                            0 for CNR, 1 for noise
                :param pixel: int, optional
                            The number of pixels along one direction that were aggregated, defaults to 1
                :param end_time: int, optional
                            The end time on the plot (x-axis) in ms, defaults to 25 ms
                :param save: boolean, optional
                            Whether or not to save the figure, defaults to False
                """
        px_idx = np.squeeze(np.argwhere(self.AnalyzeUniformity.pxp == pixel))  # Find the index of the pixel value

        pixel_path = f'{pixel}x{pixel} Data'

        if cnr_or_noise == 0:
            cnr_vals = self.AnalyzeUniformity.cnr_time[px_idx]  # <pixels, bin, val, time>
            path = os.path.join(self.save_dir, 'Plots/CNR vs Time', pixel_path)
        else:
            cnr_vals = self.AnalyzeUniformity.noise_time[px_idx]  # <pixels, bin, val, time>
            path = os.path.join(self.save_dir, 'Plots/Noise vs Time', pixel_path)
        os.makedirs(path, exist_ok=True)

        frames = self.AnalyzeUniformity.frames
        titles = self.titles

        plot_cnr = np.zeros([2, 2, len(frames)])
        plot_cnr[0] = cnr_vals[bin_num]  # Get only SEC data up to the frame desired
        plot_cnr[1] = cnr_vals[bin_num + 6]  # Get only CC data up to the frame desired

        # Make some smooth data
        pc_shape = np.shape(plot_cnr)
        cnr_smth = np.zeros([pc_shape[0], 1000])
        for idx, ypts in enumerate(plot_cnr[:, 0]):
            frms_smth, cnr_smth[idx] = self.smooth_data(frames, ypts, cnr_or_noise)

        fig = plt.figure(figsize=(3, 3))

        plt.plot(frms_smth, cnr_smth[1], color='k')
        plt.plot(frms_smth, cnr_smth[0], color='r')
        # plt.errorbar(frames, plot_cnr[1, 0], yerr=plot_cnr[1, 1], fmt='none', color='k')
        # plt.errorbar(frames, plot_cnr[0, 0], yerr=plot_cnr[0, 1], fmt='none', color='r')
        plt.legend(['CC', 'SEC'])

        if cnr_or_noise == 0:
            plt.ylabel('CNR')
        else:
            plt.set_ylabel('Noise')
            plt.set_xlim([0, end_time])
            plt.set_title(titles[bin_num] + ' keV')

        #plt.subplots_adjust(hspace=0.45, bottom=0.17)
        plt.show()

        if save:
            plt.savefig(path + f'/TestNum{self.AnalyzeUniformity.test_num}_Bin{bin_num}.png', dpi=fig.dpi)
            plt.close()
        else:
            plt.pause(5)
            plt.close()

    def blank_vs_time_six_bins(self, cnr_or_noise=0, pixel=1, end_time=25, save=False):
        """
        This function plots CNR or noise over time (up to the end_time) for both CC and SEC bins
        :param cnr_or_noise: int
                    0 for CNR, 1 for noise
        :param pixel: int, optional
                    The number of pixels along one direction that were aggregated, defaults to 1
        :param end_time: int, optional
                    The end time on the plot (x-axis) in ms, defaults to 25 ms
        :param save: boolean, optional
                    Whether or not to save the figure, defaults to False
        """
        px_idx = np.squeeze(np.argwhere(self.AnalyzeUniformity.pxp == pixel))  # Find the index of the pixel value

        pixel_path = f'{pixel}x{pixel} Data'

        if cnr_or_noise == 0:
            cnr_vals = self.AnalyzeUniformity.cnr_time[px_idx]  # <pixels, bin, val, time>
            path = os.path.join(self.save_dir, 'Plots/CNR vs Time', pixel_path)
        else:
            cnr_vals = self.AnalyzeUniformity.noise_time[px_idx]  # <pixels, bin, val, time>
            path = os.path.join(self.save_dir, 'Plots/Noise vs Time', pixel_path)
        os.makedirs(path, exist_ok=True)

        frames = self.AnalyzeUniformity.frames
        titles = self.titles

        plot_cnr = np.zeros([11, 2, len(frames)])
        plot_cnr[0:5] = cnr_vals[0:5]  # Get only SEC data up to the frame desired
        plot_cnr[5:10] = cnr_vals[6:11]  # Get only CC data up to the frame desired
        plot_cnr[10] = cnr_vals[12]  # Get EC bin for summed bin

        # Make some smooth data
        pc_shape = np.shape(plot_cnr)
        cnr_smth = np.zeros([pc_shape[0], 1000])
        for idx, ypts in enumerate(plot_cnr[:, 0]):
            frms_smth, cnr_smth[idx] = self.smooth_data(frames, ypts, cnr_or_noise)

        fig, axes = plt.subplots(2, 3, figsize=(8, 6), sharey=True)
        for i, ax in enumerate(axes.flatten()):
            if i < 5:
                ax.plot(frms_smth, cnr_smth[i+5], color='k')
                ax.plot(frms_smth, cnr_smth[i], color='r')
                #ax.errorbar(frames, plot_cnr[i+5, 0], yerr=plot_cnr[i+5, 1], fmt='none', color='k')
                #ax.errorbar(frames, plot_cnr[i, 0], yerr=plot_cnr[i, 1], fmt='none', color='r')
                ax.legend(['CC', 'SEC'])
                ax.set_title(titles[i] + ' keV')
            else:
                ax.plot(frms_smth, cnr_smth[-1], color='k')
                #ax.errorbar(frames, plot_cnr[-1, 0], yerr=plot_cnr[-1, 1], fmt='none', color='k')
                ax.set_title(titles[i])
            ax.set_xlabel('Time (ms)')
            if cnr_or_noise == 0:
                ax.set_ylabel('CNR')
            else:
                ax.set_ylabel('Noise')
            ax.set_xlim([0, end_time])


        plt.subplots_adjust(hspace=0.45, bottom=0.17)
        plt.show()

        if save:
            plt.savefig(path + f'/2_TestNum{self.AnalyzeUniformity.test_num}.png', dpi=fig.dpi)
            plt.close()
        else:
            plt.pause(5)
            plt.close()

    def blank_vs_pixels_six_bins(self, cnr_or_noise=0, time=10, y_lim=None, save=False):
        """
        This function plots the 6 energy bins with pixels vs. CNR or noise
        :param cnr_or_noise: int
                    0 for CNR, 1 for noise
        :param time: int, optional
                    The accumulated time to plot in ms, defaults to 10 ms
        :param y_lim: float, optional
                    If there is an outlier, you may set the upper y-limit to your desired number
        :param save: boolean, optional
                    Whether or not to save the figure, defaults to False
        :return:
        """
        titles = self.titles
        if cnr_or_noise == 0:
            cnr_vals = self.AnalyzeUniformity.cnr_time  # <pixels, bin, val, time>
            path = os.path.join(self.save_dir, 'Plots/CNR vs Pixels')
        else:
            cnr_vals = self.AnalyzeUniformity.noise_time  # <pixels, bin, val, time>
            path = os.path.join(self.save_dir, 'Plots/Noise vs Pixels')
        os.makedirs(path, exist_ok=True)

        pixels = self.AnalyzeUniformity.pxp
        time_idx = np.squeeze(np.argwhere(self.AnalyzeUniformity.frames == time))

        # Grab only the pixel values at the time value and rearrange to <bin, val, pixels>

        cnr_vals = np.transpose(cnr_vals[:, :, :, time_idx], axes=(1, 2, 0))

        # Cut out overflow bins
        plot_cnr = np.zeros([11, 2, len(pixels)])
        plot_cnr[0:5] = cnr_vals[0:5]  # Get only SEC data up to the frame desired
        plot_cnr[5:10] = cnr_vals[6:11]  # Get only CC data up to the frame desired
        plot_cnr[10] = cnr_vals[12]  # Get EC bin for summed bin

        # Make some smooth data
        pc_shape = np.shape(plot_cnr)
        cnr_smth = np.zeros([pc_shape[0], 1000])
        for idx, ypts in enumerate(plot_cnr[:, 0]):
            pxs_smth, cnr_smth[idx] = self.smooth_data(pixels, ypts, cnr_or_noise)

        fig, axes = plt.subplots(2, 3, figsize=(8, 6), sharey=True)
        for i, ax in enumerate(axes.flatten()):
            if i < 5:
                #ax.plot(pxs_smth, cnr_smth[i + 5], color='k')
                #ax.plot(pxs_smth, cnr_smth[i], color='r')
                ax.errorbar(pixels, plot_cnr[i + 5, 0], yerr=plot_cnr[i + 5, 1], fmt='none', capsize=3, color='k')
                ax.errorbar(pixels, plot_cnr[i, 0], yerr=plot_cnr[i, 1], fmt='none', capsize=3, color='r')
                ax.legend(['CC', 'SEC'])
                ax.set_title(titles[i] + ' keV')
            else:
                #ax.plot(pxs_smth, cnr_smth[i], color='k')
                ax.errorbar(pixels, plot_cnr[-1, 0], yerr=plot_cnr[-1, 1], fmt='none', capsize=3, color='k')
                ax.set_title(titles[i])
            ax.set_xlabel('Pixels')
            if cnr_or_noise == 0:
                ax.set_ylabel('CNR')
            else:
                ax.set_ylabel('Noise')
            ax.set_xlim([0, pixels[-1]+0.5])
            if y_lim:
                ax.set_ylim([0, y_lim])


        plt.subplots_adjust(hspace=0.45, bottom=0.17)
        plt.show()

        if save:
            plt.savefig(path + f'/2_TestNum{self.AnalyzeUniformity.test_num}_Time{time}ms.png', dpi=fig.dpi)
            plt.close()
        else:
            plt.pause(5)
            plt.close()

    def cnr_bar_plot(self, time=10, pixel=1, save=False):
        """
        This function plots a bar plot with the CNR at a specific time in each of the six bins
        :param time: int, optional
                    The accumulated time to plot in ms, defaults to 10 ms
        :param pixel: int, optional
                    The number of aggregated pixels, defaults to 1
        :param save: boolean, optional
                    Whether or not to save the figure, defaults to False
        :return:
        """
        fig = plt.figure(figsize=(6, 6))
        #titles = np.delete(self.titles, 5)

        time_idx = np.squeeze(np.argwhere(self.AnalyzeUniformity.frames == time))
        pix_idx = np.squeeze(np.argwhere(self.AnalyzeUniformity.pxp == pixel))

        vals = self.AnalyzeUniformity.cnr_time[pix_idx, 6:, 0, time_idx]
        uncer = self.AnalyzeUniformity.cnr_time[pix_idx, 6:, 1, time_idx]

        vals = np.delete(vals, 5)  # Delete the overflow bin
        uncer = np.delete(uncer, 5)

        pts = np.arange(6)
        plt.bar(pts, vals, yerr=uncer, capsize=3)
        plt.xlabel('Energy bins (keV)')
        plt.ylabel('CNR')
        #plt.xticks(pts, titles)

        plt.show()

        if save:
            plt.savefig(self.save_dir + 'blah.png', dpi=fig.dpi)
            plt.close()
        else:
            plt.pause(5)
            plt.close()

    def pixel_images(self, save=False):
        """

        :param time:
        :param save:
        :return:
        """
        path = os.path.join(self.save_dir, 'Pixel Images')
        os.makedirs(path, exist_ok=True)

        data = np.load(self.AnalyzeUniformity.data_a0)
        airdata = np.load(self.AnalyzeUniformity.air_data.data_a0)

        for p, pix in enumerate(self.AnalyzeUniformity.pxp):
            if pix == 1:
                data_pxp = np.sum(data[12], axis=0)
                air_pxp = np.sum(airdata[12], axis=0)
            else:
                data_pxp = np.squeeze(self.AnalyzeUniformity.sumpxp(data, pix))  # Aggregate the pixels
                air_pxp = np.squeeze(self.AnalyzeUniformity.sumpxp(airdata, pix))
                data_pxp = np.sum(data_pxp[12], axis=0)
                air_pxp = np.sum(air_pxp[12], axis=0)

            fig = plt.figure(figsize=(3, 3))
            plt.imshow(-1*np.log(data_pxp/air_pxp))
            plt.title(f'{pix}x{pix} pixels', fontsize=16)
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

            if save:
                plt.savefig(path + f'/Pixel{pix}.png', dpi=fig.dpi)
                plt.close()
            else:
                plt.pause(5)
                plt.close()


    # titles_all = [['20-30', '30-50', '50-70', '70-90', '90-120', 'EC'],
    #               ['20-50', '50-70', '70-90', '90-100', '100-120', 'EC'],
    #               ['20-55', '55-70', '70-90', '90-100', '100-120', 'EC'],
    #               ['20-60', '60-75', '75-90', '90-100', '100-120', 'EC'],
    #               ['20-65', '65-80', '80-90', '90-100', '100-120', 'EC'],
    #               ['20-70', '70-80', '80-90', '90-100', '100-120', 'EC'],
    #               ['20-80', '80-90', '90-100', '100-110', '110-120', 'EC'],
    #               ['20-90', '90-100', '100-105', '105-110', '110-120', 'EC']]

 # def contrast_vs_time(folder, test_num, pixel, time, titles, ws=['1w', '3w', '8w'], CC='CC', save=False,
    #                 directory='C:/Users/10376/Documents/Phantom Data/'):
    #
    #     colors = ['black', 'red', 'blue', 'green']
    #     path = directory + folder + '/'
    #     w_folders = glob(path + '*w*/')  # Get each of the different w folders
    #
    #     gof.create_folder('Plots', path) # Create the folder
    #     path = path + 'Plots/'
    #     gof.create_folder('Contrast vs Time', path)
    #     path = path + 'Contrast vs Time/'
    #
    #     fig, axes = plt.subplots(2, 3, figsize=(8, 6), sharey=True)
    #     pixel_path = str(pixel) + 'x' + str(pixel) + ' Data/'
    #     gof.create_folder(pixel_path, path)
    #     path = path + pixel_path
    #
    #     for c, fold in enumerate(w_folders):
    #         contrast_path = fold + pixel_path + '/*' + str(test_num) + '_Avg_contrast.npy'
    #         frames_path = fold + pixel_path + '/*' + str(test_num) + '_Frames.npy'
    #
    #         contrast = np.load(glob(contrast_path)[0])
    #         frames = np.load(glob(frames_path)[0])
    #         full_frames = frames
    #
    #         stop_idx = int(np.argwhere(frames == time) + 1)
    #         frames = frames[0:stop_idx]
    #
    #         plot_contrast = np.zeros([len(frames), 6, 2])
    #         if CC == 'CC':
    #             plot_contrast[:, 0:5] = contrast[0:stop_idx, 6:11]  # Get only CC data up to the frame desired
    #         else:
    #             plot_contrast[:, 0:5] = contrast[0:stop_idx, 0:5]  # Get only SEC data up to the frame desired
    #         plot_contrast[:, 5] = contrast[0:stop_idx, 12]  # Get EC bin for summed bin
    #         contrast = np.abs(contrast)
    #         for i, ax in enumerate(axes.flatten()):
    #             # Make some smooth data
    #             xnew = np.linspace(0, 1000, 1000)
    #             if i == 5:
    #                 spl = spline(full_frames, contrast[:, 12, 0])
    #             elif CC == 'CC':
    #                 spl = spline(full_frames, contrast[:, i+6, 0])
    #             else:
    #                 spl = spline(full_frames, contrast[:, i, 0])
    #             contrast_smth = spl(xnew)
    #
    #             ax.plot(xnew, contrast_smth, color=colors[c])
    #             #ax.errorbar(frames, plot_contrast[:, i, 0], yerr=plot_contrast[:, i, 1], fmt='none', color=colors[c])
    #             ax.set_xlabel('Time (ms)')
    #             ax.set_ylabel('Contrast')
    #             ax.set_xlim([0, time])
    #             if i == 5:
    #                 ax.set_title(titles[i] + ' Bin')
    #             else:
    #                 ax.set_title(titles[i] + ' keV ' + CC)
    #
    #     ax1 = fig.add_subplot(111, frameon=False)
    #     ax1.grid(False)
    #     # Hide axes ticks
    #     ax1.set_xticks([])
    #     ax1.set_yticks([])
    #     blackpatch = mpatches.Patch(color=colors[0], label=ws[0])
    #     redpatch = mpatches.Patch(color=colors[1], label=ws[1])
    #     bluepatch = mpatches.Patch(color=colors[2], label=ws[2])
    #     leg = plt.legend(handles=[blackpatch, redpatch, bluepatch], loc='lower center', bbox_to_anchor=(0.5, -0.19), ncol=3,
    #                      fancybox=True)
    #     ax1.add_artist(leg)
    #     plt.subplots_adjust(hspace=0.45, bottom=0.17)
    #
    #     if save:
    #         plt.savefig(path + 'TestNum_' + str(test_num) + '_' + CC + '.png', dpi=fig.dpi)
    #         plt.close()
    #     else:
    #         plt.show()