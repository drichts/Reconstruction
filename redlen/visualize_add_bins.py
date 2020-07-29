import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from scipy.interpolate import make_interp_spline as spline
from redlen.uniformity_analysis_add_bins import AddBinsUniformity
from general_functions import load_object
from redlen.visualize import VisualizeUniformity


class AddBinsVisualize(VisualizeUniformity):

    def __init__(self, label, AddBinsUniformity, AnalyzeUniformity=None):
        self.AddBinsUniformity = AddBinsUniformity
        self.AnalyzeUniformity = None
        self.save_dir = self.AddBinsUniformity.save_dir
        if AnalyzeUniformity:
            self.AnalyzeUniformity = AnalyzeUniformity
            self.save_dir = self.AnalyzeUniformity.save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.titles = ['20-30', '30-50', '50-70', '70-90', '90-120', 'EC']
        self.label = label

    def plot_comparison(self, bin_num_add, bin_num_reg, cnr_or_noise=0, pixel=1, end_time=25, save=False):
        """
        his function plots CNR or noise over time (up to the end_time) for both CC and SEC bins
        :param bin_num_add: int
        :param bin_num_reg: int
        :param cnr_or_noise: int
                   0 for CNR, 1 for noise
        :param pixel: int, optional
                   The number of pixels along one direction that were aggregated, defaults to 1
        :param end_time: int, optional
                   The end time on the plot (x-axis) in ms, defaults to 25 ms
        :param save: boolean, optional
                   Whether or not to save the figure, defaults to False
        """
        label = self.label
        px_idx = np.squeeze(np.argwhere(self.AddBinsUniformity.pxp == pixel))  # Find the index of the pixel value

        if cnr_or_noise == 0:
            cnr_vals_add = self.AddBinsUniformity.cnr_time[px_idx]  # <pixels, bin, val, time>
            cnr_vals_reg = self.AnalyzeUniformity.cnr_time[px_idx]  # <pixels, bin, val, time>
            path = os.path.join(self.save_dir, 'Added Bins/CNR vs Time')
        else:
            cnr_vals_add = self.AddBinsUniformity.noise_time[px_idx]  # <pixels, bin, val, time>
            cnr_vals_reg = self.AnalyzeUniformity.noise_time[px_idx]  # <pixels, bin, val, time>
            path = os.path.join(self.save_dir, 'Added Bins/Noise vs Time')
        os.makedirs(path, exist_ok=True)

        frames = self.AddBinsUniformity.frames

        plot_cnr_add = np.zeros([2, 2, len(frames)])
        plot_cnr_add[0] = cnr_vals_add[bin_num_add]  # Get only SEC data up to the frame desired
        plot_cnr_add[1] = cnr_vals_add[bin_num_add + self.AddBinsUniformity.cc_step]  # Get only CC data up to the frame desired

        plot_cnr_reg = np.zeros([2, 2, len(frames)])
        plot_cnr_reg[0] = cnr_vals_reg[bin_num_reg]  # Get only SEC data up to the frame desired
        plot_cnr_reg[1] = cnr_vals_reg[bin_num_reg + 6]  # Get only CC data up to the frame desired

        print(np.sum(plot_cnr_add - plot_cnr_reg))

        # Make some smooth data
        cnr_smth_add = np.zeros([2, 1000])
        for idx, ypts in enumerate(plot_cnr_add[:, 0]):
            frms_smth, cnr_smth_add[idx] = self.smooth_data(frames, ypts, cnr_or_noise)

        cnr_smth_reg = np.zeros([2, 1000])
        for idx, ypts in enumerate(plot_cnr_reg[:, 0]):
            frms_smth, cnr_smth_reg[idx] = self.smooth_data(frames, ypts, cnr_or_noise)

        fig, ax = plt.subplots(1, 2, figsize=(6, 3))

        ax[0].plot(frms_smth, cnr_smth_reg[1], color='k')
        ax[0].plot(frms_smth, cnr_smth_reg[0], color='r')
        ax[0].legend(['CC', 'SEC'])

        ax[0].set_xlabel('Time (ms)')
        if cnr_or_noise == 0:
            ax[0].set_ylabel('CNR')
        else:
            ax[0].set_ylabel('Noise')
        ax[0].set_xlim([0, end_time])
        ax[0].set_title(label + ' keV (Actual bin)')

        ax[1].plot(frms_smth, cnr_smth_add[1], color='k')
        ax[1].plot(frms_smth, cnr_smth_add[0], color='r')
        ax[1].legend(['CC', 'SEC'])

        ax[1].set_xlabel('Time (ms)')
        if cnr_or_noise == 0:
            ax[1].set_ylabel('CNR')
        else:
            ax[1].set_ylabel('Noise')
        ax[1].set_xlim([0, end_time])
        ax[1].set_title(label + ' keV (Added bins)')
        plt.subplots_adjust(left=0.12, bottom=0.15, right=0.95, top=0.9, wspace=0.32)
        plt.show()

        if save:
            plt.savefig(path + f'/TestNum{self.AddBinsUniformity.test_num}_Comparison_' + label + '.png', dpi=fig.dpi)
            plt.close()
        else:
            plt.pause(5)
            plt.close()


    def plot_bin(self, bin_num, cnr_or_noise=0, pixel=1, end_time=25, save=False):
        """
        This function plots CNR or noise over time (up to the end_time) for both CC and SEC bins
        :param bin_num: int
        :param cnr_or_noise: int
                    0 for CNR, 1 for noise
        :param pixel: int, optional
                    The number of pixels along one direction that were aggregated, defaults to 1
        :param end_time: int, optional
                    The end time on the plot (x-axis) in ms, defaults to 25 ms
        :param save: boolean, optional
                    Whether or not to save the figure, defaults to False
        """
        px_idx = np.squeeze(np.argwhere(self.AddBinsUniformity.pxp == pixel))  # Find the index of the pixel value

        if cnr_or_noise == 0:
            cnr_vals = self.AddBinsUniformity.cnr_time[px_idx]  # <pixels, bin, val, time>
            path = os.path.join(self.save_dir, 'Plots/CNR vs Time')
        else:
            cnr_vals = self.AddBinsUniformity.noise_time[px_idx]  # <pixels, bin, val, time>
            path = os.path.join(self.save_dir, 'Plots/Noise vs Time')
        os.makedirs(path, exist_ok=True)

        frames = self.AddBinsUniformity.frames
        titles = self.titles

        plot_cnr = np.zeros([2, 2, len(frames)])
        plot_cnr[0] = cnr_vals[bin_num]  # Get only SEC data up to the frame desired
        plot_cnr[1] = cnr_vals[bin_num + self.AddBinsUniformity.cc_step]  # Get only CC data up to the frame desired

        # Make some smooth data
        cnr_smth = np.zeros([2, 1000])
        for idx, ypts in enumerate(plot_cnr[:, 0]):
            frms_smth, cnr_smth[idx] = self.smooth_data(frames, ypts, cnr_or_noise)

        fig = plt.figure(figsize=(3, 3))

        plt.plot(frms_smth, cnr_smth[1], color='k')
        plt.plot(frms_smth, cnr_smth[0], color='r')
        #plt.errorbar(frames, plot_cnr[1, 0], yerr=plot_cnr[1, 1], fmt='none', color='k')
        #plt.errorbar(frames, plot_cnr[0, 0], yerr=plot_cnr[0, 1], fmt='none', color='r')
        plt.legend(['CC', 'SEC'])

        plt.xlabel('Time (ms)')
        if cnr_or_noise == 0:
            plt.ylabel('CNR')
        else:
            plt.ylabel('Noise')
        plt.xlim([0, end_time])
        plt.title(titles[bin_num])
        plt.show()

        if save:
            plt.savefig(path + f'/TestNum{self.AddBinsUniformity.test_num}.png', dpi=fig.dpi)
            plt.close()
        else:
            plt.pause(5)
            plt.close()
