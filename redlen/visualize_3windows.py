import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from scipy.interpolate import make_interp_spline as spline
from redlen.uniformity_analysis import AnalyzeUniformity
from redlen.visualize import VisualizeUniformity
from general_functions import load_object


class Visualize3Windows(VisualizeUniformity):
    def __init__(self, AnalyzeUniformity1, AnalyzeUniformity2, AnalyzeUniformity3):
        self.AnalyzeUniformity = []
        self.AnalyzeUniformity.append(AnalyzeUniformity1)
        self.AnalyzeUniformity.append(AnalyzeUniformity2)
        self.AnalyzeUniformity.append(AnalyzeUniformity3)
        self.save_dir = os.path.join(self.AnalyzeUniformity[0].save_dir, 'Figures')
        os.makedirs(self.save_dir, exist_ok=True)
        self.titles = ['20-30 keV', '30-50 keV', '50-70 keV', '70-90 keV', '90-120 keV', 'EC']

    def plot_cnr_vs_time(self, cnr_or_noise=0, pixel=1, end_time=25, windows=['1w', '3w', '8w'], save=False):
        """
        This function plots CNR or noise over time (up to the end_time) for CC bins at all 3 CC window widths
        :param cnr_or_noise: int
                    0 for CNR, 1 for noise
        :param pixel: int, optional
                    The number of pixels along one direction that were aggregated, defaults to 1
        :param end_time: int, optional
                    The end time on the plot (x-axis) in ms, defaults to 100 ms
        :param windows: list, strings
                    The list of the three window widths for the legend
        :param save: boolean, optional
                    Whether or not to save the figure, defaults to False
        """
        colors = ['k', 'r', 'b']
        px_idx = np.squeeze(np.argwhere(self.AnalyzeUniformity[0].pxp == pixel))  # Find the index of the pixel value

        if cnr_or_noise == 0:
            path = os.path.join(self.save_dir, 'Plots/Three Bins/CNR vs Time')
        else:
            path = os.path.join(self.save_dir, 'Plots/Three Bins/Noise vs Time')
        os.makedirs(path, exist_ok=True)

        frames = self.AnalyzeUniformity[0].frames
        titles = self.titles

        fig, axes = plt.subplots(2, 3, figsize=(8, 6), sharey=True)
        for au in np.arange(3):
            if cnr_or_noise == 0:
                cnr_vals = self.AnalyzeUniformity[au].cnr_time[px_idx]  # <pixels, bin, val, time>
            else:
                cnr_vals = self.AnalyzeUniformity[au].noise_time[px_idx]  # <pixels, bin, val, time>

            plot_cnr = np.zeros([6, 2, len(frames)])
            plot_cnr[0:5] = cnr_vals[6:11]  # Get only CC data up to the frame desired
            plot_cnr[5] = cnr_vals[12]  # Get EC bin for summed bin

            # Make some smooth data
            pc_shape = np.shape(plot_cnr)
            cnr_smth = np.zeros([pc_shape[0], 1000])
            for idx, ypts in enumerate(plot_cnr[:, 0]):
                frms_smth, cnr_smth[idx] = self.smooth_data(frames, ypts, cnr_or_noise)


            for i, ax in enumerate(axes.flatten()):
                ax.plot(frms_smth, cnr_smth[i], color=colors[au])
                # ax.errorbar(frames, plot_cnr[i, 0], yerr=plot_cnr[i, 1], fmt='none', color=colors[au])
                if au == 2:
                    ax.legend(windows)
                    ax.set_xlabel('Time (ms)')
                    if cnr_or_noise == 0:
                        ax.set_ylabel('CNR')
                    else:
                        ax.set_ylabel('Noise')
                    ax.set_xlim([0, end_time])
                    ax.set_title(titles[i])

        plt.subplots_adjust(hspace=0.45, bottom=0.17)
        plt.show()

        if save:
            plt.savefig(path + '/3windows.png', dpi=fig.dpi)
            plt.close()
        else:
            plt.pause(5)
            plt.close()
