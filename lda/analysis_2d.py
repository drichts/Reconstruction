import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
from analysis import Analyze


class AnalyzeLDA(Analyze):

    def __init__(self, folder, load_dir='D:/Research/Python Data/Redlen/LDA'):
        self.folder = folder
        self.load_dir = load_dir
        self.save_dir = os.path.join(load_dir, folder)

        self.thresholds = ['16-30 keV', '30-50 keV', '50-70 keV', '70-90 keV', '90-120 keV', '>120 keV', '16-120 keV']

        self.data = np.sum(np.load(glob(os.path.join(self.load_dir, self.folder, '*phantom*'))[0]), axis=0)
        self.airdata = np.sum(np.load(glob(os.path.join(self.load_dir, self.folder, '*air*'))[0]), axis=0)

        self.corr = np.transpose(self.intensity_correction(self.data, self.airdata), axes=(2, 0, 1))

    def visualize(self, vi, vm):
        fig, axes = plt.subplots(7, 1, figsize=(7, 7))
        for i, ax in enumerate(axes.flatten()):
            ax.imshow(self.corr[i], vmin=vi, vmax=vm)
            ax.set_title(self.thresholds[i])
        plt.subplots_adjust(bottom=0.05, top=0.95, hspace=0.95)
        plt.show()
        plt.savefig(os.path.join(self.save_dir, self.folder + '.png'), dpi=fig.dpi)
        plt.close()

    @staticmethod
    def intensity_correction(data, air_data):
        """
        This function corrects flatfield data to show images, -ln(I/I0), I is the intensity of the data, I0 is the
        intensity in an airscan
        :param data: The data to correct (must be the same shape as air_data)
        :param air_data: The airscan data (must be the same shape as data)
        :return: The corrected data array
        """
        return np.log(np.divide(air_data, data))
