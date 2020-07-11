import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from redlen.redlen_analysis import RedlenAnalyze
import os


class AnalyzeSpectrum(RedlenAnalyze):
    def __init__(self, folder, test_num=0, mm='M20358_D32', load_dir=r'X:\TEST LOG\MINI MODULE\Canon',
                 save_dir=r'C:\Users\10376\Documents\Phantom Data'):
        super().__init__(folder, test_num, mm, 'SPECTRUM', load_dir, save_dir)
        self.data_a0 = np.transpose(self.data_a0, axes=(1, 0, 2, 3))
        self.data_a1 = np.transpose(self.data_a1, axes=(1, 0, 2, 3))

        self.num_bins = np.shape(self.data_a0)[0]

        self.num_pts = np.shape(self.data_a0)[1]
        self.au_to_kev = 0.72

        self.energy = np.arange(1, self.num_pts+1)*self.au_to_kev
        self.spectrum = self.get_spectrum()

    def get_spectrum(self):
        """
        This takes the given data matrix and outputs the counts (both sec and cc) over the energy range in AU
        :param data: The data matrix: form [bin, time, view, row, column])
        :return: Array of the counts at each AU value, sec counts and cc counts
        """
        data = np.load(self.data_a0)
        spectrum = np.zeros([self.num_pts])

        for i in np.arange(self.num_pts):
            # This finds the median pixel in each capture
            temp = np.squeeze(np.median(np.sum(data[6:11, i], axis=0)))

            spectrum[i] = temp
        return spectrum

