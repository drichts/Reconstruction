import numpy as np
from glob import glob
from redlen.redlen_analysis import RedlenAnalyze
import os


class AnalyzeSpectrum(RedlenAnalyze):
    def __init__(self, folder, test_num=0, mm='M20358_D32', load_dir=r'X:\TEST LOG\MINI MODULE\Canon',
                 save_dir=r'C:\Users\10376\Documents\Phantom Data'):
        super().__init__(folder, test_num, mm, 'SPECTRUM', load_dir)
        self.data_a0 = np.transpose(self.data_a0, axes=(1, 0, 2, 3))
        self.data_a1 = np.transpose(self.data_a1, axes=(1, 0, 2, 3))

        self.num_bins = np.shape(self.data_a0)[0]

        self.save_dir = os.path.join(save_dir, 'Spectrum', folder)
        print(self.save_dir)
        os.makedirs(self.save_dir, exist_ok=True)
        self.filename = f'Run{test_num}Spectrum.pk1'

        self.num_pts = np.shape(self.data_a0)[1]
        self.au_to_kev = 0.72

        self.energy = np.linspace(1, self.num_pts)*self.au_to_kev
        self.spectrum = self.get_spectrum()

    def get_spectrum(self):
        """
        This takes the given data matrix and outputs the counts (both sec and cc) over the energy range in AU
        :param data: The data matrix: form [bin, time, view, row, column])
        :return: Array of the counts at each AU value, sec counts and cc counts
        """
        spectrum = np.zeros([self.num_bins, self.num_pts])

        for i in np.arange(self.num_pts):
            # This finds the median pixel in each capture
            temp = np.squeeze(np.median(self.data_a0[:, i], axis=[1, 2]))

            spectrum = np.add(spectrum, temp)

        return spectrum

