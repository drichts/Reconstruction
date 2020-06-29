import numpy as np
from glob import glob
from redlen.redlen_analysis import RedlenAnalyze


class AnalyzeSpectrum(RedlenAnalyze):
    def __init__(self, folder, mm='M20358_D32', load_dir=r'X:\TEST LOG\MINI MODULE\Canon',
                 save_dir=r'C:\Users\10376\Documents\Phantom Data'):
        super.__init__(folder, mm, 'SPECTRUM', load_dir, save_dir)

    @staticmethod
    def get_spectrum(data):
        """
        This takes the given data matrix and outputs the counts (both sec and cc) over the energy range in AU
        :param data: The data matrix: form [time, bin, view, row, column])
        :return: Array of the counts at each AU value, sec counts and cc counts
        """
        length = len(data)  # The number of energy points in AU
        spectrum_sec = np.zeros(length)
        spectrum_cc = np.zeros(length)

        for i in np.arange(5):
            # This finds the median pixel in each capture
            temp_sec = np.squeeze(np.median(data[:, i, :, :, :], axis=[2, 3]))
            temp_cc = np.squeeze(np.median(data[:, i + 6, :, :, :], axis=[2, 3]))

            spectrum_sec = np.add(spectrum_sec, temp_sec)
            spectrum_cc = np.add(spectrum_cc, temp_cc)

        return spectrum_cc, spectrum_sec

