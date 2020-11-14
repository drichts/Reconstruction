import numpy as np
import matplotlib.pyplot as plt
from os import path
from glob import glob
from analysis import Analyze
import


class AnalyzeCT(Analyze):

    def __init__(self, folder, directory='D:/Research/LDA Data/'):
        self.folder = folder
        self.directory = path.join(directory, folder)

        self.data = path.join(self.directory, 'raw_data.npy')
        self.airdata = path.join(self.directory, 'airscan.npy')

        self.proj = path.join(self.directory, 'projections.npy')

        if not path.exists(self.proj):
            np.save(self.proj, self.intensity_correction(np.load(self.data), np.load(self.airdata)))

    @staticmethod
    def intensity_correction(data, air_data, dark_data):
        """
        This function corrects flatfield data to show images, -ln(I/I0), I is the intensity of the data, I0 is the
        intensity in an airscan
        :param data: The data to correct (must be the same shape as air_data)
        :param air_data: The airscan data (must be the same shape as data)
        :param dark_data: The darkscan data (must be the same shape as airdata)
        :return: The corrected data array
        """
        return np.log(np.subtract(air_data, dark_data)) - np.log(np.subtract(data, dark_data))
