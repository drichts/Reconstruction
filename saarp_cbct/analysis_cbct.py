import os
import numpy as np
import pydicom as pyd
import matplotlib.pyplot as plt
from glob import glob


class AnalyzeCBCT:

    def __init__(self, folder, reanalyze=False, directory=r'D:\OneDrive - University of Victoria\Research/CBCT'):
        self.folder = os.path.join(directory, folder)
        self.raw_data = os.path.join(self.folder, 'Raw Data')
        self.npy_data = os.path.join(self.folder, 'Npy Data')
        os.makedirs(self.raw_data, exist_ok=True)
        os.makedirs(self.npy_data, exist_ok=True)

        raw_files = glob(os.path.join(self.folder, '*volume*.dcm'))

        data0 = pyd.dcmread(raw_files[0])
        data0 = data0.pixel_array
        self.img_shape = np.shape(data0)
        self.num_slices = len(raw_files)
        self.data_shape = [self.num_slices, *self.img_shape]

        raw_npy = np.zeros(self.data_shape)
        raw_npy[0] = data0

        os.rename(raw_files[0], os.path.join(self.raw_data, 'volume0001.dcm'))

        for i, file in enumerate(raw_files[1:]):
            new_file = os.path.join(self.raw_data, file.split(self.folder)[1][1:])

            temp_data = pyd.dcmread(file)
            raw_npy[i+1] = temp_data.pixel_array
            os.rename(file, os.path.join(self.raw_data, new_file))


        self.npy_data = os.path.join(self.npy_data, 'data.npy')
        np.save(self.npy_data, raw_npy)