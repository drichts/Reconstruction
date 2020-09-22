import numpy as np
import os
from redlen.uniformity_analysis import AnalyzeUniformity


class AddBinsUniformity(AnalyzeUniformity):

    def __init__(self, folder, air_folder, test_num=1, mm='M20358_D32', load_dir=r'X:\TEST LOG\MINI MODULE\Canon',
                 save_dir=r'C:\Users\10376\Documents\Phantom Data'):

        super().__init__(folder, air_folder, test_num, mm, load_dir, save_dir)
        self.save_dir = os.path.join(self.save_dir, 'Added Bins')
        self.air_data.save_dir = os.path.join(self.air_data.save_dir, 'Added Bins')
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.air_data.save_dir, exist_ok=True)

        a0_file = os.path.join(self.save_dir, f'TestNum{self.test_num}_DataA0.npy')
        air_a0_file = os.path.join(self.air_data.save_dir, f'TestNum{self.test_num}_DataA0.npy')
        np.save(a0_file, np.load(self.data_a0))
        np.save(air_a0_file, np.load(self.air_data.data_a0))

        self.data_a0 = a0_file
        self.air_data.data_a0 = air_a0_file
        self.cc_step = 6

    def add_adj_bins(self, bins):
        """
        This function takes the adjacent bins given in bins and sums along the bin axis, can sum multiple bins
        :param bins: 1D array
                    Bin numbers (as python indices, i.e the 1st bin would be 0) to sum
                    Form: [Starting bin, Ending bin]
                    Ex. for the 2nd through 5th bins, bins = [1, 4]
        :return: The summed data with the summed bins added together and the rest of the data intact
                    shape <counters, views, rows, columns>
        """
        data = np.load(self.data_a0)
        air_data = np.load(self.air_data.data_a0)

        data_shape = np.array(np.shape(data))

        # The new data will have the number of added bins - 1 new counters
        data_shape[0] = data_shape[0] - 2*(bins[1] - bins[0])
        cc_step = 6 - (bins[1] - bins[0])
        new_data = np.zeros(data_shape)
        new_air_data = np.zeros(data_shape)

        new_data[0:bins[0]] = data[0:bins[0]]
        new_data[cc_step:bins[0]+cc_step] = data[6:bins[0]+6]
        new_air_data[0:bins[0]] = air_data[0:bins[0]]
        new_air_data[cc_step:bins[0] + cc_step] = air_data[6:bins[0] + 6]

        new_data[bins[0]] = np.sum(data[bins[0]:bins[-1] + 1], axis=0)
        new_data[bins[0] + cc_step] = np.sum(data[bins[0] + 6:bins[-1] + 7], axis=0)
        new_air_data[bins[0]] = np.sum(air_data[bins[0]:bins[-1] + 1], axis=0)
        new_air_data[bins[0] + cc_step] = np.sum(air_data[bins[0] + 6:bins[-1] + 7], axis=0)

        new_data[bins[0] + 1:cc_step] = data[bins[1] + 1:6]
        new_data[bins[0] + cc_step + 1:] = data[bins[1] + 7:]
        new_air_data[bins[0] + 1:cc_step] = air_data[bins[1] + 1:6]
        new_air_data[bins[0] + cc_step + 1:] = air_data[bins[1] + 7:]

        np.save(self.data_a0, new_data)
        np.save(self.air_data.data_a0, new_air_data)
        self.cc_step = cc_step
        self.data_shape = data_shape
        self.num_bins = data_shape[0]

    def add_cc_and_sec(self):
        data = np.load(self.data_a0)
        air_data = np.load(self.air_data.data_a0)

        data_shape = np.array(self.data_shape)
        data_shape[0] = int((data_shape[0] - 1) / 2) + 1
        new_data = np.zeros(data_shape)
        new_air_data = np.zeros(data_shape)

        new_data[-1] = data[-1]
        new_air_data[-1] = air_data[-1]

        for i in np.arange(data_shape[0] - 1):
            new_data[i] = np.add(data[i], data[i + self.cc_step])
            new_air_data[i] = np.add(air_data[i], air_data[i + self.cc_step])

        np.save(self.data_a0, new_data)
        np.save(self.air_data.data_a0, new_air_data)
        self.data_shape = data_shape
        self.num_bins = data_shape[0]
