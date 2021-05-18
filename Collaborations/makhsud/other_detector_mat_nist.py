import os
import numpy as np
import matplotlib.pyplot as plt

directory = r'D:\OneDrive - University of Victoria\Research\Single Pixel\NIST Data X-ray detectors'
file = 'CsI'
density = 4.51

data = np.loadtxt(os.path.join(directory, file + '.txt'))

data[:, 0] = data[:, 0]
data[:, 1] = data[:, 1] * density  # Change to linear attenuation coefficient


def save_csv(data, name):
    """
    This function will save the data as .csv with the format shown in DetectorCode/array.csv

    :param data: ndarray
            The array to save to 'data.csv'
    """

    data_shp = np.asarray(np.shape(data))
    data_shp[0] = data_shp[0] + 1

    new_array = np.zeros(data_shp)
    new_array[1:, :] = data

    csv_array = new_array.astype(str)

    # Format the first row with legend
    leg = ['Energy (MeV)', 'Attentuation (cm^-1)']

    csv_array[0, :] = leg

    np.savetxt(os.path.join(directory, f'{name}.csv'), csv_array, delimiter=',', fmt='%s')


save_csv(data, file)
