import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from glob import glob

directory = ''
mm = '/M20358_D32/'
folder = ''
air_folder = ''
filenum = 0

data_path = glob(directory + mm + folder + '/Raw Test Data/' + mm + 'UNIFORMITY/*A0*')[filenum]
air_path = glob(directory + mm + air_folder + '/Raw Test Data/' + mm + 'UNIFORMITY/*A0*')[filenum]


def test_phantom_alignment(data_path, air_path):
    """
    This function is to test the inital alignment of a phantom at Redlen
    :param data_path: The data path to the mat file
    :param air_path: The data path to the mat air file
    :param num: The file number to look at
    :return: shows an image of the phantom after air correction and True if the phantom is small, False if not
    """

    data = np.squeeze(loadmat(data_path)['cc_struct']['data'][0][0][0][0][0])
    air = np.squeeze(loadmat(air_path)['cc_struct']['data'][0][0][0][0][0])

    # Do the air correction
    corr = np.squeeze(np.log(np.divide(air, data)))
    corr = np.sum(corr, axis=1)

    plt.imshow(corr[12])
    plt.pause(2)
    plt.close()

test_phantom_alignment(data_path, air_path)