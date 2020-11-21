import numpy as np
import matplotlib.pyplot as plt
import glob
import mask_functions as grm
from obsolete import sct_analysis as sct


def read_csv():
    directory = r'D:\Research\Python Data\Redlen\Attenuation/'

    files = glob.glob(directory + '*.txt')
    print(files)
    for filename in files:
        savename = filename.replace(filename[-3:], 'npy')
        print(savename)

        data = np.loadtxt(filename)

        print(np.shape(data))

        np.save(savename, data)
        print()

def show_fig():
    directory = 'C:/Users/10376/Documents/ndt_excel/'
    filename = 'steel.npy'

    data = np.load(directory + filename)
    air = np.load(directory + 'air.npy')

    data_corr = np.log(np.divide(air, data))

    continue_flag = True
    while continue_flag:
        contrast_mask = grm.background_ROI(data_corr)
        val = input('Were the ROIs acceptable? (y/n)')
        if val is 'y':
            continue_flag = False

    continue_flag2 = True
    while continue_flag2:
        bg_mask = grm.background_ROI(data_corr)
        val = input('Were the ROIs acceptable? (y/n)')
        if val is 'y':
            continue_flag2 = False

    cnr_data, err_data = sct.cnr(data, contrast_mask, bg_mask)
    cnr_corr, err_corr = sct.cnr(data_corr, contrast_mask, bg_mask)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    title = filename.replace('.npy', '')
    ax[0].imshow(data)
    ax[0].set_xlabel('CNR = ' + '{:.2f}'.format(cnr_data) + ' +- ' + '{:.2f}'.format(err_data))
    ax[0].set_title(title + ' raw')
    ax[1].imshow(data_corr)
    ax[1].set_title(title + ' corrected for air')
    ax[1].set_xlabel('CNR = ' + '{:.2f}'.format(cnr_corr) + ' +- ' + '{:.2f}'.format(err_corr))
    plt.show()
    plt.savefig(directory + title + '.png', dpi=fig.dpi)


if __name__ == '__main__':
    read_csv()