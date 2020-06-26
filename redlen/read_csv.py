import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import sys
import glob
import generateROImask as grm
import sct_analysis as sct

def read_csv():
    directory = 'C:/Users/10376/Documents/ndt_excel/'

    files = glob.glob(directory + '*.csv')
    print(files)
    for filename in files:
        savename = filename.replace(filename[-3:], 'npy')
        print(savename)

        data = np.zeros([24, 36])

        with open(filename, 'rt') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            i = 0
            for csv_row in spamreader:
                if csv_row[0] == 'Pixel':
                    continue
                data[int(csv_row[1]), int(csv_row[2])] = float(csv_row[3])
                i += 1

        np.save(savename, data)
        print(i)
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

show_fig()