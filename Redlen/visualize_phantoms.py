import numpy as np
import matplotlib.pyplot as plt
import glob
import generateROImask as grm

directory = 'C:/Users/10376/Documents/Phantom Data/Uniformity/Multiple Energy Thresholds/1w/'

folders = ['Data', '3x3 Data']


def get_masks(name='4x4'):
    # Get masks for the folders
    directory = 'C:/Users/10376/Documents/Phantom Data/Uniformity/Multiple Energy Thresholds/1w/'
    folders = [name + ' Data']

    for folder in folders:
        data = np.load(directory + folder + '/Thresholds_1.npy')
        data = np.squeeze(np.sum(data, axis=2))

        continue_flag = True
        while continue_flag:
            mask = grm.single_pixels_mask(data[12])
            val = input('Were the ROIs acceptable? (y/n)')
            if val is 'y':
                continue_flag = False

        continue_flag = True
        while continue_flag:
            bg = grm.single_pixels_mask(data[12])
            val = input('Were the ROIs acceptable? (y/n)')
            if val is 'y':
                continue_flag = False

        np.save(directory + name + '_a0_Mask.npy', mask)
        np.save(directory + name + '_a0_Background.npy', bg)

        np.save('C:/Users/10376/Documents/Phantom Data/Uniformity/Multiple Energy Thresholds/3w/' + name + '_a0_Mask.npy', mask)
        np.save('C:/Users/10376/Documents/Phantom Data/Uniformity/Multiple Energy Thresholds/3w/' + name + '_a0_Background.npy', bg)


def show_imgs():
    # Plot the 5 bins plus the EC bin for any folder, with auto corrected data
    folder = folders[0]
    titl = 'Blue belt in acrylic 330 um'
    data = np.load(directory + folder + '/Corrected Data/Run008_a0.npy')

    fig, axes = plt.subplots(2, 3, figsize=(8, 6))
    ax1 = fig.add_subplot(111, frameon=False)
    ax1.grid(False)
    ax1.set_xticks([])
    ax1.set_yticks([])

    data = np.sum(data, axis=2)
    data = np.squeeze(data)

    titles = ['20-30 keV', '30-50 keV', '50-70 keV', '70-90 keV', '90-120 keV', '20-120 keV']

    for i, ax in enumerate(axes.flat):
        if i == 5:
            ax.imshow(np.sum(data[6:12], axis=0)[:, 0:35])
        else:
            ax.imshow(data[i+6, :, 0:35])
        ax.set_title(titles[i])
        ax.set_xticks([])
        ax.set_yticks([])

    ax1.set_title(titl, fontsize=15, pad=10)
    plt.show()



