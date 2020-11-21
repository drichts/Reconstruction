import numpy as np
from scipy.io import loadmat
import os
import matplotlib.colors as clr
import matplotlib.pyplot as plt

#%%
def image_noise_with_bin(folder, method='water', directory='D:/Research/Python Data/Spectral CT/'):
    """
    This function loads the image noise array created in sCT_Analysis and calculates the maximum noise in each bin
    It outputs the x and y coordinates to graph
    :param folder: scan folder
    :param method: 'water' or 'phantom' (which noise ROI to use)
    :param directory: where all scan folders live
    :return: x (bin) and y (noise) coordinates to graph
    """
    path = directory + folder

    noise_matrix = np.load(path + '/Image_Noise_' + method + '.npy')
    max_noise = np.empty(6)

    # Find the maximum noise in a slice per bin
    for i in np.arange(5):
        temp_noise = np.nanmax(noise_matrix[i])
        max_noise[i] = temp_noise
    max_noise[5] = np.nanmax(noise_matrix[6])

    bins = np.array([0, 1, 2, 3, 4, 6])

    return bins, max_noise

#%%
def air_flux_with_bin(folder, bins=[16, 50, 54, 64, 81, 120], directory='D:/Research/Python Data/Spectral CT/'):

    flux = np.load(directory+folder+'/Airscan_Flux.npy')
    #print(flux)
    # Get rid of bin 5 and 6
    flux = np.delete(flux, [5, 6])
    #flux = np.append(flux, [0])
    #widths = np.array([bins[1]-bins[0], bins[2]-bins[1], bins[3]-bins[2], bins[4]-bins[3], bins[5]-bins[4], 1])
    bins = np.array([0, 1, 2, 3, 4])

    return bins, flux

#%%
def all_images(folder, slice=14, low=-500, high=1000, bins=[16, 50, 54, 64, 81, 120],
               directory='D:/Research/Python Data/Spectral CT/'):

    slice = str(slice)
    path = directory + folder + '/Slices/'
    img0 = np.load(path + 'Bin0_Slice' + slice + '.npy')
    img1 = np.load(path + 'Bin1_Slice' + slice + '.npy')
    img2 = np.load(path + 'Bin2_Slice' + slice + '.npy')
    img3 = np.load(path + 'Bin3_Slice' + slice + '.npy')
    img4 = np.load(path + 'Bin4_Slice' + slice + '.npy')
    img6 = np.load(path + 'Bin6_Slice' + slice + '.npy')

    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    plt.setp(ax, xticks=[4, 60, 116], xticklabels=['-1.5', '0.0', '1.5'],
             yticks=[4, 60, 116], yticklabels=['-1.5', '0.0', '1.5'])

    ax[0, 0].imshow(img0, cmap='gray', vmin=low, vmax=high)
    ax[0, 0].set_title(str(bins[0]) + '-' + str(bins[1]) + ' keV', fontsize=25)
    ax[0, 0].tick_params(labelsize=25)
    ax[0, 0].set_xlabel('x (cm)', fontsize=25)
    ax[0, 0].set_ylabel('y (cm)', fontsize=25)

    ax[0, 1].imshow(img1, cmap='gray', vmin=low, vmax=high)
    ax[0, 1].set_title(str(bins[1]) + '-' + str(bins[2]) + ' keV', fontsize=25)
    ax[0, 1].tick_params(labelsize=25)
    ax[0, 1].set_xlabel('x (cm)', fontsize=25)
    ax[0, 1].set_ylabel('y (cm)', fontsize=25)

    ax[0, 2].imshow(img2, cmap='gray', vmin=low, vmax=high)
    ax[0, 2].set_title(str(bins[2]) + '-' + str(bins[3]) + ' keV', fontsize=25)
    ax[0, 2].tick_params(labelsize=25)
    ax[0, 2].set_xlabel('x (cm)', fontsize=25)
    ax[0, 2].set_ylabel('y (cm)', fontsize=25)

    ax[1, 0].imshow(img3, cmap='gray', vmin=low, vmax=high)
    ax[1, 0].set_title(str(bins[3]) + '-' + str(bins[4]) + ' keV', fontsize=25)
    ax[1, 0].tick_params(labelsize=25)
    ax[1, 0].set_xlabel('x (cm)', fontsize=25)
    ax[1, 0].set_ylabel('y (cm)', fontsize=25)

    ax[1, 1].imshow(img4, cmap='gray', vmin=low, vmax=high)
    ax[1, 1].set_title(str(bins[4]) + '-' + str(bins[5]) + ' keV', fontsize=25)
    ax[1, 1].tick_params(labelsize=25)
    ax[1, 1].set_xlabel('x (cm)', fontsize=25)
    ax[1, 1].set_ylabel('y (cm)', fontsize=25)

    ax[1, 2].imshow(img6, cmap='gray', vmin=low, vmax=high)
    ax[1, 2].set_title(str(bins[0]) + '-' + str(bins[5]) + ' keV', fontsize=25)
    ax[1, 2].tick_params(labelsize=25)
    ax[1, 2].set_xlabel('x (cm)', fontsize=25)
    ax[1, 2].set_ylabel('y (cm)', fontsize=25)
    plt.subplots_adjust(top=0.94, bottom=0.125, left=0.125, right=0.88, hspace=0.45, wspace=0.2)
    #plt.savefig(directory + folder + '/AllBins_save.svg', dpi=1000)
    plt.show()

#%% Graph noise to compare

directory='D:/Research/Python Data/Spectral CT/'
folder1 = 'AuGd_width_5_12-2-19/'
folder2 = 'AuGd_width_10_12-2-19/'
folder3 = 'AuGd_width_14_12-2-19/'
title = 'Noise: Bin Width'
lgnd = ['5 keV', '10 keV', '14 keV']
energies = ['16-50', '50-54', '54-64', '64-81', '81-120', '16-120']
bins = [0, 1, 2, 3, 4, 5]

folders = [folder1, folder2, folder3]

fig = plt.figure(figsize=(12, 6))

clrs = ['black', 'red', 'blue']

for i, folder in enumerate(folders):

    path = directory + folder
    mean_noise = np.load(path + 'Mean_Noise.npy')
    std_noise = np.load(path + 'Noise_Error.npy')
    mean_noise = np.delete(mean_noise, 5)
    std_noise = np.delete(std_noise, 5)

    plt.errorbar(bins, mean_noise, yerr=std_noise, lw=3, color=clrs[i], capsize=10)

plt.title(title, fontsize=20)

plt.xlabel('Bin Energy Range (keV)', fontsize=18)
plt.ylabel('HU', fontsize=18)
plt.xticks([0, 1, 2, 3, 4, 5], labels=energies, fontsize=16)
plt.yticks(fontsize=16)
plt.legend(lgnd, fontsize=18, fancybox=True, shadow=True)
plt.show()


#%%

