import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from pixel_correction.non_uniformity_corr import di

# Directory
directory = r'D:\OneDrive - University of Victoria\Research\LDA Data'
folder = r'pixel_corr_3mmCu_high_bins\Thresholds_70_80_90_100_110_120'
fluxes = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]  # List of the fluxes
expos_time = 0.25  # Exposure time in seconds

dark = np.load(os.path.join(directory, r'21-04-14_CT_bin_width_10\darkscan_65s_Thresholds_70_80_90_100_110_120_EC_CC', 'Data', 'data.npy'))
dark = np.sum(dark[1:], axis=0)/240

# dead pixel mask
dpm = np.load(os.path.join(directory, 'dead_pixel_mask_4.npy'))

# Array to hold mean and std of the di values of each bin and each pair
di_vals = np.zeros((2, 7, len(fluxes)-1))

# Go through each pair of fluxes and calculate di
for i in np.arange(len(fluxes)-1):
    folder1 = glob(os.path.join(directory, folder, f'*{fluxes[i]}mA*'))[0]
    folder2 = glob(os.path.join(directory, folder, f'*{fluxes[i+1]}mA*'))[0]

    data1 = np.load(os.path.join(folder1, 'Data', 'data.npy')) - dark
    data2 = np.load(os.path.join(folder2, 'Data', 'data.npy')) - dark

    # Put bin axes second <frames, bins, rows, columns>
    data1 = np.transpose(data1, axes=[0, 3, 1, 2])
    data2 = np.transpose(data2, axes=[0, 3, 1, 2])

    # Correct for dead pixels
    data1 *= dpm
    data2 *= dpm

    # Calculate lambda (average count in each pixel)
    data1 = np.nanmean(data1, axis=0)
    data2 = np.nanmean(data2, axis=0)

    # Calculate di
    di_temp = di(data1, data2, fluxes[i], fluxes[i+1], expos_time)

    di_vals[0, :, i] = np.nanmean(di_temp, axis=(1, 2))  # Mean value over all pixels in each bin
    di_vals[1, :, i] = np.nanstd(di_temp, axis=(1, 2))  # Std over all pixels in each bin

fig = plt.figure(figsize=(7, 7))
plt.errorbar(np.arange(len(di_vals[0])), di_vals[0, 6], yerr=di_vals[1, 6], capsize=4)
plt.xlabel('Flux pair')
plt.ylabel('di')
plt.show()

