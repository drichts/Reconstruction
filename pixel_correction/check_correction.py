import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from pixel_correction.non_uniformity_corr import linear_range_corr
from general_functions import correct_dead_pixels

# Directory
directory = r'D:\OneDrive - University of Victoria\Research\LDA Data'
dpm = np.load(os.path.join(directory, 'dead_pixel_mask_4.npy'))


corr_folder = r'pixel_corr_3mmCu_high_bins\Thresholds_70_80_90_100_110_120'

fluxes = [2.5, 3]  # List of the fluxes
expos_time = 0.25  # Exposure time in seconds

# Get the correction data
corr1_folder = glob(os.path.join(directory, corr_folder, f'*{fluxes[0]}mA*'))[0]
corr2_folder = glob(os.path.join(directory, corr_folder, f'*{fluxes[1]}mA*'))[0]
# corr1 = np.load(os.path.join(corr1_folder, 'Data', 'data.npy'))
# corr2 = np.load(os.path.join(corr2_folder, 'Data', 'data.npy'))

# Get the actual data
data_folder = r'21-04-14_CT_bin_width_10\water_phantom'
# data = np.load(os.path.join(directory, data_folder, 'Data', 'data.npy'))[10:730]

# Get the air data as well
air_folder = r'21-04-14_CT_bin_width_10\airscan_65s_Thresholds_70_80_90_100_110_120_EC_CC'
# air = np.load(os.path.join(directory, air_folder, 'Data', 'data.npy'))

# Get the dark scan
# dark = np.load(os.path.join(directory, r'21-04-14_CT_bin_width_10\darkscan_65s_Thresholds_70_80_90_100_110_120_EC_CC',
#                             'Data', 'data.npy'))
# dark = np.sum(dark[1:], axis=0)  # This is 60s

# Correct each for the darkscan
# corr1 -= (dark/(60/expos_time))
# corr2 -= (dark/(60/expos_time))
# data -= (dark/(60/expos_time))
# air -= (dark/(60/5))
#
# # Correct the data and the airscan
# data_corr = linear_range_corr(corr1, corr2, fluxes[0], fluxes[1], expos_time, expos_time, data, dpm)
# air_corr = linear_range_corr(corr1, corr2, fluxes[0], fluxes[1], expos_time, 5, air, dpm)
#
# air = np.sum(air, axis=0)/240
# air_corr = np.sum(air_corr, axis=0)/240
#
# # Correct all data for the dead pixels
# data = correct_dead_pixels(data, dpm)
# air = correct_dead_pixels(air, dpm)
# data_corr = correct_dead_pixels(data_corr, dpm)
# air_corr = correct_dead_pixels(air_corr, dpm)

data = np.load(os.path.join(directory, data_folder, 'Data', 'data_dpm.npy'))
data_corr = np.load(os.path.join(directory, data_folder, 'Data', 'data_dpm_corr.npy'))
air = np.load(os.path.join(directory, air_folder, 'Data', 'data_dpm.npy'))
air_corr = np.load(os.path.join(directory, air_folder, 'Data', 'data_dpm_corr.npy'))

# # Do the airscan correction for both non-correction and corrected data
sino_non_corr = np.log(air) - np.log(data)
sino_corr = np.log(air_corr) - np.log(data_corr)
#
# np.save(os.path.join(directory, data_folder, 'Data', 'data_dpm.npy'), data)
# np.save(os.path.join(directory, data_folder, 'Data', 'data_dpm_corr.npy'), data_corr)
# np.save(os.path.join(directory, air_folder, 'Data', 'data_dpm.npy'), air)
# np.save(os.path.join(directory, air_folder, 'Data', 'data_dpm_corr.npy'), air_corr)

fig, ax = plt.subplots(2, 1, figsize=(12, 10))
ax[0].axis('off')
ax[1].axis('off')

ax[0].imshow(np.rot90(sino_non_corr[:, 12, 138:430, 6]))
ax[0].set_title('Non-corrected')

ax[1].imshow(np.rot90(sino_corr[:, 12, 138:430, 6]))
ax[1].set_title('Corrected')

plt.show()
