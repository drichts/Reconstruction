import os
import numpy as np
import matplotlib.pyplot as plt
import general_functions as gen

directory = r'D:\OneDrive - University of Victoria\Research\LDA Data\pixel_corr_3mmCu_high_bins\Thresholds_95_100_105_110_115_120'


dark = np.load(r'D:\OneDrive - University of Victoria\Research\LDA Data\21-04-14_CT_bin_width_5\darkscan_60s\Data\data.npy')/240

air1 = np.load(os.path.join(directory, '3mA_pixel_corr_Thresholds_95_100_105_110_115_120_EC_CC', 'Data', 'data.npy'))
# air2 = np.load(os.path.join(directory, folder2, 'Data', 'data.npy'))
air2 = np.sum(air1[201:], axis=0)
air1 = np.sum(air1[2:201], axis=0)

# dpm = np.load('/home/knoll/LDAData/dead_pixel_mask_2.npy')

# air1 = gen.correct_dead_pixels(air1, dpm)
# air2 = gen.correct_dead_pixels(air2, dpm)
# dark = gen.correct_dead_pixels(dark, dpm)

air1 = air1 - dark
air2 = air2 - dark
num = 2
corr = np.abs(np.log(air1) - np.log(air2)) * 100
# fig = plt.figure(figsize=(12, 4))
# plt.imshow(corr[:, :, num], vmin=1, vmax=2)
# plt.show()

base = np.array(np.argwhere(corr > 3), dtype='int')
base_nan = np.array(np.argwhere(np.isnan(corr)), dtype='int')

dpm = np.ones((24, 576))
dpm[9, 75] = np.nan
for pixel in base:
    pixel = tuple(pixel[:-1])
    dpm[pixel] = np.nan
for pixel in base_nan:
    pixel = tuple(pixel[:-1])
    dpm[pixel] = np.nan

print(len(np.argwhere(np.isnan(dpm)))/(24*576)*100)

np.save(os.path.join(directory, 'dead_pixel_mask_width5.npy'), dpm)

fig = plt.figure(figsize=(14, 3))
plt.imshow(dpm)
fig.show()
