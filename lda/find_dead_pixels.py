import os
import numpy as np
import matplotlib.pyplot as plt
import general_functions as gen

directory = r'D:\OneDrive - University of Victoria\Research\LDA Data\21-03-31_CT_AuNPs_2'

folder1 = 'airscan_60s'
folder2 = 'airscan2_60s'

dark = np.load(os.path.join(directory, r'darkscan_60s\Data\data.npy'))

air1 = np.load(os.path.join(directory, folder1, 'Data', 'data.npy'))
air2 = np.load(os.path.join(directory, folder2, 'Data', 'data.npy'))

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

base = np.array(np.argwhere(corr > 2), dtype='int')
base_nan = np.array(np.argwhere(np.isnan(corr)), dtype='int')

dpm = np.ones((24, 576, 7))

for pixel in base:
    pixel = tuple(pixel)
    dpm[pixel] = np.nan
for pixel in base_nan:
    pixel = tuple(pixel)
    dpm[pixel] = np.nan

np.save(os.path.join(directory, 'dead_pixel_mask_3.npy'), dpm)

fig, ax = plt.subplots(7, 1, figsize=(8, 8))
for i in range(7):
    ax[i].imshow(dpm[:, :, i])
fig.show()
