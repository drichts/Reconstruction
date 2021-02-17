import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import general_functions as gen
import mask_functions as msk

file = r'D:\OneDrive - University of Victoria\Research\LDA Data'

# folder1 = 'CT_26-01-21_1800'
# folder2 = '10-cm_25-01-21_2'
#
# data1 = np.load(os.path.join(file, folder1, 'airscan_60s', 'Data', 'data_corr.npy'))
# data2 = np.load(os.path.join(file, folder2, 'airscan_60s', 'Data', 'data.npy'))

folder = r'D:\OneDrive - University of Victoria\Research\LDA Data\CT_02-03-21-v2'
data1 = np.load(os.path.join(folder, 'recon_CGLS.npy'))
data2 = np.load(os.path.join(folder, 'recon_SIRT.npy'))

for i in range(len(data2)):
    fig = plt.figure(figsize=(5, 5))
    plt.imshow(data2[i], cmap='gray', vmin=0.002, vmax=0.008)
    plt.title(f'{(i+1)*10} iterations')
    plt.show()
    plt.savefig(os.path.join(folder, f'SIRT_{(i+1)*10}.png'), dpi=fig.dpi)
    # plt.pause(10)
    plt.close()

# data = data2 - data1
# data_other = data1 - data2

# corr = np.log(data1) - np.log(data2)
# dpm = np.load(os.path.join(file, 'dead_pixel_mask.npy'))
# dpm[9, 75] = np.nan
# np.save(os.path.join(file, 'dead_pixel_mask.npy'), dpm)
# data2 = gen.correct_dead_pixels(data2, dpm)

# np.save(os.path.join(file, folder2, 'airscan_60s', 'Data', 'data_corr.npy'), data2)

# plt.imshow(data2[:, :, 6], vmin=1E8, vmax=4E8)
# fig, ax = plt.subplots(5, 1, figsize=(12, 12))
# ax[0].imshow(data[4, :, :, 6], vmin=0, vmax=0.09)
# ax[1].imshow(data[40, :, :, 6], vmin=0, vmax=0.09)
# ax[2].imshow(data[80, :, :, 6], vmin=0, vmax=0.09)
# ax[3].imshow(data[120, :, :, 6], vmin=0, vmax=0.09)
# ax[4].imshow(data[160, :, :, 6], vmin=0, vmax=0.09)
# plt.show()
