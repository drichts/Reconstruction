import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.io import savemat, loadmat

directory = r'D:\OneDrive - University of Victoria\Research\LDA Data'

data_folder = r'ct_test_110920\Data'
air_folder_60 = r'airscan_120kVP_1mA_1mmAl_3x8coll_60s\Data'
dark_folder_60 = r'darkscan_60s\Data'
air_folder_1 = r'airscan_120kVP_1mA_1mmAl_3x8coll_1s\Data'
dark_folder_1 = r'darkscan_1s\Data'

# x60 = loadmat(os.path.join(directory, data_folder, 'ct_60s.mat'))['ct_img']
# x1 = loadmat(os.path.join(directory, data_folder, 'ct_1s.mat'))['ct_img']
#
# np.save(os.path.join(directory, data_folder, 'ct_60s.npy'), x60)
# np.save(os.path.join(directory, data_folder, 'ct_1s.npy'), x1)

x60 = np.load(os.path.join(directory, data_folder, 'ct_60s.npy'))
x1 = np.load(os.path.join(directory, data_folder, 'ct_1s.npy'))

data = np.load(os.path.join(directory, data_folder, 'data.npy'))
air60 = np.load(os.path.join(directory, air_folder_60, 'data.npy'))/60
dark60 = np.load(os.path.join(directory, dark_folder_60, 'data.npy'))/60

air1 = np.load(os.path.join(directory, air_folder_1, 'data.npy'))
dark1 = np.load(os.path.join(directory, dark_folder_1, 'data.npy'))

proj60 = np.log(np.subtract(air60, dark60)) - np.log(np.subtract(data, dark60))
proj60 = proj60[:, 13, 50:250, 6]
proj60 = np.transpose(np.roll(proj60, 20, axis=0))

proj1 = np.log(np.subtract(air1, dark1)) - np.log(np.subtract(data, dark1))
proj1 = proj1[:, 13, 50:250, 6]
proj1 = np.transpose(np.roll(proj1, 20, axis=0))

# fig, ax = plt.subplots(1, 2, figsize=(14, 8))
# ax[0].imshow(proj1, vmin=0, vmax=1)
# ax[1].imshow(proj60, vmin=0, vmax=1)
# ax[0].set_title('1 s air and darkscans', fontsize=18)
# ax[1].set_title('60 s air and darkscans over 60', fontsize=18)
# plt.show()
# plt.savefig(os.path.join(directory, data_folder[:-5], 'fig', 'sinograms.png'), dpi=fig.dpi)

# air = np.squeeze(np.divide(air1, air60))
# air = air[:, :, 6]  # [:, 50:225, 6]
# mask1 = np.array(air < 0.9, dtype=int)
# mask2 = np.array(air > 1, dtype=int)
# mask = mask1 + mask2
# mask[:, 0:50] = 0
# mask[:, 240:] = 0
#
# mask[mask == 0] = 2
# mask[mask == 1] = 0
# mask[mask == 2] = 1
#
# savemat(os.path.join(directory, data_folder, 'dp_mask.mat'), {'data': mask, 'label': 'dead'})

# fig = plt.figure(figsize=(14, 3))
# # plt.imshow(air*100, vmin=50, vmax=100)
# plt.imshow(mask)
# plt.xlabel('Relative Response (%)', fontsize=15, labelpad=55)
# plt.colorbar(orientation='horizontal')
# plt.title('Darkscan, 1 s over 60s/60', fontsize=18)
# plt.subplots_adjust(top=1, bottom=0.2)
# plt.show()
# plt.savefig(os.path.join(directory, data_folder[:-5], 'fig', 'airscans.png'), dpi=fig.dpi)

# dark = np.squeeze(np.divide(dark1, dark60))
# fig = plt.figure(figsize=(14, 3))
# plt.imshow(dark[:, :, 6]*100, vmin=0, vmax=1200)
# plt.title('Darkscan, 1 s over 60s/60', fontsize=18)
# plt.xlabel('Relative Response (%)', fontsize=15, labelpad=55)
# plt.colorbar(orientation='horizontal')
# plt.subplots_adjust(top=1, bottom=0.2)
# plt.show()
# plt.savefig(os.path.join(directory, data_folder[:-5], 'fig', 'darkscans.png'), dpi=fig.dpi)

slices = [12, 14, 15]

fig, ax = plt.subplots(2, 3, figsize=(12, 8))
ax[0, 0].imshow(x1[6, 85:200, 85:200, slices[0]], vmin=0.01, vmax=0.1, cmap='gray')
ax[0, 1].imshow(x1[6, 85:200, 85:200, slices[1]], vmin=0.01, vmax=0.1, cmap='gray')
ax[0, 2].imshow(x1[6, 85:200, 85:200, slices[2]], vmin=0.01, vmax=0.1, cmap='gray')
ax[1, 0].imshow(x60[6, 85:200, 85:200, slices[0]], vmin=0.01, vmax=0.1, cmap='gray')
ax[1, 1].imshow(x60[6, 85:200, 85:200, slices[1]], vmin=0.01, vmax=0.1, cmap='gray')
ax[1, 2].imshow(x60[6, 85:200, 85:200, slices[2]], vmin=0.01, vmax=0.1, cmap='gray')
ax[0, 0].set_title(f'1s Slice {slices[0]}', fontsize=13)
ax[0, 1].set_title(f'1s Slice {slices[1]}', fontsize=13)
ax[0, 2].set_title(f'1s Slice {slices[2]}', fontsize=13)
ax[1, 0].set_title(f'60s Slice {slices[0]}', fontsize=13)
ax[1, 1].set_title(f'60s Slice {slices[1]}', fontsize=13)
ax[1, 2].set_title(f'60s Slice {slices[2]}', fontsize=13)

plt.show()
plt.savefig(os.path.join(directory, data_folder[:-5], 'fig', 'ct_images.png'), dpi=fig.dpi)

