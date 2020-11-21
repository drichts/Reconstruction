import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.io import savemat, loadmat
import mask_functions as mf

directory = r'D:\OneDrive - University of Victoria\Research\LDA Data'

data_folder = r'ct_720frames_0.25sproj_111220'

data = loadmat(os.path.join(directory, data_folder, 'CT', 'CT_proj180_1-60s.mat'))['ct_img']
#
mask = mf.single_circular_ROI(data[6, :, :, 14])
np.save(os.path.join(directory, 'mask.npy'), mask)

# for i in np.arange(1, 6):
#     data = loadmat(os.path.join(directory, data_folder, 'CT', 'proj360_' + str(i) + '-60s.mat'))['data']
#     fig = plt.figure(figsize=(8, 10))
#     plt.imshow(np.transpose(np.roll(data[6, :, 12, :], 20, axis=0)), vmin=0.1, vmax=1)
#     plt.title(str(i), fontsize=14)
#     plt.show()
#     plt.savefig(rf'D:\OneDrive - University of Victoria\Research\LDA Data\fig\sinogram\360_{i}_60s.png', dpi=fig.dpi)
#     plt.close()

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


fig, ax = plt.subplots(2, 3, figsize=(12, 8))
nums = [0.25, 0.5, 1]
fram = 360
mask = np.load(os.path.join(directory, 'mask.npy'))
for i, folder in enumerate(['ct_720frames_0.25sproj_111220', 'ct_720frames_0.5sproj_111220', 'ct_720frames_1sproj_111220']):
    data60 = loadmat(os.path.join(directory, folder, 'CT', f'CT_proj{fram//2}_5-60s.mat'))['ct_img']
    data300 = loadmat(os.path.join(directory, folder, 'CT', f'CT_proj{fram}_5-60s.mat'))['ct_img']
    ax[0, i].imshow(data60[3, 90:200, 90:200, 8], vmin=0.01, vmax=0.03, cmap='gray')
    ax[1, i].imshow(data300[3, 90:200, 90:200, 8], vmin=0.01, vmax=0.03, cmap='gray')

    ax[0, i].set_title(f'{fram//2} frames, {nums[i]*4}s per frame,\n 60 s airscan', fontsize=13)
    ax[1, i].set_title(f'{fram} frames, {nums[i]*2}s per frame,\n 60 s airscan', fontsize=13)

    ax[0, i].set_xlabel(f'Noise: {np.nanstd(data60[3, :, :, 8]*mask)/np.nanmean(data60[6, :, :, 14]*mask)*100:.3f}', fontsize=11)
    ax[1, i].set_xlabel(f'Noise: {np.nanstd(data300[3, :, :, 8]*mask)/np.nanmean(data300[6, :, :, 14]*mask)*100:.3f}', fontsize=11)

plt.subplots_adjust(wspace=0.4, bottom=0.1, top=0.95)
plt.show()
# plt.savefig(os.path.join(directory, 'fig', 'CT', f'frame_comp.png'), dpi=fig.dpi)

