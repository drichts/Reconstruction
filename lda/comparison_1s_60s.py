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
# mask = mf.single_circular_ROI(data[6, :, :, 14])
# np.save(os.path.join(directory, 'mask.npy'), mask)

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

# fig, ax = plt.subplots(2, 3, figsize=(12, 8))
# slices = [8, 9, 14]
# folder = 'ct_180frames_1sproj_111220 - Synth'
# for i in range(3):
#     data60 = loadmat(os.path.join(directory, folder, 'CT', 'CT_proj_filt_no_dpc.mat'))['ct_img']
#     data300 = loadmat(os.path.join(directory, folder, 'CT', 'CT_proj_filt.mat'))['ct_img']
#     ax[0, i].imshow(data60[6, 90:200, 90:200, slices[i]], vmin=0.01, vmax=0.03, cmap='gray')
#     ax[1, i].imshow(data300[6, 90:200, 90:200, slices[i]], vmin=0.01, vmax=0.03, cmap='gray')
#
#     ax[0, i].set_title(f'With no dead pixel corr, Slice {slices[i]}', fontsize=13)
#     ax[1, i].set_title(f'With dead pixel corr, Slice {slices[i]}', fontsize=13)
#
# plt.subplots_adjust(wspace=0.4, bottom=0.1, top=0.95)
# plt.show()
# plt.savefig(os.path.join(directory, 'fig', 'CT', f'dpcorr11.png'), dpi=fig.dpi)

# fig, ax = plt.subplots(2, 3, figsize=(12, 8))
# slices = [8, 9, 14]
# folder = 'ct_180frames_1sproj_111220 - Synth2'
# for i in range(3):
#     data60 = loadmat(os.path.join(directory, folder, 'CT', 'CT_proj_filt_no_dpc.mat'))['ct_img']
#     data300 = loadmat(os.path.join(directory, folder, 'CT', 'CT_proj_filt.mat'))['ct_img']
#     ax[0, i].imshow(data60[6, 90:200, 90:200, slices[i]], vmin=0.01, vmax=0.03, cmap='gray')
#     ax[1, i].imshow(data300[6, 90:200, 90:200, slices[i]], vmin=0.01, vmax=0.03, cmap='gray')
#
#     ax[0, i].set_title(f'With no dead pixel corr, Slice {slices[i]}', fontsize=13)
#     ax[1, i].set_title(f'With dead pixel corr, Slice {slices[i]}', fontsize=13)
#
# plt.subplots_adjust(wspace=0.4, bottom=0.1, top=0.95)
# plt.show()
# plt.savefig(os.path.join(directory, 'fig', 'CT', f'dpcorr2.png'), dpi=fig.dpi)

#
# slices = [8, 9, 14]
# folder = 'ct_180frames_1sproj_111220 - Synth3'
# data = loadmat(os.path.join(directory, folder, 'CT', 'CT_proj_filt.mat'))['ct_img']
# # mask = mf.entire_phantom(data[2, :, :, 14], 10)
# # np.save(os.path.join(directory, 'mask.npy'), mask)
# mask = np.load(os.path.join(directory, 'mask.npy'))
#
# fig, ax = plt.subplots(2, 3, figsize=(12, 8))
# for i in range(3):
#
#     ax[0, i].imshow(data[2, 90:200, 90:200, slices[i]], vmin=0.01, vmax=0.03, cmap='gray')
#     ax[1, i].imshow(data[1, 90:200, 90:200, slices[i]], vmin=0.01, vmax=0.03, cmap='gray')
#
#     ax[0, i].set_title(f'20-120 keV, Slice {slices[i]}', fontsize=13)
#     ax[1, i].set_title(f'30-120 keV, Slice {slices[i]}', fontsize=13)
#
#     ax[0, i].set_xlabel(f'Rel. Noise: {np.nanstd(data[2, :, :, slices[i]] * mask) / np.nanmean(data[2, :, :, slices[i]] * mask) * 100:.3f}', fontsize=11)
#     ax[1, i].set_xlabel(f'Rel. Noise: {np.nanstd(data[1, :, :, slices[i]] * mask) / np.nanmean(data[1, :, :, slices[i]] * mask) * 100:.3f}', fontsize=11)
#
# plt.subplots_adjust(wspace=0.4, bottom=0.1, top=0.95)
# plt.show()
# plt.savefig(os.path.join(directory, 'fig', 'CT', f'add_bins1.png'), dpi=fig.dpi)


slices = [8, 9, 14]
folder = 'ct_180frames_1sproj_111220 - Synth4'
data = loadmat(os.path.join(directory, folder, 'CT', 'CT_proj_filt.mat'))['ct_img']
# mask = mf.entire_phantom(data[2, :, :, 14], 10)
# np.save(os.path.join(directory, 'mask.npy'), mask)
mask = np.load(os.path.join(directory, 'mask.npy'))

fig, ax = plt.subplots(2, 3, figsize=(12, 8))
for i in range(3):

    ax[0, i].imshow(data[3, 90:200, 90:200, slices[i]], vmin=0.01, vmax=0.08, cmap='gray')
    ax[1, i].imshow(data[2, 90:200, 90:200, slices[i]], vmin=0.01, vmax=0.08, cmap='gray')

    ax[0, i].set_title(f'20-120 keV, Slice {slices[i]}', fontsize=13)
    ax[1, i].set_title(f'50-120 keV, Slice {slices[i]}', fontsize=13)

    ax[0, i].set_xlabel(f'Rel. Noise: {np.nanstd(data[3, :, :, slices[i]] * mask) / np.nanmean(data[3, :, :, slices[i]] * mask) * 100:.3f}', fontsize=11)
    ax[1, i].set_xlabel(f'Rel. Noise: {np.nanstd(data[2, :, :, slices[i]] * mask) / np.nanmean(data[2, :, :, slices[i]] * mask) * 100:.3f}', fontsize=11)

plt.subplots_adjust(wspace=0.4, bottom=0.1, top=0.95)
plt.show()
# plt.savefig(os.path.join(directory, 'fig', 'CT', f'add_bins2.png'), dpi=fig.dpi)

# slices = [8, 9, 14]
# folder1 = 'ct_180frames_1sproj_111220 - Synth6'
# folder2 = 'ct_180frames_1sproj_111220 - Synth5'
# data1 = loadmat(os.path.join(directory, folder1, 'CT', 'CT_proj_filt.mat'))['ct_img']
# print(data1.shape)
# data2 = loadmat(os.path.join(directory, folder2, 'CT', 'CT_proj_filt.mat'))['ct_img']
# print(data2.shape)
# # mask = mf.entire_phantom(data[2, :, :, 14], 10)
# # np.save(os.path.join(directory, 'mask.npy'), mask)
# mask = np.load(os.path.join(directory, 'mask.npy'))

# fig, ax = plt.subplots(2, 3, figsize=(12, 8))
# for i in range(3):
#
#     ax[0, i].imshow(data2[1, 90:200, 90:200, slices[i]], vmin=0.01, vmax=0.03, cmap='gray')
#     ax[1, i].imshow(data1[2, 90:200, 90:200, slices[i]], vmin=0.01, vmax=0.03, cmap='gray')
#
#     ax[0, i].set_title(f'30-120 keV, Slice {slices[i]}', fontsize=13)
#     ax[1, i].set_title(f'50-120 keV, Slice {slices[i]}', fontsize=13)
#
#     ax[0, i].set_xlabel(f'Rel. Noise: {np.nanstd(data2[1, :, :, slices[i]] * mask) / np.nanmean(data2[1, :, :, slices[i]] * mask) * 100:.3f}', fontsize=11)
#     ax[1, i].set_xlabel(f'Rel. Noise: {np.nanstd(data1[2, :, :, slices[i]] * mask) / np.nanmean(data1[2, :, :, slices[i]] * mask) * 100:.3f}', fontsize=11)
#
# plt.subplots_adjust(wspace=0.4, bottom=0.1, top=0.95)
# plt.show()
# plt.savefig(os.path.join(directory, 'fig', 'CT', f'add_bins3_nopileup.png'), dpi=fig.dpi)


# slices = [8, 9, 14]
# folder1 = 'ct_180frames_1sproj_111220 - wwo dead pixel corr diff air and dark'
# folder2 = 'ct_180frames_1sproj_111220 - long darkscan'
# data1 = loadmat(os.path.join(directory, folder1, 'CT', 'CT_proj_filt.mat'))['ct_img']
# print(data1.shape)
# data2 = loadmat(os.path.join(directory, folder2, 'CT', 'CT_proj_filt.mat'))['ct_img']
# print(data2.shape)
# # mask = mf.entire_phantom(data[2, :, :, 14], 10)
# # np.save(os.path.join(directory, 'mask.npy'), mask)
# mask = np.load(os.path.join(directory, 'mask.npy'))
#
# fig, ax = plt.subplots(2, 3, figsize=(12, 8))
# for i in range(3):
#
#     ax[0, i].imshow(data1[6, 90:200, 90:200, slices[i]], vmin=0.01, vmax=0.03, cmap='gray')
#     ax[1, i].imshow(data2[6, 90:200, 90:200, slices[i]], vmin=0.01, vmax=0.03, cmap='gray')
#
#     ax[0, i].set_title(f'Short darkscan, Slice {slices[i]}', fontsize=13)
#     ax[1, i].set_title(f'Long darkscan, Slice {slices[i]}', fontsize=13)
#
#     ax[0, i].set_xlabel(f'Rel. Noise: {np.nanstd(data1[6, :, :, slices[i]] * mask) / np.nanmean(data1[6, :, :, slices[i]] * mask) * 100:.3f}', fontsize=11)
#     ax[1, i].set_xlabel(f'Rel. Noise: {np.nanstd(data2[6, :, :, slices[i]] * mask) / np.nanmean(data2[6, :, :, slices[i]] * mask) * 100:.3f}', fontsize=11)
#
# plt.subplots_adjust(wspace=0.4, bottom=0.1, top=0.95)
# plt.show()
# plt.savefig(os.path.join(directory, 'fig', 'CT', f'long_vs_short_darkscan.png'), dpi=fig.dpi)


