import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import mask_functions as msk

directory = '/home/knoll/LDAData'
folder = '21-02-26_CT_resolution'

# data = np.load(r'D:\OneDrive - University of Victoria\Research\LDA Data\21-03-11_CT_AuNPs\phantom_scan_1\Norm CT\CT_norm.npy')[2]
# data2 = np.load(r'D:\OneDrive - University of Victoria\Research\LDA Data\21-03-11_CT_AuNPs\phantom_scan_2\Norm CT\CT_norm.npy')[2]
# data3 = np.load(r'D:\OneDrive - University of Victoria\Research\LDA Data\21-03-11_CT_AuNPs\phantom_scan_3\Norm CT\CT_norm.npy')[2]
# data5 = np.load(r'D:\OneDrive - University of Victoria\Research\LDA Data\21-03-11_CT_AuNPs\phantom_scan_5\Norm CT\CT_norm.npy')[2]
data = np.load(r'D:\OneDrive - University of Victoria\Research\LDA Data\21-03-11_CT_AuNPs\phantom_scan_12\Norm CT\K-edge_Au.npy')

# fig, ax = plt.subplots(1, 4, figsize=(12, 4))
# ax[0].imshow(data[12], cmap='gray', vmin=-400, vmax=400)
# ax[1].imshow(data2[12], cmap='gray', vmin=-400, vmax=400)
# ax[2].imshow(data3[12], cmap='gray', vmin=-400, vmax=400)
# ax[3].imshow(data5[12], cmap='gray', vmin=-400, vmax=400)
# fig.show()

for i in range(9, 15):
    fig = plt.figure(figsize=(8, 8))
    plt.imshow(data[i], cmap='gray', vmin=0, vmax=24)
    plt.title(f'{i}')
    plt.show()
    plt.pause(5)
    plt.close()
    # fig.savefig(fr'D:\OneDrive - University of Victoria\Research\LDA Data\21-03-11_CT_AuNPs\phantom_scan_12\fig\K_slice{i}.png', dpi=fig.dpi)

# plt.imshow(data[10], cmap='gray', vmin=-400, vmax=400)
# plt.show()



#
# def get_ROI_vals(mtf_data, num_patterns):
#     """
#     Get the ROIs for the phantom material, background material, and the various patterns
#     :return:
#     """
#
#     # Get the phantom material
#     phantom = msk.square_ROI(mtf_data)
#
#     # Get the background material
#     background = msk.square_ROI(mtf_data)
#
#     # Get the pattern rois
#     std_patt = np.zeros(num_patterns)
#     for i in range(num_patterns):
#         mask = msk.square_ROI(mtf_data)
#         std_patt[i] = np.nanstd(mask * mtf_data)
#
#     contrast = np.abs(np.nanmean(mtf_data * phantom) - np.nanmean(mtf_data * background))
#     p_noise = np.nanstd(mtf_data * phantom)
#     b_noise = np.nanstd(mtf_data * background)
#
#     np.savez(os.path.join(directory, folder, 'phantom_scan', 'mtf_contrast_vals.npz'), {'contrast': contrast, 'std_1': p_noise,
#                                                                   'std_2': b_noise})
#     np.save(os.path.join(directory, folder, 'phantom_scan', 'mtf_pattern_std.npy'), std_patt)
#
#
# data = np.load(os.path.join(directory, folder, 'phantom_scan', 'Norm CT', 'CT_norm.npy'))[12]
# num_pattern = 6
# get_ROI_vals(data, num_pattern)


# folder1 = '21-02-26_CT_min_Au_SEC'
# folder2 = '21-03-01_CT_min_Gd_3mA_SEC'
#
# data1_ct = np.load(os.path.join(directory, folder1, 'phantom_scan', 'Norm CT', 'CT_norm.npy'))[12]
# data1_k = np.load(os.path.join(directory, folder1, 'phantom_scan', 'Norm CT', 'K-edge.npy'))[12]
#
# data2_ct = np.load(os.path.join(directory, folder2, 'phantom_scan', 'Norm CT', 'CT_norm.npy'))[12]
# data2_k = np.load(os.path.join(directory, folder2, 'phantom_scan', 'Norm CT', 'K-edge.npy'))[12]
#
# water1 = np.load(os.path.join(directory, folder1, 'phantom_scan', 'water_mask.npy'))
# air1 = np.load(os.path.join(directory, folder1, 'phantom_scan', 'air_mask.npy'))
# contrast1 = np.load(os.path.join(directory, folder1, 'phantom_scan', 'contrast_masks.npy'))
#
# water2 = np.load(os.path.join(directory, folder2, 'phantom_scan', 'water_mask.npy'))
# air2 = np.load(os.path.join(directory, folder2, 'phantom_scan', 'air_mask.npy'))
# contrast2 = np.load(os.path.join(directory, folder2, 'phantom_scan', 'contrast_masks.npy'))
#
# print(f'Mean water CC: {np.nanmean(data1_ct*water1)}, {np.nanstd(data1_ct*water1)}')
# print(f'Mean water SEC: {np.nanmean(data2_ct*water2)}, {np.nanstd(data2_ct*water2)}')
# print()
# print(f'Mean air CC: {np.nanmean(data1_ct*air1)}, {np.nanstd(data1_ct*air1)}')
# print(f'Mean air SEC: {np.nanmean(data2_ct*air2)}, {np.nanstd(data2_ct*air2)}')
# print()
#
# for i in range(6):
#     print(f'Mean contrast {i} CC: {np.nanmean(data1_k * contrast1[i])}, CNR: {(np.nanmean(data1_k * contrast1[i]) - np.nanmean(data1_k*water1))/np.nanstd(data1_k* water1)}')
#     print(f'Mean constrast {i} SEC: {np.nanmean(data2_k * contrast2[i])}, CNR: {(np.nanmean(data2_k * contrast2[i]) - np.nanmean(data2_k*water2))/np.nanstd(data2_k* water2)}')
#     print()
#
# print(f'Mean contrast water CC: {np.nanmean(data1_k*water1)}')
# print(f'Mean constrast water SEC: {np.nanmean(data2_k*water2)}')
#
# fig, ax = plt.subplots(2, 2, figsize=(10, 10))
# ax[0, 0].imshow(data1_ct)
# ax[0, 1].imshow(data2_ct)
#
# ax[1, 0].imshow(data1_k)
# ax[1, 1].imshow(data2_k)
# plt.show(block=True)