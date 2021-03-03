import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import mask_functions as msk

directory = '/home/knoll/LDAData'
folder = '21-02-26_CT_resolution'


def get_ROI_vals(mtf_data, num_patterns):
    """
    Get the ROIs for the phantom material, background material, and the various patterns
    :return:
    """

    # Get the phantom material
    phantom = msk.square_ROI(mtf_data)
    np.save(os.path.join(directory, folder, 'mtf_bar.npy'), phantom)

    # Get the background material
    background = msk.square_ROI(mtf_data)
    np.save(os.path.join(directory, folder, 'mtf_background.npy'), background)

    # Get the pattern rois
    std_patt = np.zeros(num_patterns)
    masks = np.zeros((num_patterns, *np.shape(mtf_data)))
    for i in range(num_patterns):
        masks[i] = msk.square_ROI(mtf_data)
        std_patt[i] = np.nanstd(masks[i] * mtf_data)
    np.save(os.path.join(directory, folder, 'mtf_masks.npy'), masks)

    contrast = np.abs(np.nanmean(mtf_data * phantom) - np.nanmean(mtf_data * background))
    p_noise = np.nanstd(mtf_data * phantom)
    b_noise = np.nanstd(mtf_data * background)

    np.savez(os.path.join(directory, folder, 'mtf_contrast_vals.npz'), contrast=contrast, std_1=p_noise, std_2=b_noise)
    np.save(os.path.join(directory, folder, 'mtf_pattern_std.npy'), std_patt)


data = np.load(os.path.join(directory, folder, 'phantom_scan', 'Norm CT', 'CT_norm.npy'))[12]
num_pattern = 6
get_ROI_vals(data, num_pattern)


# folder1 = '21-02-26_CT_min_Au_SEC'
# folder2 = '21-03-01_CT_min_Gd_3mA_SEC'
# alg = '_FDK'
# z = 13
# data1_ct = np.load(os.path.join(directory, folder1, 'phantom_scan', 'Norm CT', f'CT_norm{alg}.npy'))[z]
# data1_k = np.load(os.path.join(directory, folder1, 'phantom_scan', 'Norm CT', f'K-edge{alg}.npy'))[z]
#
# data2_ct = np.load(os.path.join(directory, folder2, 'phantom_scan', 'Norm CT', f'CT_norm{alg}.npy'))[z]
# data2_k = np.load(os.path.join(directory, folder2, 'phantom_scan', 'Norm CT', f'K-edge{alg}.npy'))[z]
#
# water1 = np.load(os.path.join(directory, folder1, 'phantom_scan', f'water_mask{alg}.npy'))
# air1 = np.load(os.path.join(directory, folder1, 'phantom_scan', f'air_mask{alg}.npy'))
# contrast1 = np.load(os.path.join(directory, folder1, 'phantom_scan', f'contrast_masks{alg}.npy'))
#
# water2 = np.load(os.path.join(directory, folder2, 'phantom_scan', f'water_mask{alg}.npy'))
# air2 = np.load(os.path.join(directory, folder2, 'phantom_scan', f'air_mask{alg}.npy'))
# contrast2 = np.load(os.path.join(directory, folder2, 'phantom_scan', f'contrast_masks{alg}.npy'))
#
# print(f'Mean water Au: {np.nanmean(data1_ct*water1)}, {np.nanstd(data1_ct*water1)}')
# print(f'Mean water Gd: {np.nanmean(data2_ct*water2)}, {np.nanstd(data2_ct*water2)}')
# print()
# print(f'Mean air Au: {np.nanmean(data1_ct*air1)}, {np.nanstd(data1_ct*air1)}')
# print(f'Mean air Gd: {np.nanmean(data2_ct*air2)}, {np.nanstd(data2_ct*air2)}')
# print()
#
# for i in range(6):
#     print(f'Mean contrast {i} Au: {np.nanmean(data1_k * contrast1[i])}, CNR: {(np.nanmean(data1_k * contrast1[i]) - np.nanmean(data1_k*water1))/np.nanstd(data1_k* water1)}')
#     print(f'Mean constrast {i} Gd: {np.nanmean(data2_k * contrast2[i])}, CNR: {(np.nanmean(data2_k * contrast2[i]) - np.nanmean(data2_k*water2))/np.nanstd(data2_k* water2)}')
#     print()
#
# print(f'Mean contrast water Au: {np.nanmean(data1_k*water1)}')
# print(f'Mean constrast water Gd: {np.nanmean(data2_k*water2)}')
#
# fig, ax = plt.subplots(2, 2, figsize=(10, 10))
# ax[0, 0].imshow(data1_ct)
# ax[0, 1].imshow(data2_ct)
#
# ax[1, 0].imshow(data1_k)
# ax[1, 1].imshow(data2_k)
# plt.show(block=True)



