import os
import numpy as np
import matplotlib.pyplot as plt
import mask_functions as msk

# Obtain the smaller contrast vial masks

directory = r'D:\OneDrive - University of Victoria\Research\LDA Data\21-03-11_CT_AuNPs'
folder = 'phantom_scan_12'

zs = np.arange(7, 17)
num_vials = 5
num_slices = len(zs)

data = np.load(os.path.join(directory, folder, 'Norm CT', 'CT_norm.npy'))[0]

#
# # This is for the bottom of the vials
# masks = np.zeros((num_vials, num_slices, 576, 576))
# # Order is Vial 1, then go through all slices, then go to vial 2
# for i in range(num_vials):
#     for j, z in enumerate(zs):
#         print(f'Vial{i+1}, Slice{j+1}')
#         masks[i, j] = msk.single_pixels_mask(data[z])
#
# np.save(os.path.join(directory, folder, 'contrast_masks_Au.npy'), masks)

# Correct any masks if need be

# masks = np.load(os.path.join(directory, folder, 'contrast_masks_Au.npy'))
#
# # masks[1, 5] = msk.single_pixels_mask(data[zs[5]])
# # masks[2, 4] = msk.single_pixels_mask(data[zs[4]])
# temp = msk.phantom_ROIs(data[zs[0]], radius=12)
# for i in range(len(zs)):
#     masks[-1, i] = temp
#
# np.save(os.path.join(directory, folder, 'contrast_masks_Au.npy'), masks)
