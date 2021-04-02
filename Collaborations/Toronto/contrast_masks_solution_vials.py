import os
import numpy as np
import matplotlib.pyplot as plt
import mask_functions as msk

# Obtain the smaller contrast vial masks

directory = r'D:\OneDrive - University of Victoria\Research\LDA Data\21-03-31_CT_AuNPs_2'

folder = 'phantom_scan_3'
num_vials = 2
num_slices = 6
zs = [8, 14]
data = np.load(os.path.join(directory, folder, 'Norm CT', 'CT_norm.npy'))[2]

#
# # This is for the bottom of the vials (for AuCl3)
masks_control = np.zeros((num_vials, num_slices, 576, 576))
# # Order is Vial 1, then go through all slices, then go to vial 2
for i in range(num_vials):
    for j, z in enumerate(np.arange(zs[0], zs[1])):
        print(f'Vial {i} Slice ind {j} slice {z}')
        if z == 10:
            masks_control[i, j] = np.nan
        else:
            masks_control[i, j] = msk.square_ROI(data[z])


np.save(os.path.join(directory, folder, 'contrast_masks_Au_control.npy'), masks_control)

num_vials = 5
num_slices = 4
zs = [11, 15]

masks = np.zeros((num_vials, num_slices, 576, 576))
for i in range(num_vials):
    print('AUNPS NOW')
    for j, z in enumerate(np.arange(zs[0], zs[1])):
        print(f'Vial {i} Slice ind {j} slice {z}')
        if z == 10:
            masks[i, j] = np.nan
        else:
            masks[i, j] = msk.single_pixels_mask(data[z])
            masks[i, j] = msk.square_ROI(data[z])

# masks = np.load(os.path.join(directory, folder, 'contrast_masks_Au.npy'))
# masks[2, 1] = msk.square_ROI(data[12])
# masks[5, 0] = msk.square_ROI(data[11])
# # This is for same size ROIs
# masks = msk.phantom_ROIs(data[12], radius=10)

# np.save(os.path.join(directory, folder, 'contrast_masks_Au.npy'), masks)
