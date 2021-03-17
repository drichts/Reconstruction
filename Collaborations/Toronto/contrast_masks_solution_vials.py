import os
import numpy as np
import matplotlib.pyplot as plt
import mask_functions as msk

# Obtain the smaller contrast vial masks

directory = r'D:\OneDrive - University of Victoria\Research\LDA Data\21-03-11_CT_AuNPs'

folder = 'phantom_scan_5'
num_vials = 6
num_slices = 4
zs = [11, 15]
data = np.load(os.path.join(directory, folder, 'Norm CT', 'CT_norm.npy'))[2]


masks = np.zeros((num_vials, num_slices, 576, 576))
# Order is Vial 1, then go through all slices, then go to vial 2
for i in range(num_vials):
    for j, z in enumerate(np.arange(zs[0], zs[1])):
        masks[i, j] = msk.single_pixels_mask(data[z])

np.save(os.path.join(directory, folder, 'contrast_masks_Au.npy'), masks)
