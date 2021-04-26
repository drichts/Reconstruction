import os
import numpy as np
import mask_functions as msk

directory = r'D:\OneDrive - University of Victoria\Research\LDA Data'
folder = '21-04-14_CT_bin_width_10'

sub = 'metal_phantom'

data = np.load(os.path.join(directory, folder, sub, 'Norm CT', 'CT_norm.npy'))[5]

phantom_mask = msk.phantom_ROIs(data[12], radius=3)
np.save(os.path.join(directory, folder, 'phantom_mask_mtf_metal.npy'), phantom_mask)


masks = np.zeros((6, 576, 576))
for i in range(6):
    masks[i] = msk.square_ROI(data[12])

np.save(os.path.join(directory, folder, 'masks_mtf_metal.npy'), masks)
