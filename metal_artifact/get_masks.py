import os
import numpy as np
import mask_functions as msk


directory = r'D:\OneDrive - University of Victoria\Research\LDA Data'
folder = '21-05-12_CT_metal'

sub = 'metal_in'

data = np.load(os.path.join(directory, folder, sub, 'Norm CT', 'CT_norm.npy'))[5]

# phantom_mask = msk.phantom_ROIs(data[12], radius=3)
# np.save(os.path.join(directory, folder, sub, 'phantom_mask_mtf.npy'), phantom_mask)
#
# air_mask = msk.phantom_ROIs(data[12], radius=3)
# np.save(os.path.join(directory, folder, sub, 'air_mtf.npy'), air_mask)
#
# masks = np.zeros((6, 576, 576))
# for i in range(6):
#     masks[i] = msk.square_ROI(data[11])
#
# np.save(os.path.join(directory, folder, sub, 'masks_mtf.npy'), masks)

# ROI within the metal artifact
for i in np.arange(7):  # change for number of artifacts
    metal_mask = msk.phantom_ROIs(data[12], radius=3)
    # Sum all the individual vial masks together into one mask that grabs all ROIs
    metal_mask = np.nansum(metal_mask, axis=0)
    metal_mask[metal_mask == 0] = np.nan
    np.save(os.path.join(directory, folder, sub, f'artifact_mask_{i}.npy'), metal_mask)

# non_metal_mask = msk.phantom_ROIs(data[12], radius=7)
# # Sum all the individual vial masks together into one mask that grabs all ROIs
# non_metal_mask = np.nansum(non_metal_mask, axis=0)
# non_metal_mask[non_metal_mask == 0] = np.nan
# np.save(os.path.join(directory, folder, sub, 'no_artifact_mask.npy'), non_metal_mask)

# Contrast masks
# contrast_masks = msk.phantom_ROIs(data[16], radius=5)
# np.save(os.path.join(directory, folder, sub, 'contrast_masks.npy'), contrast_masks)
