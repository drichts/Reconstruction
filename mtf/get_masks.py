import os
import numpy as np
import mask_functions as msk

directory = r'D:\OneDrive - University of Victoria\Research\LDA Data'
folder = '21-05-12_CT_metal'

sub = 'water_only'

data = np.load(os.path.join(directory, folder, sub, 'Norm CT', 'CT_norm.npy'))[5]

phantom_mask = msk.phantom_ROIs(data[12], radius=3)
np.save(os.path.join(directory, folder, sub, 'phantom_mask_mtf.npy'), phantom_mask)

air_mask = msk.phantom_ROIs(data[12], radius=3)
np.save(os.path.join(directory, folder, sub, 'air_mtf.npy'), air_mask)

masks = np.zeros((6, 576, 576))
for i in range(6):
    masks[i] = msk.square_ROI(data[11])

np.save(os.path.join(directory, folder, sub, 'masks_mtf.npy'), masks)


# Noise ROI
noise_mask = msk.square_ROI(data[12])
np.save(os.path.join(directory, folder, sub, 'noise_mtf.npy'), noise_mask)
np.save(os.path.join(directory, '21-05-05_CT_metal_20keV_bins', sub, 'noise_mtf.npy'), noise_mask)
np.save(os.path.join(directory, '21-05-05_CT_metal_30keV_bins', sub, 'noise_mtf.npy'), noise_mask)

# Water ROI close to metal artifacts
# water_mask = msk.phantom_ROIs(data[12], radius=7)
# np.save(os.path.join(directory, folder, sub, 'water_mtf.npy'), water_mask)
# np.save(os.path.join(directory, '21-05-05_CT_metal_20keV_bins', sub, 'water_mtf.npy'), water_mask)
# np.save(os.path.join(directory, '21-05-05_CT_metal_30keV_bins', sub, 'water_mtf.npy'), water_mask)

# x = np.load(os.path.join(directory, folder, sub, 'water_mtf.npy'))
# y = np.load(os.path.join(directory, folder, sub, 'noise_mtf.npy'))
#
# np.save(os.path.join(directory, '21-05-05_CT_metal_20keV_bins', sub, 'water_mtf.npy'), x)
# np.save(os.path.join(directory, '21-05-05_CT_metal_30keV_bins', sub, 'water_mtf.npy'), x)
#
# np.save(os.path.join(directory, '21-05-05_CT_metal_20keV_bins', sub, 'noise_mtf.npy'), y)
# np.save(os.path.join(directory, '21-05-05_CT_metal_30keV_bins', sub, 'noise_mtf.npy'), y)