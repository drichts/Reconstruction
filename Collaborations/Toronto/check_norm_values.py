import os
import numpy as np
import matplotlib.pyplot as plt

directory = r'D:\OneDrive - University of Victoria\Research\LDA Data\21-03-31_CT_AuNPs_2'

folder1 = 'phantom_scan_1_no_ring_corr'
folder2 = 'phantom_scan_2'
folder3 = 'phantom_scan_3'

for folder in [folder1]:

    # CT file
    file = os.path.join(directory, folder, 'CT', 'CT.npy')

    ct = np.load(file)[:, 12, :, :]
    k = ct[1] - ct[0]

    # Masks
    mask = np.load(os.path.join(directory, folder, 'contrast_masks_Au_control.npy'))[0, -2]
    water_mask = np.load(os.path.join(directory, folder, 'water_mask.npy'))
    air_mask = np.load(os.path.join(directory, folder, 'air_mask.npy'))

    water_vals = np.zeros(3)
    air_vals = np.zeros(3)
    conc_vals = np.array([np.nanmean(mask*k), np.nanmean(water_mask*k)])

    for i in range(3):
        water_vals[i] = np.nanmean(ct[i]*water_mask)
        air_vals[i] = np.nanmean(ct[i]*air_mask)

    np.save(os.path.join(directory, folder, 'water_vals.npy'), water_vals)
    np.save(os.path.join(directory, folder, 'air_vals.npy'), air_vals)
    np.save(os.path.join(directory, folder, 'conc_vals.npy'), conc_vals)

    print(conc_vals[0])
    print()
