import os
import numpy as np

## This file is to check if we can even see anything in the alginate vials for the NPs
directory = r'D:\OneDrive - University of Victoria\Research\LDA Data\21-03-11_CT_AuNPs'
folders = ['phantom_scan_6', 'phantom_scan_7', 'phantom_scan_8', 'phantom_scan_9', 'phantom_scan_10', 'phantom_scan_11']

# Go through each folder and each slice within the folder to see if there is a significant difference in average
# vial value or std.
for folder in folders:
    print(folder)
    ct = np.load(os.path.join(directory, folder, 'Norm CT', 'CT_norm.npy'))
    k = np.load(os.path.join(directory, folder, 'Norm CT', 'K-edge_Au.npy'))
    masks = np.load(os.path.join(directory, folder, 'contrast_masks_Au.npy'))

    # Regular CT
    print('Regular CT')
    means = np.zeros((len(masks), 13))
    stds = np.zeros((len(masks), 13))
    for z in np.arange(6, 19):
        for i in range(len(masks)):
            means[i, z-6] = np.nanmean(masks[i] * ct[2, z])
            stds[i, z-6] = np.nanstd(masks[i] * ct[2, z])

    print(f'Std of the means {np.std(means, axis=0)}')
    print(f'Mean of the stds {np.mean(stds, axis=0)}')
    print()
    print()

