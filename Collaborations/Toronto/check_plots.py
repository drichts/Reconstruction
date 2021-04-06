import os
import numpy as np
import matplotlib.pyplot as plt

directory = r'D:\OneDrive - University of Victoria\Research\LDA Data\21-03-31_CT_AuNPs_2'

folder1 = 'phantom_scan_1'
folder2 = 'phantom_scan_2'
folder3 = 'phantom_scan_3'

fig, ax = plt.subplots(2, 3, figsize=(10, 5))

for i, folder in enumerate([folder1, folder2, folder3]):

    ct = np.load(os.path.join(directory, folder, 'Norm CT', 'CT_norm.npy'))[2]
    k = np.load(os.path.join(directory, folder, 'Norm CT', 'K-edge_Au.npy'))

    ax[0, i].imshow(ct[12, 138:440, 138:440], vmin=-400, vmax=400, cmap='gray')
    ax[1, i].imshow(k[12, 138:440, 138:440], vmin=0, vmax=50)

    masks = np.load(os.path.join(directory, folder, 'contrast_masks_Au_control.npy'))

    mean_50 = np.zeros((2, 5))
    mean_19 = np.zeros((2, 5))
    for idx, val in enumerate([8, 9, 10, 11, 12]):

        if val == 10:
            mean_50[0, idx] = np.nan
            mean_19[0, idx] = np.nan

            mean_50[1, idx] = np.nanmean(masks[0, idx-1] * k[val])
            mean_19[1, idx] = np.nanmean(masks[1, idx-1] * k[val])

        elif val == 8 or val == 11 or val == 13:
            mean_50[1, idx] = np.nan
            mean_19[1, idx] = np.nan

            mean_50[0, idx] = np.nanmean(masks[0, idx] * ct[val])
            mean_19[0, idx] = np.nanmean(masks[1, idx] * ct[val])

        else:
            mean_50[0, idx] = np.nanmean(masks[0, idx]*ct[val])
            mean_50[1, idx] = np.nanmean(masks[0, idx] * k[val])

            mean_19[0, idx] = np.nanmean(masks[1, idx] * ct[val])
            mean_19[1, idx] = np.nanmean(masks[1, idx] * k[val])

    print(f'CT 50 Mean: {np.nanmean(mean_50[0])}, std: {np.nanstd(mean_50[0])}')
    print(f'K-edge 50 Mean: {np.nanmean(mean_50[1])}, std: {np.nanstd(mean_50[1])}')
    print(f'CT 19 Mean: {np.nanmean(mean_19[0])}, std: {np.nanstd(mean_19[0])}')
    print(f'K-edge 19 Mean: {np.nanmean(mean_19[1])}, std: {np.nanstd(mean_19[1])}')
    print()

fig.show()