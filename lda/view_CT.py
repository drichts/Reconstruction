import os
import numpy as np
import matplotlib.pyplot as plt

directory = r'D:\OneDrive - University of Victoria\Research\LDA Data'
folder = '21-05-12_CT_metal'
# folder = '21-05-05_CT_metal_20keV_bins'
sub_folders = ['resolution']
phantom_masks = np.load(r'D:\OneDrive - University of Victoria\Research\LDA Data\21-05-05_CT_metal_artifact\2_metal_1and2\phantom_mask_mtf.npy')
masks = np.load(r'D:\OneDrive - University of Victoria\Research\LDA Data\21-05-05_CT_metal_artifact\2_metal_1and2\masks_mtf.npy')

# Parameters
bin_width = 1  # Multiples of 10
thresholds = [70, 80, 90, 100, 110, 120]
bin_num = 0  # 0, 1, 2, 3, 4, 5  (There is no pile-up bin)
z_slice = 13  # The slice number
low = -400  # Low window level
high = 200  # High window level

for sub_folder in sub_folders:
    data = np.load(os.path.join(directory, folder, sub_folder, 'Norm CT', 'CT_norm.npy'))


    for i in range(6):
        # Plot the desired data
        fig = plt.figure(figsize=(7, 7))
        plt.axis('off')
        # if i == 3:
        #     plt.imshow(data[i, 13, 138:440, 138:440], vmin=low, vmax=high, cmap='gray')
        # else:
        plt.imshow(data[i, z_slice, 138:440, 138:440], vmin=low, vmax=high, cmap='gray')
    # for i in range(6):
    #     plt.imshow(masks[i], alpha=0.8)


        if i < 5:
            title = f'{thresholds[i]}-{thresholds[i+bin_width]} keV'
        else:
            title = '16-120 keV'

        plt.title(title)
        plt.show()
        fig.savefig(os.path.join(directory, folder, sub_folder, title + '.png'), dpi=500)
        plt.close()
