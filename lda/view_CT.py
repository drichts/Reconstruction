import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import mask_functions as msk

directory = r'D:\OneDrive - University of Victoria\Research\LDA Data'
folder5 = '21-04-14_CT_bin_width_10'
folder10 = '21-04-14_CT_bin_width_5'

sub1 = 'metal_phantom'
sub2 = 'water_phantom'
bin_num = 2
z = 12

data5_1 = np.load(os.path.join(directory, folder5, sub1, 'Norm CT', 'CT_norm.npy'))
# data10_1 = np.load(os.path.join(directory, folder10, sub1, 'Norm CT', 'CT_norm.npy'))[bin_num]

data5_2 = np.load(os.path.join(directory, folder5, sub2, 'Norm CT', 'CT_norm.npy'))[5]
# data10_2 = np.load(os.path.join(directory, folder10, sub2, 'Norm CT', 'CT_norm.npy'))[bin_num]


fig, ax = plt.subplots(1, 3, figsize=(12, 5))
ax[0].axis('off')
ax[1].axis('off')
ax[2].axis('off')

ax[0].imshow(data5_2[z, 138:440, 138:440][100:175, 25:95], cmap='gray', vmin=-400, vmax=400)
ax[0].set_title(f'Water only: 16-120 keV')

ax[1].imshow(data5_1[5, z, 138:440, 138:440][100:175, 25:95], cmap='gray', vmin=-400, vmax=400)
ax[1].set_title(f'Metal: 16-120 keV')

ax[2].imshow(data5_1[bin_num, z, 138:440, 138:440][100:175, 25:95], cmap='gray', vmin=-400, vmax=400)
ax[2].set_title(f'Metal: 90-100 keV')

fig.show()


