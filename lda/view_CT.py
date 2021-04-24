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
bin_num = 3
z = 12

data5_1 = np.load(os.path.join(directory, folder5, sub1, 'Norm CT', 'CT_norm.npy'))[bin_num]
data10_1 = np.load(os.path.join(directory, folder10, sub1, 'Norm CT', 'CT_norm.npy'))[bin_num]

data5_2 = np.load(os.path.join(directory, folder5, sub2, 'Norm CT', 'CT_norm.npy'))[bin_num]
data10_2 = np.load(os.path.join(directory, folder10, sub2, 'Norm CT', 'CT_norm.npy'))[bin_num]


fig, ax = plt.subplots(2, 2, figsize=(10, 10))
ax[0, 0].axis('off')
ax[0, 1].axis('off')
ax[1, 0].axis('off')
ax[1, 1].axis('off')

ax[0, 0].imshow(data5_2[z, 138:440, 138:440], cmap='gray', vmin=-400, vmax=400)
ax[0, 0].set_title(f'Water only: Bin {bin_num}, width: 5 keV')

ax[0, 1].imshow(data10_2[z, 138:440, 138:440], cmap='gray', vmin=-400, vmax=400)
ax[0, 1].set_title(f'Water only: Bin {bin_num}, width: 10 keV')

ax[1, 0].imshow(data5_1[z, 138:440, 138:440], cmap='gray', vmin=-400, vmax=400)
ax[1, 0].set_title(f'With metal: Bin {bin_num}, width: 5 keV')

ax[1, 1].imshow(data10_1[z, 138:440, 138:440], cmap='gray', vmin=-400, vmax=400)
ax[1, 1].set_title(f'With metal: Bin {bin_num}, width: 10 keV')
fig.show()


