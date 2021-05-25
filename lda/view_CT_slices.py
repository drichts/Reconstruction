import os
import numpy as np
import matplotlib.pyplot as plt

directory = r'D:\OneDrive - University of Victoria\Research\LDA Data'
folder = '21-05-12_CT_metal_no_ring_corr'
sub = 'metal_in'
bin_num = 1
slices = [9, 15]

data = np.load(os.path.join(directory, folder, sub, 'Norm CT', 'CT_norm.npy'))
# for z in np.arange(slices[0], slices[1]):
#     plt.imshow(data[bin_num, z, 138:440, 138:440], vmin=-400, vmax=300, cmap='gray')
#     plt.title(f'{z}')
#     plt.show()
#     plt.pause(2)
#     plt.close()

fig = plt.figure(figsize=(5, 5))
plt.imshow(data[bin_num, 14, 138:440, 138:440], vmin=-400, vmax=300, cmap='gray')
plt.savefig(os.path.join(directory, folder, sub, f'Bin{bin_num}.png'), dpi=500)



