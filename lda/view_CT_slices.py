import os
import numpy as np
import matplotlib.pyplot as plt

directory = r'D:\OneDrive - University of Victoria\Research\LDA Data'
folder = '21-05-12_CT_metal'
sub = 'metal_in'
bin_num = 5
slices = [12, 19]

data = np.load(os.path.join(directory, folder, sub, 'Norm CT', 'CT_norm.npy'))
for z in np.arange(slices[0], slices[1]):
    plt.imshow(data[bin_num, z, 138:440, 138:440], vmin=-400, vmax=300, cmap='gray')
    plt.title(f'{z}')
    plt.show()
    plt.pause(10)
    plt.close()
