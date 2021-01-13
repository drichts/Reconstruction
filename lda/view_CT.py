import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

directory = r'D:\OneDrive - University of Victoria\Research\LDA Data'
folder = 'small_phant_120kVp_1mA_1mmAl'

# data = np.load(os.path.join(directory, folder, 'CT', 'CT.npy'))
data = loadmat(os.path.join(directory, folder, 'CT', 'CT.mat'))['ct_img']
np.save(os.path.join(directory, folder, 'CT', 'CT.npy'), data)

fig = plt.figure(figsize=(8, 8))
plt.imshow(data[6, :, :, 14])
plt.show()

# x = np.load(r'D:\OneDrive - University of Victoria\Research\LDA Data\1_7_2021\ct_4s_120kVp_3mA_1mmAl_x_ray_at_300\Data\proj_filt.npy')
# y = loadmat(r'D:\OneDrive - University of Victoria\Research\LDA Data\1_7_2021\ct_4s_120kVp_3mA_1mmAl_x_ray_at_300\Data\proj_filt.mat')['proj_filt']
# y = np.transpose(y, axes=(0, 3, 2, 1))
# print(np.array_equal(x, y, equal_nan=True))
# z = x-y
# print(np.nanmax(z))