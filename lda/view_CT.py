import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import mask_functions as msk

angles = 900

directory = r'D:\OneDrive - University of Victoria\Research\LDA Data'
# folder1 = f'CT_02-03-21-v2'
folder1 = r'21-02-24_CT_min_Au\phantom_scan'
# folder2 = 'CT_26-01-21_600\phantom_scan'
file = 'CT'
# data1 = np.load(os.path.join(directory, folder1, 'recon_SIRT.npy'))

data1 = np.load(os.path.join(directory, folder1, 'CT', file + '.npy'))
# data1 = loadmat(os.path.join(directory, folder1, 'CT', 'CT.mat'))['ct_img']
# np.save(os.path.join(directory, folder1, 'CT', 'CT.npy'), data1)

# data2 = np.load(os.path.join(directory, folder2, 'CT', 'CT.npy'))
# data2 = loadmat(os.path.join(directory, folder2, 'CT', 'CT.mat'))['ct_img']
# np.save(os.path.join(directory, folder2, 'CT', 'CT.npy'), data2)

fig = plt.figure(figsize=(7, 7))
#
# plt.imshow(data1[15, :, :], vmin=0.0, vmax=0.007, cmap='gray')
# plt.title('SIRT Recon')
# plt.show()

# data_sub = data1[1] - data1[0]
# plt.imshow(data_sub[110:470, 110:470, 15])
data = data1[3] - data1[2]

plt.imshow(data[110:470, 110:470, 12], vmin=-0.01, vmax=0.01, cmap='gray')
# plt.title('SIRT with ring correction')
plt.show()
# plt.savefig(os.path.join(directory, folder1, 'fig', file + '.png'), dpi=fig.dpi)
#
# fig = plt.figure(figsize=(8, 8))
# plt.imshow(data2[6, 125:450, 125:450, 15], cmap='gray')
# plt.title('600')
# plt.show()

