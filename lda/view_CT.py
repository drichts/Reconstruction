import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import mask_functions as msk

angles = 900

directory = r'D:\OneDrive - University of Victoria\Research\LDA Data'
# folder1 = f'CT_02-03-21-v2'
folder1 = r'CT_02-03-21-v2\ring_artifact_corr'
# folder2 = 'CT_26-01-21_600\phantom_scan'

# data1 = np.load(os.path.join(directory, folder1, 'recon_SIRT.npy'))

data1 = np.load(os.path.join(directory, folder1, 'CT', 'CT.npy'))
# data1 = loadmat(os.path.join(directory, folder1, 'CT', 'CT.mat'))['ct_img']
# np.save(os.path.join(directory, folder1, 'CT', 'CT.npy'), data1)

# data2 = np.load(os.path.join(directory, folder2, 'CT', 'CT.npy'))
# data2 = loadmat(os.path.join(directory, folder2, 'CT', 'CT.mat'))['ct_img']
# np.save(os.path.join(directory, folder2, 'CT', 'CT.npy'), data2)

# fig = plt.figure(figsize=(7, 7))
#
# plt.imshow(data1[15, :, :], vmin=0.0, vmax=0.007, cmap='gray')
# plt.title('SIRT Recon')
# plt.show()

# data_sub = data1[1] - data1[0]
# plt.imshow(data_sub[110:470, 110:470, 15])
#
plt.imshow(data1[110:470, 110:470, 30], vmin=0.0, vmax=0.3, cmap='gray')
plt.title('Jerichos method')
plt.show()
# plt.savefig(os.path.join(directory, folder1, 'fig', 'python_fig_4-3.png'), dpi=fig.dpi)
#
# fig = plt.figure(figsize=(8, 8))
# plt.imshow(data2[6, 125:450, 125:450, 15], cmap='gray')
# plt.title('600')
# plt.show()

