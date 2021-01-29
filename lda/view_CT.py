import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import mask_functions as msk

directory = r'D:\OneDrive - University of Victoria\Research\LDA Data'
folder1 = 'CT_26-01-21_900\phantom_scan'
folder2 = 'CT_26-01-21_600\phantom_scan'

data1 = np.load(os.path.join(directory, folder1, 'CT', 'CT.npy'))
# data1 = loadmat(os.path.join(directory, folder1, 'CT', 'CT.mat'))['ct_img']
# np.save(os.path.join(directory, folder1, 'CT', 'CT.npy'), data1)

# data2 = np.load(os.path.join(directory, folder2, 'CT', 'CT.npy'))
# data2 = loadmat(os.path.join(directory, folder2, 'CT', 'CT.mat'))['ct_img']
# np.save(os.path.join(directory, folder2, 'CT', 'CT.npy'), data2)

fig = plt.figure(figsize=(8, 8))
plt.imshow(data1[6, 160:240, 240:330, 15], vmin=0, vmax=0.04, cmap='gray')
plt.title('900')
plt.show()
#
# fig = plt.figure(figsize=(8, 8))
# plt.imshow(data2[6, 125:450, 125:450, 15], cmap='gray')
# plt.title('600')
# plt.show()

