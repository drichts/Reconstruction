import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import mask_functions as msk
import general_functions as gen


directory = r'D:\OneDrive - University of Victoria\Research\LDA Data'
folder1 = '21-03-03_CT_GdAu_CC'
folder2 = '21-03-03_CT_GdAu_SEC'

data1_ct = np.load(os.path.join(directory, folder1, 'phantom_scan', 'Norm CT', 'CT_norm.npy'))
data1_k = np.load(os.path.join(directory, folder1, 'phantom_scan', 'Norm CT', 'K-edge_2.npy'))

data2_ct = np.load(os.path.join(directory, folder2, 'phantom_scan', 'Norm CT', 'CT_norm.npy'))
data2_k = np.load(os.path.join(directory, folder2, 'phantom_scan', 'Norm CT', 'K-edge_2.npy'))

# Check for the right water slice
# for i in range(6, 17):
#     fig, ax = plt.subplots(1, 2, figsize=(9, 6))
#     ax[0].imshow(data1_ct[i], cmap='gray', vmin=-300, vmax=300)
#     ax[0].set_title(f'{i}')
#     ax[1].imshow(data1_k[i], vmin=0, vmax=19)
#     ax[1].set_title(f'{i}')
#     plt.show()
#     plt.pause(1)
#     plt.close()

temp_data1 = np.zeros((5, 576, 576))
temp_data2 = np.zeros((5, 576, 576))

for i, val in enumerate([11, 12, 13, 14]):
    temp_data1[i] = data1_ct[val]
    temp_data2[i] = data1_k[val]
data1_ct = temp_data1
data1_k = temp_data2
#
# for i in range(6, 17):
#     fig, ax = plt.subplots(1, 2, figsize=(9, 6))
#     ax[0].imshow(data2_ct[i], cmap='gray', vmin=-300, vmax=300)
#     ax[0].set_title(f'{i}')
#     ax[1].imshow(data2_k[i])
#     ax[1].set_title(f'{i}')
#     plt.show()
#     plt.pause(1)
#     plt.close()

temp_data1 = np.zeros((5, 576, 576))
temp_data2 = np.zeros((5, 576, 576))

for i, val in enumerate([11, 12, 13]):
    temp_data1[i] = data2_ct[val]
    temp_data2[i] = data2_k[val]
data2_ct = temp_data1
data2_k = temp_data2

water1 = np.load(os.path.join(directory, folder1, 'phantom_scan', 'water_mask.npy'))
back1 = np.load(os.path.join(directory, folder1, 'phantom_scan', 'back_mask.npy'))
contrast1 = np.load(os.path.join(directory, folder1, 'phantom_scan', 'contrast_masks_2.npy'))

water2 = np.load(os.path.join(directory, folder2, 'phantom_scan', 'water_mask.npy'))
back2 = np.load(os.path.join(directory, folder2, 'phantom_scan', 'back_mask.npy'))
contrast2 = np.load(os.path.join(directory, folder2, 'phantom_scan', 'contrast_masks_2.npy'))

# Data is of the form (axis 1) 0: K-edge CNR with water, 1: CT CNR with water, 2: K-edge CNR with phantom, 3: CT CNR with phantom
cnr1 = np.zeros((4, 4, 6))
cnr2 = np.zeros((3, 4, 6))

for i in range(4):
    for j in range(5):
        cnr1[i, 0, j], err = gen.cnr(data1_k[i], contrast1[j], water1)
        cnr1[i, 1, j], err = gen.cnr(data1_ct[i], contrast1[j], water1)

        cnr1[i, 2, j], err = gen.cnr(data1_k[i], contrast1[j], back1)
        cnr1[i, 3, j], err = gen.cnr(data1_ct[i], contrast1[j], back1)

    cnr1[i, 0, -1], err = gen.cnr(data1_k[i], water1, water1)
    cnr1[i, 1, -1], err = gen.cnr(data1_ct[i], water1, water1)

    cnr1[i, 2, -1], err = gen.cnr(data1_k[i], water1, back1)
    cnr1[i, 3, -1], err = gen.cnr(data1_ct[i], water1, back1)

for i in range(3):
    for j in range(5):
        cnr2[i, 0, j], err = gen.cnr(data2_k[i], contrast2[j], water2)
        cnr2[i, 1, j], err = gen.cnr(data2_ct[i], contrast2[j], water2)

        cnr2[i, 2, j], err = gen.cnr(data2_k[i], contrast2[j], back2)
        cnr2[i, 3, j], err = gen.cnr(data2_ct[i], contrast2[j], back2)

    cnr2[i, 2, -1], err = gen.cnr(data2_k[i], water2, water2)
    cnr2[i, 3, -1], err = gen.cnr(data2_ct[i], water2, water2)

    cnr2[i, 2, -1], err = gen.cnr(data2_k[i], water2, back2)
    cnr2[i, 3, -1], err = gen.cnr(data2_ct[i], water2, back2)

np.save(os.path.join(directory, folder1, 'phantom_scan', 'CNR_Au.npy'), cnr1)
np.save(os.path.join(directory, folder2, 'phantom_scan', 'CNR_Au.npy'), cnr2)
