import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import mask_functions as msk


directory = '/home/knoll/LDAData'
folder1 = '21-02-26_CT_min_Gd_3862_2mA'
folder2 = '21-02-26_CT_min_Gd_3862_2mA_SEC'

data1_ct = np.load(os.path.join(directory, folder1, 'phantom_scan', 'Norm CT', 'CT_norm.npy'))[12]
data1_k = np.load(os.path.join(directory, folder1, 'phantom_scan', 'Norm CT', 'K-edge.npy'))[12]

data2_ct = np.load(os.path.join(directory, folder2, 'phantom_scan', 'Norm CT', 'CT_norm.npy'))[12]
data2_k = np.load(os.path.join(directory, folder2, 'phantom_scan', 'Norm CT', 'K-edge.npy'))[12]

water1 = np.load(os.path.join(directory, folder1, 'phantom_scan', 'water_mask.npy'))
air1 = np.load(os.path.join(directory, folder1, 'phantom_scan', 'air_mask.npy'))
contrast1 = np.load(os.path.join(directory, folder1, 'phantom_scan', 'contrast_masks.npy'))

water2 = np.load(os.path.join(directory, folder2, 'phantom_scan', 'water_mask.npy'))
air2 = np.load(os.path.join(directory, folder2, 'phantom_scan', 'air_mask.npy'))
contrast2 = np.load(os.path.join(directory, folder2, 'phantom_scan', 'contrast_masks.npy'))

print(f'Mean water CC: {np.nanmean(data1_ct*water1)}, {np.nanstd(data1_ct*water1)}')
print(f'Mean water SEC: {np.nanmean(data2_ct*water2)}, {np.nanstd(data2_ct*water2)}')
print()
print(f'Mean air CC: {np.nanmean(data1_ct*air1)}, {np.nanstd(data1_ct*air1)}')
print(f'Mean air SEC: {np.nanmean(data2_ct*air2)}, {np.nanstd(data2_ct*air2)}')
print()

for i in range(8):
    print(f'Mean contrast {i} CC: {np.nanmean(data1_k * contrast1[i])}, CNR: {(np.nanmean(data1_k * contrast1[i]) - np.nanmean(data1_k*water1))/np.nanstd(data1_k* water1)}')
    print(f'Mean constrast {i} SEC: {np.nanmean(data2_k * contrast2[i])}, CNR: {(np.nanmean(data2_k * contrast2[i]) - np.nanmean(data2_k*water2))/np.nanstd(data2_k* water2)}')
    print()

print(f'Mean contrast water CC: {np.nanmean(data1_k*water1)}')
print(f'Mean constrast water SEC: {np.nanmean(data2_k*water2)}')

fig, ax = plt.subplots(2, 2, figsize=(10, 10))
ax[0, 0].imshow(data1_ct)
ax[0, 1].imshow(data2_ct)

ax[1, 0].imshow(data1_k)
ax[1, 1].imshow(data2_k)
plt.show(block=True)



