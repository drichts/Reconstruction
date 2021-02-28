import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import mask_functions as msk


directory = '/home/knoll/LDAData'
folder1 = '21-02-24_CT_min_Au_2'
folder2 = '21-02-24_CT_min_Au_2_FDK'

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

print(f'Mean water reg: {np.nanmean(data1_ct*water1)}, {np.nanstd(data1_ct*water1)}')
print(f'Mean water FDK: {np.nanmean(data2_ct*water2)}, {np.nanstd(data2_ct*water2)}')
print()
print(f'Mean air reg: {np.nanmean(data1_ct*air1)}, {np.nanstd(data1_ct*air1)}')
print(f'Mean air FDK: {np.nanmean(data2_ct*air2)}, {np.nanstd(data2_ct*air2)}')
print()

for i in range(6):
    print(f'Mean contrast {i} reg: {np.nanmean(data1_k * contrast1[i])}, {np.nanstd(data1_k * contrast1[i])}')
    print(f'Mean constrast {i} FDK: {np.nanmean(data2_k * contrast2[i])}, {np.nanstd(data2_k * contrast2[i])}')
    print()

print(f'Mean contrast {i} reg: {np.nanmean(data1_k * water1)}, {np.nanstd(data1_k * water1)}')
print(f'Mean constrast {i} FDK: {np.nanmean(data2_k * water2)}, {np.nanstd(data2_k * water2)}')

fig, ax = plt.subplots(2, 2, figsize=(10, 10))
ax[0, 0].imshow(data1_ct)
ax[0, 1].imshow(data2_ct)

ax[1, 0].imshow(data1_k)
ax[1, 1].imshow(data2_k)
plt.show(block=True)



