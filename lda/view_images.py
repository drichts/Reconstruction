import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import general_functions as gen
import mask_functions as msk

file = r'D:\OneDrive - University of Victoria\Research\LDA Data'

folder1 = 'CT_26-01-21_1800'
folder2 = '10-cm_25-01-21_2'

data1 = np.load(os.path.join(file, folder1, 'airscan_60s', 'Data', 'data_corr.npy'))
data2 = np.load(os.path.join(file, folder2, 'airscan_60s', 'Data', 'data_corr.npy'))

data = data2 - data1
data_other = data1 - data2

corr = np.log(data1) - np.log(data2)

plt.imshow(data2[:, :, 6])
# fig, ax = plt.subplots(5, 1, figsize=(12, 12))
# ax[0].imshow(data[4, :, :, 6], vmin=0, vmax=0.09)
# ax[1].imshow(data[40, :, :, 6], vmin=0, vmax=0.09)
# ax[2].imshow(data[80, :, :, 6], vmin=0, vmax=0.09)
# ax[3].imshow(data[120, :, :, 6], vmin=0, vmax=0.09)
# ax[4].imshow(data[160, :, :, 6], vmin=0, vmax=0.09)
# plt.show()
