import os
import numpy as np
import matplotlib.pyplot as plt
import general_functions as gen

folder = r'D:\OneDrive - University of Victoria\Research\LDA Data'
s1 = r'New airscans no filtration\airscan_60s_1'
s2 = 'small_phant_120kVp_1mA_1mmAl'

# f1 = os.path.join(folder, s1, 'Data/data.npy')
# f2 = os.path.join(folder, s2, 'Data/data.npy')
#
# air = np.load(f1)/12000
# data = np.load(f2)

# proj = np.log(air) - np.log(data)
proj = np.load(os.path.join(folder, s2, 'Data', 'data_corr.npy'))

# fig = plt.figure(figsize=(12, 5))
# plt.imshow(proj[143, :, :, 6])
# plt.show()

fig = plt.figure(figsize=(12, 8))
# ax[0].imshow(air[:, :, 6])
plt.imshow(proj[140, :, :, 6])
# ax[2].imshow(data[6, 140, :, :])
plt.show()
#
# fig1 = plt.figure(figsize=(12, 4))
# plt.imshow(data[6, :, 15, :])
# plt.show()

# for i in range(500):
#     fig = plt.figure(figsize=(12, 3))
#     plt.imshow(proj[i*10, :, :, 6])
#     plt.show()
#     plt.pause(0.2)
#     plt.close()