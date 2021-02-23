import os
from glob import glob
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import general_functions as gen
import mask_functions as msk
file = '/home/knoll/LDAData'

folder = r'/home/knoll/LDAData'
# data2 = np.load(os.path.join(folder, 'recon_CGLS_v2.npy'))
# # data2 = np.load(os.path.join(folder, 'recon_SIRT_v2.npy'))
#
# for i in range(len(data2)):
#     fig = plt.figure(figsize=(5, 5))
#     plt.imshow(data2[i, 11], cmap='gray', vmin=0.002, vmax=0.008)
#     plt.title(f'{(i+1)*10} iterations')
#     plt.show()
#     # plt.savefig(os.path.join(folder, f'SIRT_{(i+1)*10}.png'), dpi=fig.dpi)
#     plt.pause(1)
#     plt.close()

