import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat
from datetime import datetime
import mask_functions as mf

directory = r'D:\OneDrive - University of Victoria\Research\LDA Data'

data_folder = 'airscan_120kVP_1mA_1mmAl_3x8coll_360s_6frames'

data = np.load(os.path.join(directory, data_folder, 'Data', 'data_corr.npy'))

# for i in np.arange(7):
#     fig = plt.figure(figsize=(10, 4))
#     plt.imshow(data[100, :, :, i])
#     plt.show()


data1 = data[2, :, :, 6]
data2 = data[3, :, :, 6]

corr = np.abs((np.log(data1) - np.log(data2)) * 100)

plt.imshow(corr, vmin=0, vmax=0.5)
plt.show()

# plt.imshow(corr, vmin=0, vmax=1)
# plt.show()

def compare_dark(dark1, dark2, name):
    fig = plt.figure(figsize=(10, 4))

    dt = datetime.now()
    filename = name + '_' + dt.strftime('%Y-%m-%d_%H-%M-%S') + '.png'

    max_value = np.nanmax(dark2)

    img = (dark1 - dark2) / max_value
    print(np.max(img))
    plt.imshow(img, vmin=-0.00000002, vmax=0.00000002)
    plt.title(name, fontsize=18)
    plt.colorbar(orientation='horizontal')
    plt.show()
    plt.savefig(os.path.join(directory, 'fig', 'dark', 'Subtract_DivideMax' + name + '.png'), dpi=fig.dpi)
    plt.close()


#for j in np.arange(1, 6):
# for i in np.arange(1, 6):
#     compare_dark(data[i, :, :, 6]/60, np.sum(data[1:, :, :, 6], axis=0)/300, f'{i} minus 300 s')
