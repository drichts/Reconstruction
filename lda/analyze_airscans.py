import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

directory = r'D:\OneDrive - University of Victoria\Research\LDA Data'
folder = 'darkscan_360s_6frames'
data = np.load(os.path.join(directory, folder, 'Data', 'data.npy'))


def compare_data(scan1, scan2, name, air_or_dark='dark', vmin=0, vmax=200):

    comp = np.divide(scan1, scan2)

    dt = datetime.now()
    filename = name + '_' + dt.strftime('%Y-%m-%d_%H-%M-%S') + '.png'
    savepath = os.path.join(directory, 'fig', air_or_dark)
    os.makedirs(savepath, exist_ok=True)
    savepath = os.path.join(savepath, filename)

    fig = plt.figure(figsize=(14, 3))
    plt.imshow(comp[:, :, 6]*100, vmin=vmin, vmax=vmax)
    plt.xlabel('Relative Response (%)', fontsize=15, labelpad=55)
    plt.colorbar(orientation='horizontal')
    plt.subplots_adjust(top=1, bottom=0.2)
    plt.title(name, fontsize=18)
    plt.show()
    #plt.savefig(savepath, dpi=fig.dpi)
    #plt.close()

compare_data(data[2], data[4], f'60s Airscan {2} vs. {4}')
# for j in np.arange(6):
#     for i in np.arange(6):
#         compare_data(data[j], data[i], f'60s Airscan {i} vs. {j}')
