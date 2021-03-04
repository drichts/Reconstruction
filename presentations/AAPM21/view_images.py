import os
import numpy as np
import matplotlib.pyplot as plt

# K-edge images

au = np.load(r'D:\OneDrive - University of Victoria\Research\LDA Data\AAPM21\Au\Two\K-edge.npy')
gd = np.load(r'D:\OneDrive - University of Victoria\Research\LDA Data\AAPM21\Gd\Two\K-edge.npy')

ct = np.load(r'D:\OneDrive - University of Victoria\Research\LDA Data\AAPM21\CT Resolution\Two\CT_norm.npy')

for i in range(8, 15):

    fig, ax = plt.subplots(1, 3, figsize=(12, 5))
    ax[0].imshow(au[12], vmin=0, vmax=19)
    ax[0].set_title('Au')
    ax[1].imshow(gd[12], vmin=0.5, vmax=50)
    ax[1].set_title('Gd')
    ax[2].imshow(ct[i], vmin=-800, vmax=100, cmap='gray')
    ax[2].set_title(f'{i}')
    plt.show()
    plt.pause(1)
    plt.close()