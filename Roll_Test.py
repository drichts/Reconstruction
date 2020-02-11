import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate as rotate

folder = 'D:/Research/Python Data/CBCT/SARRP_7-25-19/RawMatrices/'  # Folder name

energies = ['40kVp', '80kVp']
angle = 1.5
index = np.round(np.abs(angle), decimals=0)
index = int(index)

# Graph both
both = False

im_40 = np.load(folder + energies[0] + '/volume0175.npy')
im_80 = np.load(folder + energies[1] + '/volume0175.npy')

#im_80 = np.roll(im_80, 1, axis=0)  # Y movement
#im_80 = np.roll(im_80, 1, axis=1)  # X movement
#im_80 = rotate(im_80, angle)
#im_80 = im_80[index:index+120, index:index+120]
print(np.shape(im_80))
print(np.shape(im_40))

if both:
    fig, ax = plt.subplots(1, 2, figsize=(11, 11))
    # add a big axes, hide frame
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axes
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)

    ax[0].imshow(im_40)

    ax[1].imshow(im_80)

    plt.show()

else:
    plt.imshow(im_80)
    plt.show()


