import numpy as np
import matplotlib.pyplot as plt

# Main file path
file_path = 'D:/Research/Python Data/CBCT/Mouse_6-4-19/'

# Flag for if we're operating with a cropped image (background is cropped)
crop = True

percentages = ['1_Percent/']#, '3_Percent/', '5_Percent/']

energies = ['40kVp', '80kVp']

im_40 = []
im_80 = []

for percent in percentages:

    filepath = file_path# + percent

    for i in np.arange(125, 151):
        path40 = filepath + energies[0] + '/volume0' + str(i) + '.npy'
        path80 = filepath + energies[1] + '/volume0' + str(i) + '.npy'

        temp40 = np.load(path40)
        temp80 = np.load(path80)

        # If you want to crop the background out, load the mask and multiply the
        if crop:
            crop_mask = np.load(filepath + '/Crop_Mask.npy')

            temp40 = np.multiply(temp40, crop_mask)
            temp80 = np.multiply(temp80, crop_mask)

        # Flatten to a single dimension array
        temp40 = temp40.flatten()
        temp80 = temp80.flatten()

        # Take out any nan values
        temp40 = temp40[~np.isnan(temp40)]
        temp80 = temp80[~np.isnan(temp80)]

        im_40 = np.append(im_40, temp40)
        im_80 = np.append(im_80, temp80)

print(len(im_40))
plt.scatter(im_40, im_80, s=1)
plt.title('Mouse Phantom; Pixel Count: ' + str(len(im_40)), fontsize=50)
plt.ylabel('HU (80 kVp)', fontsize=40)
plt.xlabel('HU (40 kVp)', fontsize=40)
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
plt.xlim([-250, 700])
plt.ylim([-250, 450])
plt.show()
