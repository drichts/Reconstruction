import numpy as np
import matplotlib.pyplot as plt

# Main file path
filepath = 'D:/Research/Python Data/CBCT/Mouse_6-4-19/'

vials = 4  # Number of ROIs

energies = ['40kVp', '80kVp']

# Load all of the ROIs
v40 = np.empty([vials, 120, 120])
for k in np.arange(vials):
    v40[k, :, :] = np.load(filepath + energies[0] + '/Vial' + str(k) + '_MaskMatrix.npy')
back40 = np.load(filepath + energies[0] + '/BackgroundMaskMatrix.npy')  # Background ROI matrix

v80 = np.empty([vials, 120, 120])
for k in np.arange(vials):
    v80[k, :, :] = np.load(filepath + energies[1] + '/Vial' + str(k) + '_MaskMatrix.npy')
back80 = np.load(filepath + energies[1] + '/BackgroundMaskMatrix.npy')  # Background ROI matrix

# Keep track of HU values of
HU_40kVp_water = []
HU_80kVp_water = []
HU_40kVp_Lu = []
HU_80kVp_Lu = []
HU_40kVp_Gd = []
HU_80kVp_Gd = []
HU_40kVp_Au = []
HU_80kVp_Au = []
HU_40kVp_back = []
HU_80kVp_back = []


for i in np.arange(153, 166):
    path40 = filepath + energies[0] + '/volume0' + str(i) + '.npy'
    path80 = filepath + energies[1] + '/volume0' + str(i) + '.npy'

    temp40 = np.load(path40)
    temp80 = np.load(path80)

    # Go through each ROI
    # Find the mean HU in each ROI
    mean40 = np.nanmean(temp40*v40[0])
    mean80 = np.nanmean(temp80*v80[0])

    # Append the value
    HU_40kVp_water.append(mean40)
    HU_80kVp_water.append(mean80)

    mean40 = np.nanmean(temp40 * v40[3])
    mean80 = np.nanmean(temp80 * v80[3])
    HU_40kVp_Lu.append(mean40)
    HU_80kVp_Lu.append(mean80)

    mean40 = np.nanmean(temp40 * v40[2])
    mean80 = np.nanmean(temp80 * v80[2])
    HU_40kVp_Gd.append(mean40)
    HU_80kVp_Gd.append(mean80)

    mean40 = np.nanmean(temp40 * v40[1])
    mean80 = np.nanmean(temp80 * v80[1])
    HU_40kVp_Au.append(mean40)
    HU_80kVp_Au.append(mean80)

    # Same for the background
    mean40 = np.nanmean(temp40*back40)
    mean80 = np.nanmean(temp80*back80)

    HU_40kVp_back.append(mean40)
    HU_80kVp_back.append(mean80)

plt.scatter(HU_40kVp_water, HU_80kVp_water, color='blue', s=100)
plt.scatter(HU_40kVp_back, HU_80kVp_back, color='lightblue', s=100)
plt.scatter(HU_40kVp_Lu, HU_80kVp_Lu, color='purple', s=100)
plt.scatter(HU_40kVp_Gd, HU_80kVp_Gd, color='green', s=100)
plt.scatter(HU_40kVp_Au, HU_80kVp_Au, color='gold', s=100)

plt.title('Mouse Phantom (ROI)', fontsize=50)
plt.ylabel('HU (80 kVp)', fontsize=40)
plt.xlabel('HU (40 kVp)', fontsize=40)
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
plt.xlim([-250, 700])
plt.ylim([-250, 450])
plt.legend(['water', 'PMMA', 'Lu', 'Gd', 'Au'], fontsize=35, loc='lower right')

plt.show()


