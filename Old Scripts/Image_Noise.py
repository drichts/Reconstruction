import numpy as np

# Main file path
file_path = 'D:/Research/Python Data/CBCT/Lan_7-15-19/'

vials = 6  # Number of ROIs

start, stop = 125, 150

percentages = ['1_Percent/', '3_Percent/', '5_Percent/']

energies = ['40kVp', '80kVp']

for percent in percentages:

    filepath = file_path #+ percent

    # Load all of the ROIs
    water40 = np.load(filepath + energies[0] + '/Vial0_MaskMatrix.npy')  # Water ROI matrix
    back40 = np.load(filepath + energies[0] + '/BackgroundMaskMatrix.npy')  # Background ROI matrix

    water80 = np.load(filepath + energies[1] + '/Vial0_MaskMatrix.npy')  # Water ROI matrix
    back80 = np.load(filepath + energies[1] + '/BackgroundMaskMatrix.npy')  # Background ROI


    WAT40 = np.array([])
    WAT80 = np.array([])

    phant_40 = np.array([])
    phant_80 = np.array([])

    for i in np.arange(start, stop):

        temp40 = np.load(filepath + energies[0] + '/volume0' + str(i) + '.npy')
        temp80 = np.load(filepath + energies[1] + '/volume0' + str(i) + '.npy')

        WAT40 = np.append(WAT40, np.nanstd(temp40*water40))
        WAT80 = np.append(WAT80, np.nanstd(temp80 * water80))

        phant_40 = np.append(phant_40, np.nanstd(temp40*back40))
        phant_80 = np.append(phant_80, np.nanstd(temp80*back80))

    print(percent, 'water 40 kVp', np.max(WAT40))
    print(percent, 'water 80 kVp', np.max(WAT80))

    print(percent, 'background 40 kVp', np.max(phant_40))
    print(percent, 'background 80 kVp', np.max(phant_80))
