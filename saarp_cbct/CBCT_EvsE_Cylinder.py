import numpy as np
import matplotlib.pyplot as plt

# Main file path
file_path = 'D:/Research/Python Data/CBCT/Au_6-5-19/'

vials = 6  # Number of ROIs

percentages = ['1_Percent/']#, '3_Percent/', '5_Percent/']

energies = ['40kVp', '80kVp']

counter = 1
for percent in percentages:

    filepath = file_path# + percent

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
    HU_40kVp_water = np.array([])
    HU_80kVp_water = np.array([])
    HU_40kVp_1 = np.array([])
    HU_80kVp_1 = np.array([])
    HU_40kVp_2 = np.array([])
    HU_80kVp_2 = np.array([])
    HU_40kVp_3 = np.array([])
    HU_80kVp_3 = np.array([])
    HU_40kVp_4 = np.array([])
    HU_80kVp_4 = np.array([])
    HU_40kVp_5 = np.array([])
    HU_80kVp_5 = np.array([])
    HU_40kVp_back = np.array([])
    HU_80kVp_back = np.array([])

    for i in np.arange(155, 180):

        path40 = filepath + energies[0] + '/volume0' + str(i) + '.npy'
        path80 = filepath + energies[1] + '/volume0' + str(i) + '.npy'

        temp40 = np.load(path40)
        temp80 = np.load(path80)

        # Go through each ROI
        # Find the mean HU in each ROI
        mean40 = np.nanmean(temp40*v40[0])
        mean80 = np.nanmean(temp80*v80[0])

        # Append the value
        HU_40kVp_water = np.append(HU_40kVp_water, mean40)
        HU_80kVp_water = np.append(HU_80kVp_water, mean80)

        mean40 = np.nanmean(temp40 * v40[1])
        mean80 = np.nanmean(temp80 * v80[1])
        HU_40kVp_1 = np.append(HU_40kVp_1, mean40)
        HU_80kVp_1 = np.append(HU_80kVp_1, mean80)

        mean40 = np.nanmean(temp40 * v40[2])
        mean80 = np.nanmean(temp80 * v80[2])
        HU_40kVp_2 = np.append(HU_40kVp_2, mean40)
        HU_80kVp_2 = np.append(HU_80kVp_2, mean80)

        mean40 = np.nanmean(temp40 * v40[3])
        mean80 = np.nanmean(temp80 * v80[3])
        HU_40kVp_3 = np.append(HU_40kVp_3, mean40)
        HU_80kVp_3 = np.append(HU_80kVp_3, mean80)

        mean40 = np.nanmean(temp40 * v40[4])
        mean80 = np.nanmean(temp80 * v80[4])
        HU_40kVp_4 = np.append(HU_40kVp_4, mean40)
        HU_80kVp_4 = np.append(HU_80kVp_4, mean80)

        mean40 = np.nanmean(temp40 * v40[5])
        mean80 = np.nanmean(temp80 * v80[5])
        HU_40kVp_5 = np.append(HU_40kVp_5, mean40)
        HU_80kVp_5 = np.append(HU_80kVp_5, mean80)

        # Same for the background
        mean40 = np.nanmean(temp40*back40)
        mean80 = np.nanmean(temp80*back80)

        HU_40kVp_back = np.append(HU_40kVp_back, mean40)
        HU_80kVp_back = np.append(HU_80kVp_back, mean80)

    plt.scatter(HU_40kVp_water, HU_80kVp_water, color='blue', s=100)
    plt.scatter(HU_40kVp_back, HU_80kVp_back, color='lightblue', s=100)
    plt.scatter(HU_40kVp_1, HU_80kVp_1, color='orange', s=100)
    plt.scatter(HU_40kVp_2, HU_80kVp_2, color='green', s=100)
    plt.scatter(HU_40kVp_3, HU_80kVp_3, color='purple', s=100)
    plt.scatter(HU_40kVp_4, HU_80kVp_4, color='red', s=100)
    plt.scatter(HU_40kVp_5, HU_80kVp_5, color='black', s=100)

    counter += 1

HU_40_all = np.concatenate((HU_40kVp_water, HU_40kVp_1))
HU_40_all = np.concatenate((HU_40_all, HU_40kVp_2))
HU_40_all = np.concatenate((HU_40_all, HU_40kVp_3))
HU_40_all = np.concatenate((HU_40_all, HU_40kVp_4))
HU_40_all = np.concatenate((HU_40_all, HU_40kVp_5))

HU_80_all = np.concatenate((HU_80kVp_water, HU_80kVp_1))
HU_80_all = np.concatenate((HU_80_all, HU_80kVp_2))
HU_80_all = np.concatenate((HU_80_all, HU_80kVp_3))
HU_80_all = np.concatenate((HU_80_all, HU_80kVp_4))
HU_80_all = np.concatenate((HU_80_all, HU_80kVp_5))

# Construct data for a linear fit of the data
coeffs = np.polyfit(HU_40_all, HU_80_all, 1)

# Calculate y points from the fit above
xpts = np.linspace(-1000, 1500, 2000)
y_all = coeffs[0] * xpts + coeffs[1]

# r-squared
p = np.poly1d(coeffs)
# fit values, and mean
yhat = p(HU_40_all)  #
ybar = np.sum(HU_80_all) / len(HU_80_all)  # average value of y
ssreg = np.sum((yhat - ybar) ** 2)  # or sum([ (yihat - ybar)**2 for yihat in yhat])
sstot = np.sum((HU_80_all - ybar) ** 2)  # or sum([ (yi - ybar)**2 for yi in y])
r_sq = ssreg / sstot
r_sq = '%.3f' % r_sq
r_squared = str(r_sq)

plt.title('Gold Phantom (ROI)', fontsize=50)
plt.ylabel('HU (80 kVp)', fontsize=40)
plt.xlabel('HU (40 kVp)', fontsize=40)
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
plt.xlim([-250, 1250])
plt.ylim([-250, 750])
plt.legend(['water', 'PLA', '0.5%', '1%', '2%', '3%', '4%', '5%'], fontsize=35, loc='lower right')
plt.plot(xpts, p(xpts), lw=2)
plt.annotate("$R^2$ = " + r_squared, xy=(0, 1), xycoords='axes fraction', xytext=(50, -50), textcoords='offset pixels',
             horizontalalignment='left', verticalalignment='top', fontsize=35)

plt.show()
