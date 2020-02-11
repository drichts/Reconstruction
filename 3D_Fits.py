import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as m3d

# Main file path
folder = 'D:/Research/Python Data/CBCT/Lan_7-15-19/'

# Number of ROIs
ROIs = 6

# Pick the vial you would like to analyze
vial = 1  # 1-5

percentages = ['1_Percent/', '3_Percent/', '5_Percent/']

energies = ['40kVp', '80kVp']

# Keep track of HU values of
HU_40kVp_water = []
HU_80kVp_water = []
HU_40kVp_1 = []
HU_80kVp_1 = []
HU_40kVp_2 = []
HU_80kVp_2 = []
HU_40kVp_3 = []
HU_80kVp_3 = []
HU_40kVp_4 = []
HU_80kVp_4 = []
HU_40kVp_5 = []
HU_80kVp_5 = []
HU_40kVp_back = []
HU_80kVp_back = []

for percent in percentages:

    filepath = folder + percent

    # Load all of the ROIs
    v40 = np.empty([ROIs, 120, 120])
    for k in np.arange(ROIs):
        v40[k, :, :] = np.load(filepath + '/Vial' + str(k) + '_MaskMatrix.npy')
    back40 = np.load(filepath + '/BackgroundMaskMatrix.npy')  # Background ROI matrix

    v80 = np.empty([ROIs, 120, 120])
    for k in np.arange(ROIs):
        v80[k, :, :] = np.load(filepath + '/Vial' + str(k) + '_MaskMatrix.npy')
    back80 = np.load(filepath + '/BackgroundMaskMatrix.npy')  # Background ROI matrix

    for i in np.arange(125, 151):
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

        mean40 = np.nanmean(temp40 * v40[1])
        mean80 = np.nanmean(temp80 * v80[1])
        HU_40kVp_1.append(mean40)
        HU_80kVp_1.append(mean80)

        mean40 = np.nanmean(temp40 * v40[2])
        mean80 = np.nanmean(temp80 * v80[2])
        HU_40kVp_2.append(mean40)
        HU_80kVp_2.append(mean80)

        mean40 = np.nanmean(temp40 * v40[3])
        mean80 = np.nanmean(temp80 * v80[3])
        HU_40kVp_3.append(mean40)
        HU_80kVp_3.append(mean80)

        mean40 = np.nanmean(temp40 * v40[4])
        mean80 = np.nanmean(temp80 * v80[4])
        HU_40kVp_4.append(mean40)
        HU_80kVp_4.append(mean80)

        mean40 = np.nanmean(temp40 * v40[5])
        mean80 = np.nanmean(temp80 * v80[5])
        HU_40kVp_5.append(mean40)
        HU_80kVp_5.append(mean80)

        # Same for the background
        mean40 = np.nanmean(temp40*back40)
        mean80 = np.nanmean(temp80*back80)

        HU_40kVp_back.append(mean40)
        HU_80kVp_back.append(mean80)

wat_len = len(HU_40kVp_water)
oth_len = int(len(HU_40kVp_1) / 3)

# Create the z-axis points (Concentrations)
con = np.empty(wat_len + 3 * oth_len)
con[0:wat_len] = 0
con[wat_len:wat_len + oth_len] = 1
con[wat_len+oth_len:wat_len+2*oth_len] = 3
con[wat_len+2*oth_len:wat_len+3*oth_len] = 5

HU_40 = np.array(HU_40kVp_water)
HU_80 = np.array(HU_80kVp_water)
# Check the vial and select the correct lists
if vial is 1:
    HU_40 = np.concatenate((HU_40, HU_40kVp_1))
    HU_80 = np.concatenate((HU_80, HU_80kVp_1))
elif vial is 2:
    HU_40 = np.concatenate((HU_40, HU_40kVp_2))
    HU_80 = np.concatenate((HU_80, HU_80kVp_2))
elif vial is 3:
    HU_40 = np.concatenate((HU_40, HU_40kVp_3))
    HU_80 = np.concatenate((HU_80, HU_80kVp_3))
elif vial is 4:
    HU_40 = np.concatenate((HU_40, HU_40kVp_4))
    HU_80 = np.concatenate((HU_80, HU_80kVp_4))
elif vial is 5:
    HU_40 = np.concatenate((HU_40, HU_40kVp_5))
    HU_80 = np.concatenate((HU_80, HU_80kVp_5))

# Get data in 1 matrix, each data set is a column
data = np.concatenate((HU_40[:, np.newaxis], HU_80[:, np.newaxis], con[:, np.newaxis]), axis=1)


# Calculate the mean of the points, i.e. the 'center' of the cloud
datamean = data.mean(axis=0)

# Do an SVD on the mean-centered data.
uu, dd, vv = np.linalg.svd(data - datamean)

# Now vv[0] contains the first principal component, i.e. the direction
# vector of the 'best fit' line in the least squares sense.

# Now generate some points along this best fit line, for plotting.

# I use -7, 7 since the spread of the data is roughly 14
# and we want it to have mean 0 (like the points we did
# the svd on). Also, it's a straight line, so we only need 2 points.
linepts = vv[0] * np.mgrid[-300:800:2j][:, np.newaxis]

# shift by the mean to get the line in the right place
linepts += datamean

# Verify that everything looks right.

ax = m3d.Axes3D(plt.figure())
ax.scatter3D(*data.T, color='black', s=40)
ax.plot3D(*linepts.T, lw=2)
ax.set_title('Iodine', fontsize=40)
ax.set_xlabel('HU 40kVp', fontsize=35, labelpad=40)
ax.set_ylabel('HU 80kVp', fontsize=35, labelpad=40)
ax.set_zlabel('Concentration (%)', fontsize=35, labelpad=15)
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(25)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(25)
for tick in ax.zaxis.get_major_ticks():
    tick.label.set_fontsize(25)
plt.show()

print(vv[0])
#np.save(filepath+vial+'3Dcoords.npy', vv[0])

z = 5
z1 = data[0][2]
y1 = data[0][1]
x1 = data[0][0]
l = vv[0][0]
m = vv[0][1]
n = vv[0][2]

x = (z-z1)/n * l + x1
y = (z-z1)/n * m + y1

print(x, y)
b = (40-x1)/l * m + y1
print('80kVp HU:', b)
print('Lowest concentration:', (b-y1)/m * n + z1)
