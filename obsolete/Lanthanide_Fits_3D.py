import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as m3d

filepath = 'D:/Research/Python Data/CBCT/Lan_7-15-19/'
vial = 'Vial5'

HU_40 = np.load(filepath+vial+'_40kVp_points.npy')  # x-points
HU_80 = np.load(filepath+vial+'_80kVp_points.npy')  # y-points
con = np.array([0, 1, 3, 5])  # z-points (concentrations)

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
linepts = vv[0] * np.mgrid[-1000:1000:2j][:, np.newaxis]

# shift by the mean to get the line in the right place
linepts += datamean

# Verify that everything looks right.

ax = m3d.Axes3D(plt.figure())
ax.scatter3D(*data.T)
ax.plot3D(*linepts.T)
ax.set_xlabel('HU 40kVp')
ax.set_ylabel('HU 80kVp')
ax.set_zlabel('Concentration (%)')
plt.show()

print(vv[0])
#np.save(filepath+vial+'3Dcoords.npy', vv[0])

print(data)
print(datamean)

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
print(b)
print((b-y1)/m * n + z1)


