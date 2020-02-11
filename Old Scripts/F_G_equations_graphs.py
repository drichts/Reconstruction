import numpy as np
import matplotlib.pyplot as plt
import os

# Folder where the attenuation data lives
folder = 'D:/Research/Attenuation Data/NPY Files/'

# Z and attenuation data
Z_points = []
G40_points = []
F40_points = []

G80_points = []
F80_points = []

# Go through the .npy files and get the correct info
dirs = os.listdir(folder)
dirs.sort()
for file in dirs:
    temp = np.load(folder+file)

    # Get the Z value from the file
    file = file.replace('Z', '')
    file = file.replace('.npy', '')
    Z = int(file)

    Z_points.append(Z)

    # Go through each row to get the 40 kVp and 80 kVp data
    for row in temp:

        if row[0] == 0.05:
            G40_points.append(row[1])
            F40_points.append(row[2])

        if row[0] == 0.1:
            G80_points.append(row[1])
            F80_points.append(row[2])


# Fit the data
G40_coeffs = np.polyfit(Z_points, G40_points, 3)

#F40_coeffs_before = np.polyfit(Z_points[0:18], F40_points[0:18], 4)
#F40_coeffs_after = np.polyfit(Z_points[18:], F40_points[18:], 2)
F40_coeffs = np.polyfit(Z_points, F40_points, 3)
G80_coeffs = np.polyfit(Z_points, G80_points, 3)
#F80_coeffs_before = np.polyfit(Z_points[0:31], F80_points[0:31], 4)
#F80_coeffs_after = np.polyfit(Z_points[31:], F80_points[31:], 2)
F80_coeffs = np.polyfit(Z_points, F80_points, 3)

# Calculate fit from the coefficients above
Zpts = np.arange(4, 30.25, 0.25)
G40_fit = np.polyval(G40_coeffs, Zpts)
F40_fit = np.polyval(F40_coeffs, Zpts)

#F40_fit_before = np.polyval(F40_coeffs_before, Zpts[0:113])
#F40_fit_after = np.polyval(F40_coeffs_after, Zpts[113:])
#F40_fit = np.append(F40_fit_before, F40_fit_after)

G80_fit = np.polyval(G80_coeffs, Zpts)
F80_fit = np.polyval(F80_coeffs, Zpts)

#F80_fit_before = np.polyval(F80_coeffs_before, Zpts[0:155])
#F80_fit_after = np.polyval(F80_coeffs_after, Zpts[155:])
#F80_fit = np.append(F80_fit_before, F80_fit_after)


fig, ax = plt.subplots(1, 2, figsize=(11, 11))
# add a big axes, hide frame
fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axes
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.grid(False)

ax[0].scatter(Z_points, G40_points, color='darkblue')
ax[0].scatter(Z_points, G80_points, color='red')
ax[0].plot(Zpts, G40_fit, color='lightblue')
ax[0].plot(Zpts, G80_fit, color='orange')
ax[0].legend(['50 kVp fit', '100 kVp fit', '50 kVp points', '100 kVp points'])
ax[0].set_xlabel('Z')
ax[0].set_ylabel('G')
ax[0].set_ylim([0.2, 0.6])
ax[0].set_xlim([5, 15])
ax[0].set_title('G vs. Z')

ax[1].scatter(Z_points, F40_points, color='darkblue')
ax[1].scatter(Z_points, F80_points, color='red')
ax[1].plot(Zpts, F40_fit, color='lightblue')
ax[1].plot(Zpts, F80_fit, color='orange')
ax[1].legend(['50 kVp fit', '100 kVp fit', '50 kVp points', '100 kVp points'])
ax[1].set_xlabel('Z')
ax[1].set_ylabel('F')
ax[1].set_ylim([0, 2.5E-5])
ax[1].set_xlim([5, 15])
ax[1].set_title('F vs. Z')

plt.show()


