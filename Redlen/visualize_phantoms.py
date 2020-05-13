import numpy as np
import matplotlib.pyplot as plt
import glob
import generateROImask as grm

directory = 'C:/Users/10376/Documents/Phantom Data/Uniformity/'

folders = ['m20358_q20_al_bluebelt_acryl_1w', 'm20358_q20_al_bluebelt_acryl_4w',
           'm20358_q20_al_bluebelt_fat_1w', 'm20358_q20_al_bluebelt_fat_4w',
           'm20358_q20_al_bluebelt_solidwater_1w', 'm20358_q20_al_bluebelt_solidwater_4w',
           'm20358_q20_al_polypropylene_1w', 'm20358_q20_al_polypropylene_4w']

air_folders = ['m20358_q20_al_air_1w', 'm20358_q20_al_air_4w']

#%% Plot the 5 bins plus the EC bin for any folder, correcting the raw data manually
folder = folders[7]
air_fold = air_folders[1]

data = np.load(directory + folder + '/Raw Data/Run010_a0.npy')
air = np.load(directory + air_fold + '/Raw Data/a0.npy')

fig, axes = plt.subplots(2, 3, figsize=(8, 6))

# Correct for air
data = np.sum(data, axis=2)
air = np.sum(air, axis=2)

data = -1*np.log(np.divide(data, air))
data = np.squeeze(data)

titles = ['20-30 keV', '30-50 keV', '50-70 keV', '70-90 keV', '90-120 keV', '20-120 keV']

for i, ax in enumerate(axes.flat):
    if i == 5:
        ax.imshow(data[12])
    else:
        ax.imshow(data[i+6])
    ax.set_title(titles[i])
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()

#%% Plot the 5 bins plus the EC bin for any folder, with auto corrected data
folder = folders[7]
titl = 'Polypropylene in acrylic 4w'
data = np.load(directory + folder + '/Corrected Data/Run008_a0.npy')

fig, axes = plt.subplots(2, 3, figsize=(8, 6))
ax1 = fig.add_subplot(111, frameon=False)
ax1.grid(False)
ax1.set_xticks([])
ax1.set_yticks([])

data = np.sum(data, axis=2)
data = np.squeeze(data)

titles = ['20-30 keV', '30-50 keV', '50-70 keV', '70-90 keV', '90-120 keV', '20-120 keV']

for i, ax in enumerate(axes.flat):
    mn, mx = np.min(data[6:11, :, 0:35]), np.max(data[6:11, :, 0:35])
    if i == 5:
        ax.imshow(data[12, :, 0:35], vmin=mn, vmax=mx)
    else:
        ax.imshow(data[i+6, :, 0:35], vmin=mn, vmax=mx)
    ax.set_title(titles[i])
    ax.set_xticks([])
    ax.set_yticks([])

ax1.set_title(titl, fontsize=15, pad=10)
plt.show()



#%% See A0 and A1
folder = folders[0]

data1 = np.load(directory + folder + '/Corrected Data/Run008_a0.npy')
data2 = np.load(directory + folder + '/Corrected Data/Run008_a1.npy')
data2 = np.flip(data2, axis=3)
data2 = np.flip(data2, axis=4)

data = np.concatenate((data1, data2), axis=4)
data = np.squeeze(np.sum(data, axis=2))
plt.imshow(data[12])
plt.show()

#%% Get masks for the folders

for idx, folder in enumerate(folders[0:8:2]):
    data = np.load(directory + folder + '/Corrected Data/Run010_a0.npy')
    data = np.sum(data, axis=2)
    data = np.squeeze(data)

    mask = grm.background_ROI(data[12])
    bg = grm.background_ROI(data[12])
    np.save(directory + folder + '/a0_Mask.npy', mask)
    np.save(directory + folder + '/a0_Background.npy', bg)

    np.save(directory + folders[idx*2+1] + '/a0_Mask.npy', mask)
    np.save(directory + folders[idx*2+1] + '/a0_Background.npy', bg)

#    print(folder)
#    print(folders[idx*2+1])
#    print()

#%% Test
#x = np.load(directory + folders[6] + '/Raw Data/Run001_a0.npy')
#z = np.load(directory + air_folders[0] + '/Raw Data/a0.npy')

#y = np.sum(x, axis=2)
#z = np.sum(z, axis=2)

#y = np.log(np.divide(z, y))
#x = np.sum(x, axis=2)
#for i in np.arange(2, 11):
#    file = directory + folders[6] + '/Raw Data/Run' + '{:03d}'.format(i) + '_a0.npy'
#    temp_x = np.load(file)
#    temp_x = np.sum(temp_x, axis=2)
#    x = np.add(x, temp_x)

#z = np.multiply(z, 10)

#x = np.log(np.divide(z, x))

#fig, ax = plt.subplots(1, 2)
#ax[0].imshow(x[0, 9])
#ax[1].imshow(y[0, 9])
#plt.show()

