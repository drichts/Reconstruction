# .npy to .mat for MATLAB material decomp
import numpy as np
import scipy.io as io
folder = 'D:/Research/Python Data/CBCT/SARRP_7-25-19/40kVp/Slices/'
savefolder = 'C:/Users/drich/Downloads/DECTsummary/'
start, stop = 180, 183
for i in np.arange(start, stop):

    x = np.load(folder+'volume0'+str(i)+'.npy')
    filename = '40kVp_volume0'+str(i)+'.mat'
    io.savemat(savefolder+filename, {'data': x})

#%% .mat to .npy from MATLAB material decomp
import numpy as np
import scipy.io as io

folder = 'C:/Users/drich/Downloads/DECTsummary/'

Z1 = io.loadmat(folder+'Z_values_180.mat')
Z2 = io.loadmat(folder+'Z_values_181.mat')
Z3 = io.loadmat(folder+'Z_values_182.mat')

p1 = io.loadmat(folder+'Densities_180.mat')
p2 = io.loadmat(folder+'Densities_181.mat')
p3 = io.loadmat(folder+'Densities_182.mat')

Z1 = Z1['Zc']
Z2 = Z2['Zc']
Z3 = Z3['Zc']

p1 = p1['rhoc']
p2 = p2['rhoc']
p3 = p3['rhoc']

np.save(folder + 'Z_values_180.npy', Z1)
np.save(folder + 'Z_values_181.npy', Z2)
np.save(folder + 'Z_values_182.npy', Z3)
np.save(folder + 'Densities_180.npy', p1)
np.save(folder + 'Densities_181.npy', p2)
np.save(folder + 'Densities_182.npy', p3)

#%% Plot MATLAB material decomp
import numpy as np
import matplotlib.pyplot as plt

folder = 'C:/Users/drich/Downloads/DECTsummary/'
p1 = np.load(folder + 'Densities_180.npy')
Z1 = np.load(folder + 'Z_values_180.npy')
p1 = p1.flatten()
Z1 = Z1.flatten()
print(len(Z1))
plt.scatter(Z1, p1, s=2)
plt.title('SARRP phantom (MATLAB Code)', fontsize=30)
plt.xlabel('Z', fontsize=20)
plt.ylabel(r'$\rho$', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlim([5, 20])
plt.ylim([0, 3])
plt.show()

#%% Plot images
import numpy as np
import matplotlib.pyplot as plt

folder = 'D:/Research/Python Data/CBCT/SARRP_7-25-19/'
img = np.load(folder + '40kVp/volume0180.npy')
water = np.load(folder + 'Vial0_MaskMatrix.npy')
one = np.load(folder + 'Vial1_MaskMatrix.npy')
two = np.load(folder + 'Vial2_MaskMatrix.npy')
three = np.load(folder + 'Vial3_MaskMatrix.npy')
four = np.load(folder + 'Vial4_MaskMatrix.npy')
print(np.nanmean(img*water))
print(np.nanmean(img*one))
print(np.nanmean(img*two))
print(np.nanmean(img*three))
print(np.nanmean(img*four))
plt.imshow(img)

#%% MAT Decomp SARRP Python
import numpy as np
import matplotlib.pyplot as plt

folder = 'D:/Research/Python Data/CBCT/SARRP_7-25-19/'
p1 = np.load(folder + 'dens-pixel.npy')
Z1 = np.load(folder + 'Z-pixel.npy')

plt.scatter(Z1, p1, s=2)
plt.title('SARRP phantom (Python Code)', fontsize=30)
plt.xlabel('Z', fontsize=20)
plt.ylabel(r'$\rho$', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlim([5, 20])
plt.ylim([0, 3])
plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt

directory = 'D:/Research/Python Data/Spectral CT/'
folder = 'Cu_0.5_8-14-19/K-Edge/'

x = np.load(directory+folder+'Bin4-3_Slice15.npy')
y = plt.imshow(x, cmap='gray', vmin=150)
plt.title('Au, k-edge: ~81keV', fontsize=35)
plt.axis('off')
plt.show()

#%% Load a specific ROI of the SARRP Phantom and output the HU for all slices and plot
import numpy as np
import matplotlib.pyplot as plt
import os

directory = 'D:/Research/Python Data/CBCT/SARRP_7-25-19/'
folder1 = '40kVp/Slices/'
folder2 = '80kVp/Slices/'
i = 2

mask = np.load(directory + 'Vial' + str(i) + '_MaskMatrix.npy')

xpts = np.arange(175, 251)
ypts1 = np.empty(len(xpts))
ypts2 = np.empty(len(xpts))

dirs = os.listdir(directory + folder1)
j = 0
for file in dirs:
    img40 = np.load(directory + folder1 + file)
    img80 = np.load(directory + folder2 + file)

    ypts1[j] = np.nanmean(mask * img40)
    ypts2[j] = np.nanmean(mask * img80)

    j += 1

mean40 = np.mean(ypts1[0:35])
mean80 = np.mean(ypts2[0:35])

line1 = mean40*np.ones(len(ypts1))
line2 = mean80*np.ones(len(ypts2))

fig, ax = plt.subplots(1, 2, figsize=(12, 12))

ax[0].plot(xpts, ypts1)
ax[0].plot(xpts, line1, color='red')
ax[0].set_xlabel('Slice #')
ax[0].set_ylabel('HU')
ax[0].set_title('40 kvP, Vial 2, Mean HU: ' + str(int(mean40)))

ax[1].plot(xpts, ypts2)
ax[1].plot(xpts, line2, color='red')
ax[1].set_xlabel('Slice #')
ax[1].set_ylabel('HU')
ax[1].set_title('80 kvP, Vial 2, Mean HU: ' + str(int(mean80)))

plt.show()

#%% Read txt files and extract the data
import numpy as np
import os

folder = 'D:/Research/Attenuation Data/SARRP Attenuation/'

dirs = os.listdir(folder)

for file in dirs:

    f = open(folder + file, 'rt')
    matrix = []
    for line in f:
        col = line.split()
        col = np.array(col)
        matrix.append(col)

    matrix = np.array(matrix, dtype='float')
    matrix = matrix[:, 1]
    file = file.replace('.txt', '.npy')
    np.save(folder + file, matrix)
    print(file)

#%%
import numpy as np
import matplotlib.pyplot as plt
import os

directory = 'D:/Research/Python Data/Spectral CT/'
folder = 'Al_2.0_9-12-19/'
back = np.load(directory+folder+'BackgroundMaskMatrix.npy')
vial = np.load(directory+folder+'Vial5_MaskMatrix.npy')
image = np.load(directory+folder+'Slices/Bin6_Slice16.npy')

std_bg = np.nanstd(back*image)
bg = np.nanmean(back*image)
gold = np.nanmean(vial*image)

CNR = (gold-bg)/std_bg
print(CNR)


plt.imshow(image, cmap='gray', vmin=-800, vmax=1500)
plt.show()
plt.pause(2)
plt.close()

plt.imshow(back*image)
plt.show()

#%% Load a specific ROI and output the HU for all slices and plot
import numpy as np
import matplotlib.pyplot as plt
import os

directory = 'D:/Research/Python Data/Spectral CT/Au_9-17-19/'
folder = 'Slices/'
bin = 6
xpts = np.arange(24)
for vial in np.arange(6):

    mask = np.load(directory + 'Vial' + str(vial) + '_MaskMatrix.npy')

    ypts = np.empty(24)

    for i, x in enumerate(xpts):
        file = 'Bin' + str(bin) + '_Slice' + str(x) + '.npy'
        img = np.load(directory+folder+file)
        ypts[i] = np.nanmean(mask*img)


    plt.plot(xpts, ypts)

plt.show()

#%% Plot both energies for DECT images
import numpy as np
import matplotlib.pyplot as plt

folder = 'D:/Research/Python Data/CBCT/Lan_7-18-19/5_Percent/'
#folder = 'D:/Research/Python Data/CBCT/Mouse_6-4-19/'
slice = 135
slice = str(slice)

img4 = np.load(folder + '40kVp/volume0' + slice + '.npy')
img8 = np.load(folder + '80kVp/volume0' + slice + '.npy')

fig, ax = plt.subplots(1, 2, figsize=(12, 12))
plt.setp(ax, xticks=[4, 60, 116], xticklabels=['-1.5', '0.0', '1.5'],
         yticks=[4, 60, 116], yticklabels=['-1.5', '0.0', '1.5'])

ax[0].imshow(img4, cmap='gray', vmin=-500, vmax=1000)
ax[0].set_title('40 kVp', fontsize=40)
ax[0].tick_params(labelsize=30)
ax[0].set_xlabel('x (cm)', fontsize=30)
ax[0].set_ylabel('y (cm)', fontsize=30)

ax[1].imshow(img8, cmap='gray', vmin=-500, vmax=1000)
ax[1].set_title('80 kVp', fontsize=40)
ax[1].tick_params(labelsize=30)
ax[1].set_xlabel('x (cm)', fontsize=30)
ax[1].set_ylabel('y (cm)', fontsize=30)
plt.show()

#%% Take ROIs from DECT and plot the HU vs HU with errorbars
import numpy as np
import matplotlib.pyplot as plt

folder = 'D:/Research/Python Data/CBCT/Mouse_6-4-19/'
start, stop = 130, 144
val4_1 = np.empty(stop-start)
std4_1 = np.empty(stop-start)
val8_1 = np.empty(stop-start)
std8_1 = np.empty(stop-start)

val4_2 = np.empty(stop-start)
std4_2 = np.empty(stop-start)
val8_2 = np.empty(stop-start)
std8_2 = np.empty(stop-start)

val4_3 = np.empty(stop-start)
std4_3 = np.empty(stop-start)
val8_3 = np.empty(stop-start)
std8_3 = np.empty(stop-start)

i = 0

vial41 = np.load(folder + '40kVp/Vial1_MaskMatrix.npy')
vial42 = np.load(folder + '40kVp/Vial2_MaskMatrix.npy')
vial43 = np.load(folder + '40kVp/Vial3_MaskMatrix.npy')
vial81 = np.load(folder + '80kVp/Vial1_MaskMatrix.npy')
vial82 = np.load(folder + '80kVp/Vial2_MaskMatrix.npy')
vial83 = np.load(folder + '80kVp/Vial3_MaskMatrix.npy')

for slice in np.arange(start, stop):

    img4 = np.load(folder + '40kVp/volume0' + str(slice) + '.npy')

    img8 = np.load(folder + '80kVp/volume0' + str(slice) + '.npy')

    temp4 = np.multiply(img4, vial41)
    temp8 = np.multiply(img8, vial81)
    val4_1[i] = np.nanmean(temp4)

    std4_1[i] = np.nanstd(temp4)
    val8_1[i] = np.nanmean(temp8)
    std8_1[i] = np.nanstd(temp8)

    temp4 = np.multiply(img4, vial42)
    temp8 = np.multiply(img8, vial82)
    val4_2[i] = np.nanmean(temp4)
    std4_2[i] = np.nanstd(temp4)
    val8_2[i] = np.nanmean(temp8)
    std8_2[i] = np.nanstd(temp8)

    temp4 = np.multiply(img4, vial43)
    temp8 = np.multiply(img8, vial83)
    val4_3[i] = np.nanmean(temp4)
    std4_3[i] = np.nanstd(temp4)
    val8_3[i] = np.nanmean(temp8)
    std8_3[i] = np.nanstd(temp8)

    i += 1

plt.scatter(val4_1, val8_1, color='orange', s=100)
plt.scatter(val4_3, val8_3, color='green', s=100)
plt.scatter(val4_2, val8_2, color='purple', s=100)
plt.errorbar(val4_1, val8_1, xerr=std4_1, yerr=std8_1, fmt='none', color='orange')

plt.errorbar(val4_3, val8_3, xerr=std4_3, yerr=std8_3, fmt='none', color='green')
plt.errorbar(val4_2, val8_2, xerr=std4_2, yerr=std8_2, fmt='none', color='purple')
plt.legend(['Au', 'Lu', 'Gd'], fontsize=45, loc='lower right', fancybox=True, shadow=True)
plt.xlabel('40 kVp (HU)', fontsize=45)
plt.ylabel('80 kVp (HU)', fontsize=45)
plt.xticks(fontsize=40)
plt.yticks(fontsize=40)

plt.show()

#%% Same thing as above but for the Lan phantom
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

folder = 'D:/Research/Python Data/CBCT/Lan_7-18-19/'
start, stop = 120, 135

val4 = np.empty([5, 3*(stop-start)])
std4 = np.empty([5, 3*(stop-start)])
val8 = np.empty([5, 3*(stop-start)])
std8 = np.empty([5, 3*(stop-start)])

percentages = ['1_Percent/', '3_Percent/', '5_Percent/']

i = 0

for percent in percentages:

    path = folder + percent

    vial1 = np.load(path + 'Vial1_MaskMatrix.npy')
    vial2 = np.load(path + 'Vial2_MaskMatrix.npy')
    vial3 = np.load(path + 'Vial3_MaskMatrix.npy')
    vial4 = np.load(path + 'Vial4_MaskMatrix.npy')
    vial5 = np.load(path + 'Vial5_MaskMatrix.npy')


    for slice in np.arange(start, stop):

        img4 = np.load(path + '40kVp/volume0' + str(slice) + '.npy')
        img8 = np.load(path + '80kVp/volume0' + str(slice) + '.npy')

        temp4 = np.multiply(img4, vial1)
        temp8 = np.multiply(img8, vial1)
        val4[0, i] = np.nanmean(temp4)
        std4[0, i] = np.nanstd(temp4)
        val8[0, i] = np.nanmean(temp8)
        std8[0, i] = np.nanstd(temp8)

        temp4 = np.multiply(img4, vial2)
        temp8 = np.multiply(img8, vial2)
        val4[1, i] = np.nanmean(temp4)
        std4[1, i] = np.nanstd(temp4)
        val8[1, i] = np.nanmean(temp8)
        std8[1, i] = np.nanstd(temp8)

        temp4 = np.multiply(img4, vial3)
        temp8 = np.multiply(img8, vial3)
        val4[2, i] = np.nanmean(temp4)
        std4[2, i] = np.nanstd(temp4)
        val8[2, i] = np.nanmean(temp8)
        std8[2, i] = np.nanstd(temp8)

        temp4 = np.multiply(img4, vial4)
        temp8 = np.multiply(img8, vial4)
        val4[3, i] = np.nanmean(temp4)
        std4[3, i] = np.nanstd(temp4)
        val8[3, i] = np.nanmean(temp8)
        std8[3, i] = np.nanstd(temp8)

        temp4 = np.multiply(img4, vial5)
        temp8 = np.multiply(img8, vial5)
        val4[4, i] = np.nanmean(temp4)
        std4[4, i] = np.nanstd(temp4)
        val8[4, i] = np.nanmean(temp8)
        std8[4, i] = np.nanstd(temp8)

        i += 1

zeros = np.zeros([5, stop-start])
val4 = np.append(val4, zeros, axis=1)
std4 = np.append(std4, zeros, axis=1)
std8 = np.append(std8, zeros, axis=1)
val8 = np.append(val8, zeros, axis=1)


# Construct data for a linear fit of the data
coeffs1 = np.polyfit(val4[0], val8[0], 1)
coeffs2 = np.polyfit(val4[1], val8[1], 1)
coeffs3 = np.polyfit(val4[2], val8[2], 1)
coeffs4 = np.polyfit(val4[3], val8[3], 1)
coeffs5 = np.polyfit(val4[4], val8[4], 1)

# Calculate y points from the fit above
xpts = np.linspace(0, 3300, 1000)
ypts1 = coeffs1[0] * xpts + coeffs1[1]
ypts2 = coeffs2[0] * xpts + coeffs2[1]
ypts3 = coeffs3[0] * xpts + coeffs3[1]
ypts4 = coeffs4[0] * xpts + coeffs4[1]
ypts5 = coeffs5[0] * xpts + coeffs5[1]

#p = np.poly1d(coeffs)
# fit values, and mean
#yhat = p(conc)  #
#ybar = np.sum(CNR) / len(CNR)  # average value of y
#ssreg = np.sum((yhat - ybar) ** 2)  # or sum([ (yihat - ybar)**2 for yihat in yhat])
#sstot = np.sum((CNR - ybar) ** 2)  # or sum([ (yi - ybar)**2 for yi in y])
#r_sq = ssreg / sstot
#r_sq = '%.3f' % r_sq
#r_squared = str(r_sq)

sns.set()
fig = plt.figure(figsize=(14, 7))
plt.plot(xpts, ypts1, color='dodgerblue')
plt.plot(xpts, ypts2, color='darkorchid')
plt.plot(xpts, ypts4, color='orangered')
plt.plot(xpts, ypts3, color='mediumseagreen')
plt.plot(xpts, ypts5, color='orange')

plt.scatter(val4[0], val8[0], color='dodgerblue')
plt.scatter(val4[1], val8[1], color='darkorchid')
plt.scatter(val4[3], val8[3], color='orangered')
plt.scatter(val4[2], val8[2], color='mediumseagreen')
plt.scatter(val4[4], val8[4], color='orange')

plt.errorbar(val4[0], val8[0], xerr=std4[0], yerr=std8[0], fmt='none', color='dodgerblue', capsize=3)
plt.errorbar(val4[1], val8[1], xerr=std4[1], yerr=std8[1], fmt='none', color='darkorchid', capsize=3)
plt.errorbar(val4[3], val8[3], xerr=std4[3], yerr=std8[3], fmt='none', color='orangered', capsize=3)
plt.errorbar(val4[2], val8[2], xerr=std4[2], yerr=std8[2], fmt='none', color='mediumseagreen', capsize=3)
plt.errorbar(val4[4], val8[4], xerr=std4[4], yerr=std8[4], fmt='none', color='orange', capsize=3)

bluepatch_R = mpatches.Patch(color='dodgerblue', label="$R^2$ = 1.000")
purplepatch_R = mpatches.Patch(color='darkorchid', label="$R^2$ = 0.999")
redpatch_R = mpatches.Patch(color='orangered', label="$R^2$ = 0.999")
greenpatch_R = mpatches.Patch(color='mediumseagreen', label="$R^2$ = 1.000")
orangepatch_R = mpatches.Patch(color='orange', label="$R^2$ = 1.000")

bluepatch = mpatches.Patch(color='dodgerblue', label='I (Z=53)')
purplepatch = mpatches.Patch(color='darkorchid', label='Gd (Z=64)')
redpatch = mpatches.Patch(color='orangered', label='Dy (Z=66)')
greenpatch = mpatches.Patch(color='mediumseagreen', label='Lu (Z=71)')
orangepatch = mpatches.Patch(color='orange', label='Au (Z=79)')

legend1 = plt.legend(handles=[bluepatch, purplepatch, redpatch, greenpatch, orangepatch], fontsize=18, loc=0,
           fancybox=True, shadow=True)
plt.legend(handles=[bluepatch_R, purplepatch_R, redpatch_R, greenpatch_R, orangepatch_R], fontsize=18, loc=4,
           fancybox=True, shadow=True)
plt.gca().add_artist(legend1)
plt.xlabel('40 kVp (HU)', fontsize=20)
plt.ylabel('80 kVp (HU)', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

plt.ylim([0, 2200])
plt.xlim([0, 3100])

plt.show()
#%% R squared values from above
p = np.poly1d(coeffs5)
# fit values, and mean
x_points = val4[4]
y_points = val8[4]
yhat = p(x_points)  #
ybar = np.sum(y_points) / len(y_points)  # average value of y
ssreg = np.sum((yhat - ybar) ** 2)  # or sum([ (yihat - ybar)**2 for yihat in yhat])
sstot = np.sum((y_points - ybar) ** 2)  # or sum([ (yi - ybar)**2 for yi in y])
r_sq = ssreg / sstot
r_sq = '%.3f' % r_sq
r_squared = str(r_sq)
print(r_squared)

#%%

new4 = val4[3][30:]
new8 = val8[3][30:]
xpts = np.linspace(0, 2000, 2000)
ypts = np.multiply(xpts, 1729/1888)
plt.plot(xpts, ypts, color='blue')
plt.errorbar(val4[3], val8[3], xerr=std4[3], yerr=std8[3], fmt='none', color='red')
plt.legend(['Predicted', 'Experimental'], fontsize=16)
plt.title('CBCT Dy Linearity', fontsize=25)
plt.xlabel('HU (40 kVp)', fontsize=20)
plt.ylabel('HU (80 kVp)', fontsize=20)
plt.tick_params(labelsize=16)
plt.show()


#%% Normalization for older files
import numpy as np
import os
directory = 'D:/Research/Python Data/CBCT/'
folder = 'Mouse_6-4-19/'
load_path = directory + folder + 'RawMatrices/80kVp/'
save_path = directory + folder + '80kVp/'
files = os.listdir(load_path)

water_ROI = np.load(save_path + 'Vial0_MaskMatrix.npy')
air_ROI = np.load(save_path + 'BackgroundMaskMatrix.npy')


for file in files:
    if 'volume' not in file:
        continue
    temp = np.load(load_path + file)
    water = np.nanmean(temp * water_ROI)
    air = np.nanmean(temp * air_ROI)
    water_air = water - air
    temp = np.subtract(temp, water)
    temp = np.divide(temp, water_air)
    temp = np.multiply(temp, 1000)
    np.save(save_path + file, temp)

#%% Plot CNR for sCT
import numpy as np
import matplotlib.pyplot as plt

directory = 'D:/Research/Python Data/Spectral CT/'
folder1 = 'Au_9-17-19/'
folder2 = 'I_9-17-19/'
path1 = directory + folder1
path2 = directory + folder2

img1 = np.load(path1 + 'K-Edge/Bin4-3_Slice14.npy')
water_ROI1 = np.load(path1 + 'Vial0_MaskMatrix.npy')
img2 = np.load(path2 + 'K-Edge/Bin1-0_Slice14.npy')
water_ROI2 = np.load(path2 + 'Vial0_MaskMatrix.npy')

water_img1 = np.multiply(img1, water_ROI1)
std_water1 = np.nanstd(water_img1)
mean_water1 = np.nanmean(water_img1)

water_img2 = np.multiply(img2, water_ROI2)
std_water2 = np.nanstd(water_img2)
mean_water2 = np.nanmean(water_img2)

CNR1 = np.empty(5)
CNR2 = np.empty(5)

for i in np.arange(5):
    vial1 = np.load(path1 + 'Vial' + str(i+1) + '_MaskMatrix.npy')
    temp1 = np.multiply(img1, vial1)

    vial2 = np.load(path2 + 'Vial' + str(i+1) + '_MaskMatrix.npy')
    temp2 = np.multiply(img2, vial2)

    CNR1[i] = np.nanmean(temp1)
    CNR2[i] = np.nanmean(temp2)

CNR1 = np.subtract(CNR1, mean_water1)
CNR2 = np.subtract(CNR2, mean_water2)

CNR1 = np.divide(CNR1, std_water1)
CNR2 = np.divide(CNR2, std_water2)

conc = np.array([0.5, 1, 2, 3, 4])

# Construct data for a linear fit of the data
coeffs1 = np.polyfit(conc, CNR1, 1)
coeffs2 = np.polyfit(conc, CNR2, 1)

# Calculate y points from the fit above
xpts = np.linspace(-1, 5, 100)
ypts1 = coeffs1[0] * xpts + coeffs1[1]
ypts2 = coeffs2[0] * xpts + coeffs2[1]

plt.scatter(conc, CNR2, color='blue', s=50)
plt.scatter(conc, CNR1, color='orange', s=50)

plt.plot(xpts, ypts2, color='blue', lw=2)
plt.plot(xpts, ypts1, color='orange', lw=2)
plt.plot(xpts, 4*np.ones(len(xpts)), color='red', linestyle='--', lw=2)
plt.legend(['I', 'Au', 'Rose Criterion'], fontsize=25)

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Concentration (%)', fontsize=25)
plt.ylabel('CNR', fontsize=25)
plt.xlim([0, 5])
plt.ylim([0, 50])

plt.show()

#%% Draw circles for ROIs
import numpy as np
import matplotlib.pyplot as plt

folder1 = 'Cu_1.0_9-13-19/'
folder2 = 'Cu_0.5_8-14-19/'
filename = 'Slices/Bin6_Slice14.npy'
directory = 'D:/Research/Python Data/Spectral CT/'

cents = np.load(directory + folder1 + 'center-coords.npy')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(np.load(directory+folder2+filename))

for center in cents:
    circ = plt.Circle(center, radius=4, fill=False, edgecolor='red')
    ax.add_artist(circ)

plt.show()

#%% Plot K-Edge Image
import matplotlib.colors as colors
directory = 'D:/Research/Python Data/Spectral CT/'
folder = 'AuGd_width_14_12-2-19/'
x = np.load(directory + folder + 'K-Edge/Bin1-0_Slice15.npy')

# Create the colormaps
nbins = 100
c1 = (1, 0, 1)
c2 = (0, 1, 0)
c3 = (1, 0.843, 0)
c4 = (0, 0, 1)

gray_val = 0
gray_list = (gray_val, gray_val, gray_val)

c1_rng = [gray_list, c1]
cmap1 = colors.LinearSegmentedColormap.from_list('Purp', c1_rng, N=nbins)
c2_rng = [gray_list, c2]
cmap2 = colors.LinearSegmentedColormap.from_list('Gree', c2_rng, N=nbins)
c3_rng = [gray_list, c3]
cmap3 = colors.LinearSegmentedColormap.from_list('G78', c3_rng, N=nbins)
c4_rng = [gray_list, c4]
cmap4 = colors.LinearSegmentedColormap.from_list('Blu8', c4_rng, N=nbins)

fig = plt.figure(figsize=(8, 8))
plt.imshow(x, cmap=cmap2, vmin=-0.01, vmax=0.01)
plt.grid(False)
plt.xticks([4, 60, 116], labels=[-1.5, 0, 1.5], fontsize=16)
plt.yticks([4, 60, 116], labels=[-1.5, 0, 1.5], fontsize=16)
plt.title('K-Edge: 50keV with 14 keV bin width', fontsize=20)
plt.xlabel('x (cm)', fontsize=18)
plt.ylabel('y (cm)', fontsize=18)
plt.show()

