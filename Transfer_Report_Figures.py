# Transfer Report Figures

#%% Figure 2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import generateROImask as grm

directory = 'D:/Research/Python Data/Spectral CT/'
folder = 'Au_9-17-19/'
subfolder = 'Slices/'
z_slice = '15'

path = directory + folder + subfolder

img = np.load(path + 'Bin6_Slice' + z_slice + '.npy')
vials = np.load(directory + folder + 'Vial_Masks.npy')
#bg = grm.background_ROI(img)

concentration = [4, 3, 2, 1, 0.5, 0]
tot = len(concentration)
CNR = np.zeros(tot)
CNR_err = np.zeros(tot)

mean_w = np.nanmean(img*vials[0])
std_w = np.nanstd(img*vials[0])

#mean_bg = np.nanmean(img*bg)
#std_bg = np.nanstd(img*bg)

for i in np.arange(tot-1):
    mean = np.nanmean(vials[i+1]*img)
    std = np.nanstd(vials[i+1]*img)
    CNR[i] = (mean - mean_w)/std_w
    CNR_err[i] = np.sqrt(std**2 + std_w**2)/std_w

CNR[tot-1] = 0
CNR_err[tot-1] = np.sqrt(std_w**2 + std_w**2)/std_w

# Construct data for a linear fit of the data
coeffs = np.polyfit(concentration, CNR, 1)

# Calculate y points from the fit above
xpts = np.linspace(-1, 6, 50)
y_all = coeffs[0] * xpts + coeffs[1]

# r-squared
p = np.poly1d(coeffs)
# fit values, and mean
yhat = p(concentration)  #
ybar = np.sum(CNR) / len(CNR)  # average value of y
ssreg = np.sum((yhat - ybar) ** 2)  # or sum([ (yihat - ybar)**2 for yihat in yhat])
sstot = np.sum((CNR - ybar) ** 2)  # or sum([ (yi - ybar)**2 for yi in y])
r_sq = ssreg / sstot
r_sq = '%.3f' % r_sq
r_squared = str(r_sq)

ypts = 4 * np.ones(50)

sns.set()
sns.set_style({'ytick.direction': 'out', 'ytick.left': True, 'xtick.direction': 'out', 'xtick.bottom': True})

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].imshow(img, cmap='gray', vmin=-500, vmax=1000)
ax[0].grid(False)
ax[0].set_xticks([4, 60, 116])
ax[0].set_xticklabels([-1.5, 0, 1.5], fontsize=16)
ax[0].set_yticks([4, 60, 116])
ax[0].set_yticklabels([-1.5, 0, 1.5], fontsize=16)
ax[0].set_xlabel('x (cm)', fontsize=20)
ax[0].set_ylabel('y (cm)', fontsize=20)
ax[0].set_title('Gold Phantom CT Scan', fontsize=25)
# Get the ROIs
for vial in vials:
    r, c = np.where(vial == 1)
    center = (int(np.median(c)), int(np.median(r)))
    circ = plt.Circle(center, radius=6, fill=False, edgecolor='red')
    ax[0].add_artist(circ)

ax[1].plot(xpts, y_all, color='midnightblue')
ax[1].errorbar(concentration, CNR, yerr=CNR_err, fmt='none', capsize=3, color='midnightblue')
ax[1].plot(xpts, ypts, color='red', ls='--')
ax[1].legend(['Linear Fit of CNR', 'Rose Criterion'], fontsize=16, fancybox=True, shadow=True)
ax[1].set_title('CNR Linearity with Gold \nContrast Concentration', fontsize=25)
ax[1].set_xlim([-0.1, 4.1])
ax[1].set_ylim([-5, 160])
ax[1].tick_params(axis='both', color='white', labelsize=16)
ax[1].set_xlabel('Gold Concentration (wt %)', fontsize=20)
ax[1].set_ylabel('CNR', fontsize=20)
ax[1].annotate("$R^2$ = " + r_squared, xy=(1, 0), xycoords='axes fraction', xytext=(-10, 30),
               textcoords='offset pixels', horizontalalignment='right', verticalalignment='bottom', fontsize=16)

asp = np.diff(ax[1].get_xlim())[0] / np.diff(ax[1].get_ylim())[0]
ax[1].set_aspect(asp)
plt.subplots_adjust(wspace=0.3)

plt.show()


#%%
directory = 'D:/Research/Python Data/Spectral CT/'
folder = 'AuGdLu_5-27-19/'
subfolder = 'K-Edge/'

dat21 = 'Bin2-1_Slice16.npy'
dat32 = 'Bin3-2_Slice16.npy'
dat43 = 'Bin4-3_Slice16.npy'
img21 = np.load(directory + folder + subfolder + dat21)
img32 = np.load(directory + folder + subfolder + dat32)
img43 = np.load(directory + folder + subfolder + dat43)


vial0 = np.load(directory + folder + 'Vial0_MaskMatrix.npy')
vial1 = np.load(directory + folder + 'Vial1_MaskMatrix.npy')
vial2 = np.load(directory + folder + 'Vial2_MaskMatrix.npy')
vial3 = np.load(directory + folder + 'Vial3_MaskMatrix.npy')

bg = np.nanmean(img*vial0)
bgst = np.nanstd(img*vial0)

CNR_Gd = abs((np.nanmean(img21*vial2)) - np.nanmean(img21*vial0))/np.nanstd(img21*vial0)

CNR_Lu = abs((np.nanmean(img32*vial3)) - np.nanmean(img32*vial0))/np.nanstd(img32*vial0)

CNR_Au = abs((np.nanmean(img43*vial1)) - np.nanmean(img43*vial0))/np.nanstd(img43*vial0)
print(CNR_Lu, CNR_Gd, CNR_Au)

#%% Plot gold and water mass attenuation
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
directory = 'D:/Research/Bin Optimization/'
folder = 'Beam Spectrum/'

spectrum = np.load(directory + folder + 'energy_spectrum_120kV.npy')
#au = np.load(directory + 'Au.npy')
#gd = np.load(directory + 'Gd.npy')
bi = np.load(directory + 'Bi.npy')
i = np.load(directory + 'I.npy')
h2o = np.load(directory + 'H2O.npy')
energies = spectrum[:, 0]
energies = energies[2:-1:3]
energies = 1000*energies

sns.set()
fig = plt.figure(figsize=(13, 7))
plt.semilogy(energies, bi, color='green')
#plt.semilogy(energies, gd, color='crimson')
plt.semilogy(energies, i, color='crimson')
plt.semilogy(energies, h2o, color='dodgerblue')
ones = np.ones(50)
y_vals = np.linspace(0, 20000, 50)

#plt.plot(16*ones, y_vals, color='crimson', ls='--')
#plt.plot(33*ones, y_vals, color='crimson', ls='--')
#plt.plot(50*ones, y_vals, color='crimson', ls='--')
#plt.plot(71*ones, y_vals, color='crimson', ls='--')
#plt.plot(81*ones, y_vals, color='crimson', ls='--')
#plt.plot(91*ones, y_vals, color='crimson', ls='--')
plt.xlim([10.275, 120])
plt.ylim([0.1, 5E2])
plt.title('Mass Attenuation Coefficient vs. X-ray Energy', fontsize=25)
plt.ylabel(r"$\mu / \rho$ $(cm^2 / g)$", fontsize=20)
plt.xlabel('Energy (keV)', fontsize=20)
plt.tick_params(labelsize=18)
#plt.annotate('K-edge (80.7 keV)', xy=(1, 0), xycoords='axes fraction', xytext=(-50, 320),
#               textcoords='offset pixels', horizontalalignment='right', verticalalignment='bottom', fontsize=18)
plt.legend(['Bismuth', 'Iodine', 'Water'], fontsize=20, fancybox=True, shadow=True)
plt.show()

#%% Plot beam spectra for DECT
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
atten_folder = 'D:/Research/Attenuation Data/NPY Attenuation/'
spectra_folder = 'D:/Research/Attenuation Data/NPY Spectra/'

# Matrices for the scattering data based on Z (rows) and E (columns)
energies = np.load(atten_folder + 'Z04.npy')

# Load the beam energies and weights
energies = energies[:, 0]
wts_low = np.load(spectra_folder + '40kVp_weights.npy')
wts_high = np.load(spectra_folder + '80kVp_weights.npy')

low_min = np.min(wts_low)
high_min = np.min(wts_high)
low_rng = np.max(wts_low) - low_min
high_rng = np.max(wts_high) - high_min

wts_low = 10000*np.divide(np.subtract(wts_low, low_min), low_rng)
wts_high = (80/40)**2*10000*np.divide(np.subtract(wts_high, high_min), high_rng)
energies = np.multiply(energies, 1000)

fig = plt.figure(figsize=(13, 7))
plt.plot(energies, wts_low, color='midnightblue')
plt.plot(energies, wts_high, color='crimson')
plt.title('DECT Spectra', fontsize=25)
plt.ylabel('Number of Photons', fontsize=20)
plt.xlabel('Energy (keV)', fontsize=20)
plt.xlim([1, 100])
plt.ylim([0, 41000])
plt.tick_params(labelsize=18)
plt.legend(['40 kVp', '80 kVp'], fontsize=18, fancybox=True, shadow=True)
plt.show()

#%% Plot the spectrum for spectral CT
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
directory = 'D:/Research/Bin Optimization/'
file = np.load(directory + '/corrected-spectrum_120kV.npy')
energies = file[:, 0]
weights = file[:, 1]

energies = 1000*energies
wmin = np.min(weights)

rng = np.max(weights) - wmin

weights = 10000*np.divide(np.subtract(weights, wmin), rng)
ones = np.ones(50)
y_vals = np.linspace(0, 20000, 50)

fig = plt.figure(figsize=(13, 7))
plt.plot(energies, weights, color='midnightblue')
plt.plot(16*ones, y_vals, color='crimson', ls='--')
plt.plot(33*ones, y_vals, color='crimson', ls='--')
plt.plot(50*ones, y_vals, color='crimson', ls='--')
plt.plot(71*ones, y_vals, color='crimson', ls='--')
plt.plot(81*ones, y_vals, color='crimson', ls='--')
plt.plot(91*ones, y_vals, color='crimson', ls='--')
plt.title('120 kVp Spectrum', fontsize=25)
plt.ylabel('Number of Photons', fontsize=20)
plt.xlabel('Energy (keV)', fontsize=20)
plt.xlim([10, 120])
plt.ylim([0, 10500])
plt.tick_params(labelsize=18)
plt.legend(['Spectrum', 'Energy bin thresholds'], fontsize=18, fancybox=True, shadow=True)
plt.show()

