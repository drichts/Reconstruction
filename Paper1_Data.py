import numpy as np
import matplotlib.pyplot as plt
import sCT_Analysis as sct
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

directory = 'D:/Research/Python Data/Spectral CT/'
folders = ['Al_2.0_8-14-19', 'Al_2.0_10-17-19_3P', 'Al_2.0_10-17-19_1P',
           'Cu_0.5_8-14-19', 'Cu_0.5_9-13-19', 'Cu_0.5_10-17-19',
           'Cu_1.0_8-14-19', 'Cu_1.0_9-13-19', 'Cu_1.0_10-17-19',
           'Cu_0.5_Time_0.5_11-11-19', 'Cu_0.5_Time_0.1_11-4-19',
           'AuGd_width_5_12-2-19', 'AuGd_width_10_12-2-19', 'AuGd_width_14_12-2-19', 'AuGd_width_20_12-9-19']

good_slices = [[5, 19], [10, 18], [11, 18],
               [4, 15], [7, 15], [12, 19],
               [4, 14], [5, 16], [10, 19],
               [10, 19], [10, 18],
               [11, 19], [11, 19], [11, 19], [11, 19]]

#%% Scaled mass attenuation coefficients

folder = 'D:/Research/Bin Optimization/'
colors = ['orange', 'crimson', 'mediumseagreen', 'darkorchid', 'dodgerblue']  # Au, Dy, Lu, Gd, I

energies = np.load(folder + 'corrected-spectrum_120kV.npy')

# Mass attenuation of 5% mixtures and water
#au1 = np.load(folder + 'Au3P.npy')
au = np.load(folder + 'Au5P.npy')
#au = np.add(au1, 3*au2)/4
dy = np.load(folder + 'Dy5P.npy')
lu = np.load(folder + 'Lu5P.npy')
gd = np.load(folder + 'Gd5P.npy')
iod = np.load(folder + 'I5P.npy')
h2o = np.load(folder + 'H2O.npy')

# Denisities of the various mixtures
d_Au = 1.91  # 3P 1.55; 5P 1.91; 4P 1.71
d_Lu = 1.44
d_Dy = 1.38
d_Gd = 1.34
d_iod = 1.19  # Iohexol 5P 1.0057; 5P 1.19
d_h2o = 0.997

# Linear attenuation of the mixtures
au = np.multiply(au, d_Au)
lu = np.multiply(lu, d_Lu)
dy = np.multiply(dy, d_Dy)
gd = np.multiply(gd, d_Gd)
iod = np.multiply(iod, d_iod)
h2o = np.multiply(h2o, d_h2o)

# Get only the energy values
energies = energies[:, 0]
energies = 1000*energies  # Convert from MeV to keV

spectrum = np.load(folder + 'Cu0.5_spectrum.npy')

# Sum over a certain energy range
#au = np.multiply(au, spectrum)
#lu = np.multiply(lu, spectrum)
#dy = np.multiply(dy, spectrum)
#gd = np.multiply(gd, spectrum)
#iod = np.multiply(iod, spectrum)
#h2o = np.multiply(h2o, spectrum)

fig = plt.figure(figsize=(8, 8))

# Plot the elements
plt.semilogy(energies, au, color=colors[0])
plt.semilogy(energies, dy, color=colors[1])
plt.semilogy(energies, lu, color=colors[2])
plt.semilogy(energies, gd, color=colors[3])
plt.semilogy(energies, iod, color=colors[4])
plt.semilogy(energies, h2o, color='black')

# Plot vertical lines at the energy thresholds
ones = np.ones(50)
y_vals = np.linspace(-5, 1E4, 50)

bluepatch = mpatches.Patch(color='dodgerblue', label='I')
purplepatch = mpatches.Patch(color='darkorchid', label='Gd')
redpatch = mpatches.Patch(color='crimson', label='Dy')
greenpatch = mpatches.Patch(color='mediumseagreen', label='Lu')
orangepatch = mpatches.Patch(color='orange', label='Au')
blackpatch = mpatches.Patch(color='black', label='H2O')

plt.legend(handles=[bluepatch, purplepatch, redpatch, greenpatch, orangepatch, blackpatch], fancybox=True, shadow=False,
           fontsize=18)
plt.xlabel('Energy (keV)', fontsize=20, labelpad=5)
plt.ylabel(r"$\mu$ $(cm^{-1})$", fontsize=20)
plt.tick_params(labelsize=18)
plt.xlim([15, 120])
#plt.ylim([1E-1, 1E3])
#plt.ylim([0, 50])
plt.subplots_adjust(left=0.145, right=0.92)
plt.show()

#%%
# Get the sum of the weighted linear attenuation over a certain energy range
low, high = 54, 64
idx_low = (np.abs(energies-low)).argmin()
idx_high = (np.abs(energies-high)).argmin()

print(energies[idx_low], energies[idx_high])
sums = [np.sum(au[idx_low:idx_high+1]), np.sum(lu[idx_low:idx_high+1]),
        np.sum(dy[idx_low:idx_high+1]), np.sum(gd[idx_low:idx_high+1]),
        np.sum(iod[idx_low:idx_high+1])]
elems = ['Au', 'Lu', 'Dy', 'Gd', 'I']
print(sums)
for i in np.arange(5):
    maximum = np.amax(sums)
    idx = int(np.argwhere(sums == maximum))
    print(elems[idx], maximum)
    sums = np.delete(sums, idx)
    elems = np.delete(elems, idx)

#%% Plot weighted attenuation curves for one contrast agent for all filters

folder = 'D:/Research/Bin Optimization/'
dens = [1.91, 1.38, 1.44, 1.34, 1.19]  # densities of 5% solutions of Au, Dy, Lu, Gd, and I
#dens = [19.32, 8.55, 9.84, 7.895, 4.93]
elements = ['Au5P.npy', 'Dy5P.npy', 'Lu5P.npy', 'Gd5P.npy', 'I5P.npy']

energies = np.load(folder + 'corrected-spectrum_120kV.npy')

# Mass attenuation of 5% mixtures and water
mat = 1
att = np.load(folder + elements[mat])
h2o = np.load(folder + 'H2O.npy')

# Denisities of the various mixtures
d_Au = 1.91  # 3P 1.55; 5P 1.91; 4P 1.71
d_Lu = 1.44
d_Dy = 1.38
d_Gd = 1.34
d_iod = 1.19  # Iohexol 5P 1.0057; 5P 1.19
d_h2o = 0.997

# Linear attenuation of the mixtures
att = np.multiply(att, dens[mat])
h2o = np.multiply(h2o, d_h2o)

# Get only the energy values
energies = energies[:, 0]
energies = 1000*energies  # Convert from MeV to keV

Cu05_spectrum = np.load(folder + 'Cu0.5_spectrum.npy')
Cu1_spectrum = np.load(folder + 'Cu1.0_spectrum.npy')
Al_spectrum = np.load(folder + 'Al2.0_spectrum.npy')

Cu05_spectrum = np.multiply(Cu05_spectrum, 2.25)
Cu1_spectrum = np.multiply(Cu1_spectrum, 4.75)

# Multiply by the spectrums to get weighted linear attenuation at each energy
Cu05_att = np.multiply(Cu05_spectrum, att)
Cu1_att = np.multiply(Cu1_spectrum, att)
Al_att = np.multiply(Al_spectrum, att)

p_Cu05 = np.divide(Cu05_att, Cu05_spectrum)
p_Cu1 = np.divide(Cu1_att, Cu1_spectrum)
p_Al = np.divide(Al_att, Al_spectrum)

p_Cu05 = -np.log(p_Cu05)
p_Cu1 = -np.log(p_Cu1)
p_Al = -np.log(p_Al)

fig = plt.figure(figsize=(8, 8))

# Plot the elements
plt.plot(energies, Cu05_att, color='black', ls='-')
plt.plot(energies, Cu1_att, color='black', ls=':')
plt.plot(energies, Al_att, color='black', ls='--')


# Plot vertical lines at the energy thresholds
ones = np.ones(50)
y_vals = np.linspace(-5, 1E4, 50)

linepatch = mlines.Line2D([0], [0], color='black', lw=2, linestyle='-', label='0.5 mm Cu')
dashpatch = mlines.Line2D([0], [0], color='black', lw=2, linestyle='--', label='2.0 mm Al')
dotpatch = mlines.Line2D([0], [0], color='black', lw=2, linestyle=':', label='1.0 mm Cu')
plt.title(elements[mat])
plt.legend(handles=[dashpatch, linepatch, dotpatch], fancybox=True, shadow=False, fontsize=18)
plt.xlabel('Energy (keV)', fontsize=20, labelpad=5)
plt.ylabel(r"$\mu$ $(cm^{-1})$", fontsize=20)
plt.tick_params(labelsize=18)
plt.xlim([15, 120])
#plt.ylim([1E-1, 1E3])
#plt.ylim([0, 50])
plt.subplots_adjust(left=0.145, right=0.92)
plt.show()

#%% Compare the two Cu 0.5 spectra

import matplotlib.lines as mlines

folder = 'D:/Research/Bin Optimization/'
folder1 = folder + '/Npy Attenuation/'

spectrum = np.load(folder + 'Beam Spectrum/corrected-spectrum_120kV.npy')

energies1 = 1000*spectrum[:, 0]
fig = plt.figure(figsize=(8, 8))

spect1 = np.load(folder + 'Cu0.5_spectrum.npy')
spect2 = np.load(folder + 'spec_120kV_0p5mmCu.npy')

energies2 = 1000*spect2[:, 0]
spect2 = spect2[:, 1]

rng2 = spect2[23:52]
rng1 = spect1[24:76]

spect1 = np.divide(np.subtract(spect1, np.min(rng1)), (np.max(rng1)-np.min(rng1)))
spect2 = np.divide(np.subtract(spect2, np.min(rng2)), (np.max(rng2)-np.min(rng2)))

plt.plot(energies1, spect1, ls='-', color='black')
plt.plot(energies2, spect2, ls='-', color='red')

plt.legend(['Filtered', 'Monte Carlo'], fontsize=16)

plt.xlabel('Energy (keV)', fontsize=18, labelpad=10)
plt.ylabel('Relative counts', fontsize=18, labelpad=10)
plt.xlim([0, 200])
#plt.ylim([0, 7E-7])
#plt.ylim([0, 3.2E-7])
plt.tick_params(labelsize=15)
plt.show()

#%% Double-check values
f = 4  # 4 = 0.5 mm Cu
b = 4

vials = np.load(directory + folders[f] + '/Vial_Masks.npy')

img = np.load(directory + folders[f] + '/Slices/Bin' + str(b) + '_Slice15.npy')

vals = np.zeros(5)
for i in np.arange(1, 6):
    vals[i-1] = np.nanmean(img*vials[i])
elems = ['Au', 'Dy', 'Lu', 'Gd', 'I']
for i in np.arange(5):
    maximum = np.amax(vals)
    idx = int(np.argwhere(vals == maximum))
    print(elems[idx], maximum)
    vals = np.delete(vals, idx)
    elems = np.delete(elems, idx)


#%% Normalize the K-Edge Images

# Normalization values
Al_norm = np.zeros([4, 2])
Cu05_norm = np.zeros([4, 2])
Cu1_norm = np.zeros([4, 2])

for i in np.arange(4):

    # Order of the vials according to the bin
    vials = np.array([4, 2, 3, 1])

    # Find the norm values for each 5% value and 0% for each bin in each of the filters
    #Al_norm[i, :] = sct.find_norm_value(folders[0], good_slices[0], vials[i], i, directory=directory)
    #Cu05_norm[i, :] = sct.find_norm_value(folders[3], good_slices[3], vials[i], i, directory=directory)
    Cu1_norm[i, :] = sct.find_norm_value(folders[6], good_slices[6], vials[i], i, directory=directory)

    # Get the current linear fit coefficients for each filter
    #coeffs_Al = sct.linear_fit(Al_norm[i, 0], Al_norm[i, 1])
    #coeffs_Cu05 = sct.linear_fit(Cu05_norm[i, 0], Cu05_norm[i, 1])
    coeffs_Cu1 = sct.linear_fit(Cu1_norm[i, 0], Cu1_norm[i, 1])

    # Normalize the K-Edge images
    # Aluminum 2.0 mm
    #sct.norm_kedge(folders[0], coeffs_Al, i, directory=directory)  # 5%
    #sct.norm_kedge(folders[1], coeffs_Al, i, directory=directory)  # 3%
    #sct.norm_kedge(folders[2], coeffs_Al, i, directory=directory)  # 1%

    # Copper 0.5 mm
    #sct.norm_kedge(folders[3], coeffs_Cu05, i, directory=directory)  # 5%
    #sct.norm_kedge(folders[4], coeffs_Cu05, i, directory=directory)  # 3%
    #sct.norm_kedge(folders[5], coeffs_Cu05, i, directory=directory)  # 1%

    # Copper 1.0 mm
    #sct.norm_kedge(folders[6], coeffs_Cu1, i, directory=directory)  # 5%
    #sct.norm_kedge(folders[7], coeffs_Cu1, i, directory=directory)  # 3%
    #sct.norm_kedge(folders[8], coeffs_Cu1, i, directory=directory)  # 1%

    # Time Acquisitions
    #sct.norm_kedge(folders[9], coeffs_Cu05, i, directory=directory)  # 0.5s
    #sct.norm_kedge(folders[10], coeffs_Cu05, i, directory=directory)  # 0.1s

    # Bin Width
    #sct.norm_kedge(folders[11], coeffs_Cu05, i, directory=directory)  # 5, 5
    #sct.norm_kedge(folders[12], coeffs_Cu05, i, directory=directory)  # 10, 10
    #sct.norm_kedge(folders[13], coeffs_Cu05, i, directory=directory)  # 14, 14
    #sct.norm_kedge(folders[14], coeffs_Cu05, i, directory=directory)  # 8, 20

    # Copper 1.0 mm Wavelet trials
    sct.norm_kedge('Cu_1.0_9-13-19-Wavelet', coeffs_Cu1, i, directory=directory)
