import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import make_interp_spline as spline
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from math import exp

folder = r'D:\Research\Python Data\Redlen\Attenuation'
fig = plt.figure(figsize=(4, 3))

steel = np.load(folder + '/steel.npy')

glass = np.load(folder + '/sodalimeglass.npy')
pmma = np.load(folder + '/PMMA.npy')
pp = np.load(folder + '/PP.npy')
water = np.load(folder + '/solid water.npy')
pvc = np.load(folder + '/PVC.npy')

steel[:, 1] = steel[:, 1] * 7.85
glass[:, 1] = glass[:, 1] * 2.52
pmma[:, 1] = pmma[:, 1] * 1.18
pp[:, 1] = pp[:, 1] * 0.855
water[:, 1] = water[:, 1] * 1.013
pvc[:, 1] = pvc[:, 1] * 1.38

sns.set_style('whitegrid')
plt.semilogy(pmma[:, 0]*1000, pmma[:, 1], color='mediumblue')
plt.semilogy(pp[:, 0]*1000, pp[:, 1], color='mediumseagreen')
plt.legend(['Acrylic', 'Polypropylene'], fontsize=11)

# plt.semilogy(steel[:, 0]*1000, steel[:, 1], color='orangered')
# plt.semilogy(glass[:, 0]*1000, glass[:, 1], color='mediumseagreen')
# plt.semilogy(pvc[:, 0]*1000, pvc[:, 1], color='mediumblue')
# plt.semilogy(water[:, 0]*1000, water[:, 1], color='k')
# plt.legend(['Steel', 'Glass', 'Conveyor belt', 'Solid water'], fontsize=11)

plt.xlim([20, 120])
#plt.ylim([1E-1, 1E0])

plt.xlabel('Energy (keV)', fontsize=12)
plt.ylabel(r'$\mu$ (cm$^{-1}$)', fontsize=12)
plt.subplots_adjust(left=0.29, right=0.90, bottom=0.16)
plt.tick_params(labelsize=11)
plt.plot()
plt.savefig(folder + '/Acrylic.png', dpi=fig.dpi)

#%%
sns.set_style('white')
folder = 'D:/Research/Bin Optimization/'

spectrum = np.load(folder + 'Beam Spectrum/corrected-spectrum_120kV.npy')

fig = plt.figure(figsize=(4, 3))
spectrum[:, 0] = spectrum[:, 0] * 1000

plt.plot(spectrum[:, 0], spectrum[:, 1], color='mediumblue')
plt.xlabel('Energy (keV)', fontsize=12)
plt.ylabel('Relative counts', fontsize=12)
plt.yticks([])
plt.tick_params(labelsize=11)
plt.xlim([20, 120])
plt.ylim([0, np.max(spectrum[:, 1] + np.median(spectrum[:, 1]))])
plt.subplots_adjust(bottom=0.2)
plt.plot()
plt.savefig(r'D:\Research\Python Data\Redlen\Attenuation\x-ray_spectrum.png', dpi=fig.dpi)

#%%
# file = '/PVC.txt'
#
# f = open(folder + file, 'rt')
#
# matrix = []
# for line in f:
#     col = line.split()
#     col = np.array(col)
#     matrix.append(col)
#
# matrix = np.array(matrix, dtype='float')
#
# np.save(folder + file[:-4] + '.npy', matrix)
