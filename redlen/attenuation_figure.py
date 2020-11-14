import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import make_interp_spline as spline
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from math import exp

folder = r'D:\OneDrive - University of Victoria\Research\Old Detector Data\Python Data\Redlen\Attenuation'
fig = plt.figure(figsize=(6, 6))

steel = np.load(folder + '/steel.npy')

glass = np.load(folder + '/sodalimeglass.npy')
pmma = np.load(folder + '/PMMA.npy')
pp = np.load(folder + '/PP.npy')
water = np.load(folder + '/solid water.npy')
#pvc = np.load(folder + '/PVC.npy')

soft = np.load(r'D:\OneDrive - University of Victoria\Research\Attenuation Data\Soft Tissue\Muscle.npy')

# steel[:, 1] = steel[:, 1] * 7.85
# glass[:, 1] = glass[:, 1] * 2.52
# pmma[:, 1] = pmma[:, 1] * 1.18
# pp[:, 1] = pp[:, 1] * 0.855
# water[:, 1] = water[:, 1] * 1.013
# pvc[:, 1] = pvc[:, 1] * 1.38

sns.set_style('whitegrid')
plt.semilogy(pmma[:, 0]*1000, pmma[:, 1])#, color='mediumblue')
plt.semilogy(pp[:, 0]*1000, pp[:, 1])#, color='mediumseagreen')
# plt.legend(['Acrylic', 'Polypropylene'], fontsize=11)
line = np.arange(1E-2, 1E4, 100)

plt.semilogy(steel[:, 0]*1000, steel[:, 1])#, color='orangered')
plt.semilogy(glass[:, 0]*1000, glass[:, 1])#, color='mediumseagreen')
#plt.loglog(pvc[:, 0]*1000, pvc[:, 1])#, color='mediumblue')
plt.semilogy(water[:, 0]*1000, water[:, 1])#, color='k')
plt.semilogy(soft[:, 0]*1000, soft[:, 1])
plt.legend(['Acrylic', 'Polypropylene', 'Steel', 'Glass', 'Solid water', 'Muscle'], fontsize=13, loc='upper right')

plt.semilogy(30*np.ones(len(line)), line, color='r', ls='--')
plt.semilogy(50*np.ones(len(line)), line, color='r', ls='--')
plt.semilogy(70*np.ones(len(line)), line, color='r', ls='--')
plt.semilogy(90*np.ones(len(line)), line, color='r', ls='--')
plt.semilogy(110*np.ones(len(line)), line, color='r', ls='--')

plt.xlim([10, 120])
plt.ylim([1E-2, 5E3])

plt.ylabel(r'$\mu/\rho$ cm$^2$/g', fontsize=17)
plt.xlabel('Energy (kev)', fontsize=17)
plt.tick_params(labelsize=13)
plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95)
plt.plot()
plt.savefig(r'D:\OneDrive - University of Victoria\Research\Attenuation Data\Soft Tissue\NDT_2.png', dpi=fig.dpi)
#plt.savefig(folder + '/Acrylic.png', dpi=fig.dpi)

#%%
sns.set_style('white')
folder = r'D:\OneDrive - University of Victoria\Research/Bin Optimization/'

spectrum = np.load(folder + 'Beam Spectrum/corrected-spectrum_120kV.npy')

fig = plt.figure(figsize=(6, 6))
spectrum[:, 0] = spectrum[:, 0] * 1000
line = np.arange(0, 10000, 100)
plt.plot(spectrum[:, 0], spectrum[:, 1], color='mediumblue')
plt.plot(30*np.ones(len(line)), line, color='r', ls='--')
plt.plot(50*np.ones(len(line)), line, color='r', ls='--')
plt.plot(70*np.ones(len(line)), line, color='r', ls='--')
plt.plot(90*np.ones(len(line)), line, color='r', ls='--')
plt.plot(110*np.ones(len(line)), line, color='r', ls='--')
plt.xlabel('Energy (keV)', fontsize=17)
plt.ylabel('Relative counts', fontsize=17)
plt.yticks([])
plt.tick_params(labelsize=13)
plt.xlim([20, 120])
plt.ylim([0, np.max(spectrum[:, 1] + np.median(spectrum[:, 1]))])
plt.subplots_adjust(bottom=0.2)
plt.plot()
# plt.savefig(r'D:\Research\Python Data\Redlen\Attenuation\x-ray_spectrum.png', dpi=fig.dpi)

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
