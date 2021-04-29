import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import seaborn as sns

speed = 10
N = 75

folder = r'D:\OneDrive - University of Victoria\Research\resolution\21-04-28_xray_mtf'
file = f'30kV__12p4mA_10um_s_edge.txt'
# file = f'{speed}-smooth-v{v}.npy'

file = os.path.join(folder, file)

data = open(file).read().split()
times = np.array(data[1::2], dtype='float') / 1000 * (speed / 1000)
esf = np.array(data[2::2], dtype='float') * 1E9
esf = np.convolve(esf, np.ones(N)/N, mode='valid')

# esf = np.load(os.path.join(folder, file))
der = np.array([-0.5,  0, 0.5])

# plt.plot(times, esf)
# plt.show()
# plt.pause(15)
# plt.close()

# low = int(input('Low number:'))
# high = int(input('High number:'))
# low = (np.abs(times - 0.69)).argmin()
# high = (np.abs(times - 1.6)).argmin()
# low = 0
# high = len(esf) - 1

fig = plt.figure(figsize=(7, 5))
plt.plot(esf)
plt.xlabel('Distance (mm)')
plt.ylabel('Current (nA)')
plt.title('ESF')
plt.show()
# plt.savefig(os.path.join(folder, f'ESF_half.png'), dpi=500)
# plt.pause(1)
# plt.close()

# lsf = np.abs(np.convolve(esf[low:high], der))
lsf = np.abs(np.diff(esf))

# plt.plot(lsf[10:-10])
# plt.show()
# plt.pause(1)
# plt.close()

fig = plt.figure(figsize=(7, 5))
plt.plot(lsf[10:-10])
plt.xlabel('Distance (mm)')
plt.ylabel('Current (nA)')
plt.title('LSF')
plt.show()
# plt.savefig(os.path.join(folder, f'LSF_half.png'), dpi=500)
plt.show()
# plt.pause(0.5)
# plt.close()

# file = r'D:\OneDrive - University of Victoria\Research\Single Pixel\PSF_theoretical.txt'
#
# data = open(file).read().split()
# xpts = np.array(data[0::2], dtype='float')
# data = np.array(data[1::2], dtype='float')

# xpts = np.linspace(0, 0.5, 1000)
# data = np.sin(40 * 2 * np.pi * xpts) + 0.5 * np.sin(90 * 2 * np.pi * xpts)

#%%
lsf1 = lsf

# Number of data points
M = len(lsf1)

sns.set_style('whitegrid')
# fig = plt.figure(figsize=(7, 5))
# plt.plot(xpts, lsf1)
# plt.ylabel('Dose (Gy)', fontsize=14)
# plt.xlabel(r'Distance ($\mu$m)', fontsize=14)
# plt.title('Dose profile across the crystal', fontsize=15)
# plt.tick_params(labelsize=12)
# plt.show()
# plt.savefig(r'D:\OneDrive - University of Victoria\Research\Single Pixel\dose_profile.png', dpi=fig.dpi)
# plt.pause(1)
# plt.close()
#
# csv_array = np.zeros([M+1, 2])
# csv_array[1:, 0] = xpts
# csv_array[1:, 1] = data
# csv_array = csv_array.astype(str)
#
# leg = ['Distance (um)', 'Dose (Gy)']
# csv_array[0] = leg
#
# np.savetxt(r'D:\OneDrive - University of Victoria\Research\Single Pixel\dose_profile.csv', csv_array, delimiter=',', fmt='%s')

# Sampling interval
T = times[1] - times[0]

# Get the frequencies to plot the MTF of the cathode edge over
freq = np.linspace(0, 1 / T, M)
# freq = freq[0:int(M/2)] / 2 * 1000  # Switch to lp/mm from 1/um
# freq = freq/(2*M)*(x2-x1)/(spatial_resolution[0, x2]-spatial_resolution[0, x1])
freq = freq[0:int(M/2)]



# Calculate the MTF of the cathode side
mtf = fft(lsf1)
mtf = np.absolute(mtf)
mtf = mtf[0:int(M/2)]
mtf = np.divide(mtf, np.max(mtf))

idx = (np.abs(mtf - 0.2)).argmin()

sns.set_style('whitegrid')
fig = plt.figure(figsize=(7, 5))
# plt.plot(freq[0:int(len(freq)/1.75)], mtf[0:int(len(freq)//1.75)])
plt.plot(freq, mtf)
plt.title(rf'Theoretical MTF', fontsize=15)
plt.xlabel(f'Spatial frequency (lp/mm)\nLimiting frequency = {freq[idx]:.2f} lp/mm', fontsize=14)
plt.ylabel('MTF', fontsize=14)
plt.tick_params(labelsize=12)
plt.xlim([0, 25])
plt.subplots_adjust(bottom=0.2)
plt.show()
plt.savefig(os.path.join(folder, f'MTF_2_{speed}.png'), dpi=500)

# csv_array = np.zeros([len(freq)+1, 2])
# csv_array[1:, 0] = freq
# csv_array[1:, 1] = mtf
# csv_array = csv_array.astype(str)
#
# leg = ['Frequency (lp/mm)', 'MTF']
# csv_array[0] = leg

# np.savetxt(r'D:\OneDrive - University of Victoria\Research\Single Pixel\mtf.csv', csv_array, delimiter=',', fmt='%s')

# plt.savefig(rf'D:\OneDrive - University of Victoria\Research\Single Pixel\theoretical_mtf.png', dpi=fig.dpi)

