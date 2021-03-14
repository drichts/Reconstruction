import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import seaborn as sns

speed = 200
v = 3
num = 50

# folder = r'D:\OneDrive - University of Victoria\Research\Single Pixel\2021_02_03\Smooth Data'
# file = f'{speed}-smooth-v{v}-{num}.npy'
# # file = f'{speed}-smooth-v{v}.npy'
#
# esf = np.load(os.path.join(folder, file))
# der = np.array([-0.5,  0, 0.5])

# plt.plot(spatial_resolution[1])
# plt.show()
# plt.pause(7)
# plt.close()
#
# low = int(input('Low number:'))
# high = int(input('High number:'))
# low = 60
# high = 178
#
# plt.plot(esf[low:high])
# plt.show()
# plt.pause(1)
# plt.close()
#
# # lsf = np.abs(np.convolve(spatial_resolution[low:high], der))
# lsf1 = np.diff(esf[low:high])
#
# # plt.plot(lsf[10:-10])
# # plt.show()
# # plt.pause(1)
# # plt.close()
#
# plt.plot(lsf1[10:-10])
# plt.show()
# plt.pause(1)
# plt.close()

file = r'D:\OneDrive - University of Victoria\Research\Single Pixel\PSF_theoretical.txt'

data = open(file).read().split()
xpts = np.array(data[0::2], dtype='float')
data = np.array(data[1::2], dtype='float')

# xpts = np.linspace(0, 0.5, 1000)
# data = np.sin(40 * 2 * np.pi * xpts) + 0.5 * np.sin(90 * 2 * np.pi * xpts)

#%%
lsf1 = data

# Number of data points
M = len(lsf1)

sns.set_style('whitegrid')
fig = plt.figure(figsize=(7, 5))
plt.plot(xpts, lsf1)
plt.ylabel('Dose (Gy)', fontsize=14)
plt.xlabel(r'Distance ($\mu$m)', fontsize=14)
plt.title('Dose profile across the crystal', fontsize=15)
plt.tick_params(labelsize=12)
plt.show()
plt.savefig(r'D:\OneDrive - University of Victoria\Research\Single Pixel\dose_profile.png', dpi=fig.dpi)
plt.pause(1)
plt.close()

csv_array = np.zeros([M+1, 2])
csv_array[1:, 0] = xpts
csv_array[1:, 1] = data
csv_array = csv_array.astype(str)

leg = ['Distance (um)', 'Dose (Gy)']
csv_array[0] = leg

np.savetxt(r'D:\OneDrive - University of Victoria\Research\Single Pixel\dose_profile.csv', csv_array, delimiter=',', fmt='%s')

# Sampling interval
T = xpts[1] - xpts[0]

# Get the frequencies to plot the MTF of the cathode edge over
freq = np.linspace(0, 1 / T, M)
freq = freq[0:int(M/2)] / 2 * 1000  # Switch to lp/mm from 1/um
# freq = freq/(2*M)*(x2-x1)/(spatial_resolution[0, x2]-spatial_resolution[0, x1])




# Calculate the MTF of the cathode side
mtf = fft(lsf1)
mtf = np.absolute(mtf)
mtf = mtf[0:int(M/2)]
mtf = np.divide(mtf, np.max(mtf))

idx = (np.abs(mtf[0:3] - 0.1)).argmin()

sns.set_style('whitegrid')
fig = plt.figure(figsize=(7, 5))
# plt.plot(freq[0:int(len(freq)/1.75)], mtf[0:int(len(freq)//1.75)])
plt.plot(freq, mtf)
plt.title(rf'Theoretical MTF', fontsize=15)
plt.xlabel(f'Spatial frequency (lp/mm)\nLimiting frequency = {freq[idx]:.2f} lp/mm', fontsize=14)
plt.ylabel('MTF', fontsize=14)
plt.tick_params(labelsize=12)
plt.xlim([0, 100])
plt.subplots_adjust(bottom=0.2)
plt.show()

csv_array = np.zeros([len(freq)+1, 2])
csv_array[1:, 0] = freq
csv_array[1:, 1] = mtf
csv_array = csv_array.astype(str)

leg = ['Frequency (lp/mm)', 'MTF']
csv_array[0] = leg

np.savetxt(r'D:\OneDrive - University of Victoria\Research\Single Pixel\mtf.csv', csv_array, delimiter=',', fmt='%s')

plt.savefig(rf'D:\OneDrive - University of Victoria\Research\Single Pixel\theoretical_mtf.png', dpi=fig.dpi)

