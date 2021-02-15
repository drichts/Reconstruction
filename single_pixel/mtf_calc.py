import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import seaborn as sns

speed = 200
v = 3
num = 50

folder = r'D:\OneDrive - University of Victoria\Research\Single Pixel\2021_02_03\Smooth Data'
file = f'{speed}-smooth-v{v}-{num}.npy'
# file = f'{speed}-smooth-v{v}.npy'

esf = np.load(os.path.join(folder, file))
der = np.array([-0.5,  0, 0.5])

# plt.plot(esf[1])
# plt.show()
# plt.pause(7)
# plt.close()
#
# low = int(input('Low number:'))
# high = int(input('High number:'))
low = 60
high = 178

plt.plot(esf[low:high])
plt.show()
plt.pause(1)
plt.close()

# lsf = np.abs(np.convolve(esf[low:high], der))
lsf1 = np.diff(esf[low:high])

# plt.plot(lsf[10:-10])
# plt.show()
# plt.pause(1)
# plt.close()

plt.plot(lsf1[10:-10])
plt.show()
plt.pause(1)
plt.close()


M = len(lsf1)

x1 = low
x2 = high

# Get the frequencies to plot the MTF of the cathode edge over
freq = 0.5*np.linspace(0, M/2, int(M/2))
# freq = freq/(2*M)*(x2-x1)/(esf[0, x2]-esf[0, x1])


# Calculate the MTF of the cathode side
mtf = fft(lsf1)
mtf = np.absolute(mtf)
mtf = mtf[0:int(M/2)]
mtf = np.divide(mtf, np.max(mtf))

idx = (np.abs(mtf[0:3] - 0.1)).argmin()

sns.set_style('whitegrid')
fig = plt.figure(figsize=(7, 5))
# plt.plot(freq[0:int(len(freq)/1.75)], mtf[0:int(len(freq)//1.75)])
plt.plot(freq[0:10], mtf[0:10])
plt.title(rf'MTF ({speed} $\mu$m/s speed, {num} points averaged)', fontsize=15)
plt.xlabel(f'Spatial frequency (lp/mm)\nLimiting frequency = {freq[idx]:.2f} lp/mm', fontsize=14)
plt.ylabel('MTF(f)', fontsize=14)
plt.tick_params(labelsize=12)
plt.xlim([0, 4])
plt.subplots_adjust(bottom=0.2)
plt.show()

plt.savefig(rf'D:\OneDrive - University of Victoria\Research\Single Pixel\2021_02_03\MTF\{speed}-v{v}-{num}.png', dpi=fig.dpi)

