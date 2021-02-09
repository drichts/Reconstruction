import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft

esf = np.load(r'D:\OneDrive - University of Victoria\Research\Single Pixel\esf_data.npy')
der = np.array([-0.25, -0.5,  1, 0.5, 0.25])
lsf = np.abs(np.convolve(esf[0], der))

M = len(lsf)

# Get the frequencies to plot the MTF of the cathode edge over
freq = np.linspace(0, M/2, int(M/2))
freq = freq/(2*M*0.02)

# Calculate the MTF of the cathode side
mtf = fft(lsf)
mtf = np.absolute(mtf)
mtf = mtf[0:int(M/2)]
mtf = np.divide(mtf, np.max(mtf))

plt.plot(freq, mtf)
plt.show()

print(freq[2])
