import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline


###################################################################################################
# TEST 2: High Contrast Spatial Resolution
# A square ROI is placed within the 1.6, 1.3, 1.0, and 0.8mm bar patterns
# and a standard deviation measurement is taken.  If the STD is above a value
# of ____ the bar pattern is considered resolvable.

# Generate MTF from standard deviations of bar patterns according to
# 'A Practical Method to Measure the MTF of CT Scanners' (Droege & Morin, 1982)

bar_size = np.array([1, 0.75, 0.66, 0.5, 0.33, 0.25])  # [mm] spot sizes for CT resolution phantom

# calculate the frequency in line pairs per mm or use the known freq
freq = 1 / (2 * bar_size)
# freq = [0.2, 0.25, 0.45, 0.76]  # MV PipsPro
# freq = [0.66, 0.98, 1.50]  # kV PipsPro


# M' is the measured noise (STD) in each of the bar patterns, uncorrected
M_prime = np.array([std_16, std_13, std_10, std_08, std_06, std_05])

# squared noise using ROIs of air and plastic
# N^2 = (Np^2+Nw^2)/2
noise_2 = (contrast_results['Plexi STD'] ** 2 + contrast_results['Water STD'] ** 2) / 2
# correct std of bar patterns for noise using the water ROI, M = sqrt(M'^2-N^2)
M = []
for std in M_prime:
    if std ** 2 > noise_2:
        M.append(np.sqrt(std ** 2 - noise_2))
    else:
        M.append(0)

# M = np.array([np.sqrt(std_16**2-noise_2),np.sqrt(std_13**2-noise_2),np.sqrt(std_10**2-noise_2),
#             np.sqrt(std_08**2-noise_2),np.sqrt(std_06**2-noise_2),np.sqrt(std_05**2-noise_2)])

# M0 = (CT1-CT2)/2: |Air - Plastic|/2
M0 = contrast_results['Contrast'] / 2

# MTF = (pi*sqrt(2)/4)*(M/M0)
MTF = (np.pi * np.sqrt(2) / 4) * (M / M0)

# Quadratic interpolation between freq = 0 and freq = 3.125 (first 2 measured mtf points)
MTF_first2 = [1, MTF[0]]
freq_first2 = [0, freq[0]]
z = np.polyfit(freq_first2, MTF_first2, 2)
f = np.poly1d(z)
freq_first4 = np.array([0, 1, 2, 3.125])
MTF_first4 = f(freq_first4)

# Cubic Spline fit of measured MTF
MTF_0 = [1, MTF_first4[1], MTF_first4[2], MTF[0], MTF[1], MTF[2], MTF[3], MTF[4],
         MTF[5]]  # experimental MTF with MTF = 1 at 0 frequency
freq_0 = [0, 1, 2, freq[0], freq[1], freq[2], freq[3], freq[4], freq[5]]
mtf_fit = CubicSpline(freq_0, MTF_0)
fit_freq = np.linspace(0, 10)