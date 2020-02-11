import numpy as np
import matplotlib.pyplot as plt
import os

load_folder = 'D:/Research/Attenuation Data/Txt Attenuation/'
save_folder = 'D:/Research/Attenuation Data/NPY Attenuation/'

spectra_folder = 'D:/Research/Attenuation Data/NPY Spectra/'
energies = ['40kVp', '80kVp']  # Should be of the form '40kVp', '80kVp', etc.


dirs = os.listdir(load_folder)

for file in dirs:

    f = open(load_folder + file, 'rt')

    matrix = []
    for line in f:
        col = line.split()
        col = np.array(col)
        matrix.append(col)

    matrix = matrix[3:len(matrix)]
    matrix = np.array(matrix, dtype='float')
    file = file.replace('.txt', '.npy')

    np.save(save_folder + file, matrix)
    print(file)


## Load the spectra data
spectra_file_low = 'IMAGING_' + energies[0] + '_Spectra.npy'
spectra_file_high = 'IMAGING_' + energies[1] + '_Spectra.npy'

spectra_data_low = np.load(spectra_folder+spectra_file_low)
spectra_data_high = np.load(spectra_folder+spectra_file_high)

# Load the spectra information
spectra_energies = []  # Energies
spectra_weights_low = []  # Lower energy weights
spectra_weights_high = []  # Higher energy weights

# Get rid of the bin edges, just take the mid energy value and the number of photons in the bin
for m in np.arange(2, len(spectra_data_low), 3):
    spectra_energies.append(spectra_data_low[m, 0])

    spectra_weights_low.append(spectra_data_low[m, 1])
    spectra_weights_high.append(spectra_data_high[m, 1])

spectra_energies.append(spectra_data_low[len(spectra_data_low)-2, 0])

spectra_weights_low.append(spectra_data_low[len(spectra_data_low)-2, 1])
spectra_weights_high.append(spectra_data_high[len(spectra_data_high) - 2, 1])

spectra_energies = np.array(spectra_energies)

spectra_weights_low = np.array(spectra_weights_low)
spectra_weights_high = np.array(spectra_weights_high)

# Normalize the weights
sum_low = np.sum(spectra_weights_low)
sum_high = np.sum(spectra_weights_high)

spectra_weights_low = np.divide(spectra_weights_low, sum_low)
spectra_weights_high = np.divide(spectra_weights_high, sum_high)

np.save(spectra_folder + '40kVp_weights.npy', spectra_weights_low)
np.save(spectra_folder + '80kVp_weights.npy', spectra_weights_high)

plt.plot(spectra_energies, spectra_weights_low)
plt.title('40 kVp spectra')
plt.xlabel('Energy (MeV)')
plt.show()
plt.pause(2)
plt.close()

plt.plot(spectra_energies, spectra_weights_high)
plt.title('80 kVp spectra')
plt.xlabel('Energy (MeV)')
plt.show()
plt.pause(2)
plt.close()