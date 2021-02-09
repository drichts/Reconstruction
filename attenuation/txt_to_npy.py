import numpy as np
import os
from glob import glob

folder = r'D:\OneDrive - University of Victoria\Research\Attenuation Data\Soft Tissue'

files = glob(os.path.join(folder, '*.txt'))

for file in files:
    data = open(file, 'r')

    x = data.readlines()
    array = np.zeros((len(x), 2))
    for i, line in enumerate(x):
        line_data = line[7:].split('  ')
        array[i] = line_data[0:2]

    print(array)
    print()
    np.save(file.replace('txt', 'npy'), array)