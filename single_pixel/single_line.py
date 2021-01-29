import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

directory = r'D:\OneDrive - University of Victoria\Research\Single Pixel'

files = glob(os.path.join(directory, '*'))
file = files[4]
print(file)

filename = file.split(directory)[1]

data = open(file).read().split()
times = np.array(data[1::2], dtype='float') / 1000
data = np.array(data[2::2], dtype='float') * 1E9

#%%

title = 'Rate 100 um per s, source far from object and detector'

fig = plt.figure(figsize=(6, 5))
plt.plot(times[:-20], data[:-20])
plt.title(title)
plt.xlabel('Time (s)')
plt.ylabel('Current (nA)')
plt.show()
plt.savefig(os.path.join(directory, title + '.png'), dpi=fig.dpi)
