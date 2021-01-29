import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from shutil import copyfile


files = glob(r'D:\OneDrive - University of Victoria\Research\Single Pixel\ESF_renamed\*')

# for file in files:
#     data = open(file).read().split()
#     times = np.array(data[1::2], dtype='float') / 1000
#     data = np.array(data[2::2], dtype='float')
#
#     print(file)
#
#     plt.plot(times, data)
#     plt.show()
#     plt.pause(0.25)
#     plt.close()

esf = np.zeros([2, len(files)])
for i, file in enumerate(files):
    data = open(file).read().split()
    data = np.array(data[2::2], dtype='float')

    esf[0, i] = np.mean(data)
    esf[1, i] = np.std(data)

# np.save(r'D:\OneDrive - University of Victoria\Research\Single Pixel\ESF_renamed\esf_data.npy', esf)
#
fig = plt.figure(figsize=(8, 6))
plt.plot(np.arange(len(files))*0.02, esf[0]*1E9)
plt.xlabel('Distance (mm)', fontsize=14)
plt.ylabel('Average signal (nA)', fontsize=14)
plt.tick_params(labelsize=12)
plt.title('Step-wise ESF', fontsize=15)
plt.show()

