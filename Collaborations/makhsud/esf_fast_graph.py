import os
import numpy as np
import matplotlib.pyplot as plt

speed = 100
v = 1
folder = r'D:\OneDrive - University of Victoria\Research\Single Pixel'
file = 'keithley_laser_mtf_405nm_75mA_100microns_per_s_edge.txt'

# conv = False
conv = True

file = os.path.join(folder, file)

data = open(file).read().split()
times = np.array(data[1::2], dtype='float') / 1000 * (speed / 1000)
data = np.array(data[2::2], dtype='float') * 1E9

temp_data = np.zeros((2, len(times)))
temp_data[0] = times
temp_data[1] = data

if conv:
    num = 50
    average = np.ones(num)/num
    data = np.convolve(data, average)
    # save_data = np.zeros((2, len(data[0:-52])))
    # save_data[0] = times[2:-3]
    # save_data[1] = data[52:-52]
    np.save(os.path.join(folder, 'Smooth Data', f'{speed}-smooth-v{v}-{num}.npy'), data)

# fig = plt.figure(figsize=(8, 6))
# if conv:
#     plt.plot(times[2:-3], data[27:-27])
#     plt.title(rf'Moving at {speed} $\mu$m/s (Smoothed)', fontsize=15)
# else:
#     plt.plot(times, data)
#     plt.title(rf'Moving at {speed} $\mu$m/s', fontsize=15)
#
# plt.xlabel('Distance (mm)', fontsize=14)
# plt.ylabel('Signal (nA)', fontsize=14)
# plt.tick_params(labelsize=12)
#
# plt.show()

# if conv:
#     plt.savefig(os.path.join(folder, 'fig', f'{speed}um-conv-v{v}.png'), dpi=fig.dpi)
# else:
#     plt.savefig(os.path.join(folder, 'fig', f'{speed}um-v{v}.png'), dpi=fig.dpi)

