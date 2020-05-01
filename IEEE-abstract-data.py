import numpy as np
import general_OS_functions as gof
import matplotlib.pyplot as plt
import seaborn as sns


def get_median_pixels(data, time=10, energybin=0, view=0):

    matrix = data[time, energybin, view]  # Grab one capture of the corresponding bin and view
    med_idx = np.argwhere(matrix == (np.percentile(matrix, 50, interpolation='nearest')))

    return np.squeeze(med_idx)


def get_spectrum(data, pixel, energybin=0, view=0):
    num_steps = np.arange(np.shape(data)[0])
    steps = np.squeeze(data[:, energybin, view, pixel[0], pixel[1]])
    return num_steps, steps


#%%
path = r'C:\Users\10376\Documents\IEEE Abstract\Raw Data\1w_2mA_SPECTRUM.npy'

dat = np.load(path)
x, y = get_spectrum(dat, [12, 17])

plt.plot(x, y)
plt.show()
#%%
b = 6
v = 0
x = get_median_pixels(dat)
print(x)

#print(np.median(dat[b, v]))
#print(dat[b, v, x[0], x[1]])
#print()

#%%
for b in np.arange(12):
    print(b)
    x = get_median_pixels(dat, bin=b, view=v)
    print(x)
    print(np.median(dat[b, v]))
    print(dat[b, v, x[0], x[1]])
    print()
