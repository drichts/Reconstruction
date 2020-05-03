import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import re


directory = r'C:\Users\10376\Documents\IEEE Abstract\Analysis Data'


#%% Plot spectra for all tube currents

folder = '/Spectra/'
sns.set_style('whitegrid')
colors = ['black', 'mediumseagreen', 'mediumblue', 'crimson']
titles = ['10mA Spectrum', '25mA Spectrum', '2mA Spectrum', '5mA Spectrum']

fig, ax = plt.subplots(2, 2, figsize=(8, 8))
path = directory + folder

sec_spectra1w = glob.glob(directory + folder + 'SEC*mA*1w*')
cc_spectra1w = glob.glob(directory + folder + 'CC*mA*1w*')

sec_spectra4w = glob.glob(directory + folder + 'SEC*mA*4w*')
cc_spectra4w = glob.glob(directory + folder + 'CC*mA*4w*')

energies = np.arange(221)

orders = np.array([[1, 0], [1, 1], [0, 0], [0, 1]])

for idx, order in enumerate(orders):
    print(idx)
    print(order)
    ax[order[0], order[1]].plot(energies, np.load(cc_spectra1w[idx]), color=colors[0])
    ax[order[0], order[1]].plot(energies, np.load(cc_spectra4w[idx]), color=colors[1])
    ax[order[0], order[1]].plot(energies, np.load(sec_spectra1w[idx]), color=colors[2])
    ax[order[0], order[1]].plot(energies, np.load(sec_spectra4w[idx]), color=colors[3])
    ax[order[0], order[1]].legend(['cc 1w', 'cc 4w', 'sec 1w', 'sec 4w'])
    ax[order[0], order[1]].set_xlabel('Energy (AU)')
    ax[order[0], order[1]].set_ylabel('Counts')
    ax[order[0], order[1]].set_title(titles[idx])

plt.show()

#%% Plot spectra for the two isotopes

folder = '/Spectra/'
sns.set_style('whitegrid')
colors = ['black', 'mediumseagreen', 'mediumblue', 'crimson']
titles = ['Co57 Spectrum', 'Am241 Spectrum']

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
path = directory + folder

sec_Co57_1w = glob.glob(directory + folder + '*SEC*Co57*1w*')
cc_Co57_1w = glob.glob(directory + folder + '*CC*Co57*1w*')
sec_Co57_4w = glob.glob(directory + folder + '*SEC*Co57*4w*')
cc_Co57_4w = glob.glob(directory + folder + '*CC*Co57*4w*')

sec_Am241_1w = glob.glob(directory + folder + '*SEC*Am241*1w*')
cc_Am241_1w = glob.glob(directory + folder + '*CC*Am241*1w*')
sec_Am241_4w = glob.glob(directory + folder + '*SEC*Am241*4w*')
cc_Am241_4w = glob.glob(directory + folder + '*CC*Am241*4w*')

energies = np.arange(231)

ax[0].plot(energies, np.load(cc_Co57_1w[0]), color=colors[0])
ax[0].plot(energies, np.load(cc_Co57_4w[0]), color=colors[1])
ax[0].plot(energies, np.load(sec_Co57_1w[0]), color=colors[2])
ax[0].plot(energies, np.load(sec_Co57_4w[0]), color=colors[3])
ax[0].legend(['cc 1w', 'cc 4w', 'sec 1w', 'sec 4w'])
ax[0].set_xlabel('Energy (AU)')
ax[0].set_ylabel('Counts')
ax[0].set_title(titles[0])

energies = np.arange(127)

ax[1].plot(energies, np.load(cc_Am241_1w[0]), color=colors[0])
ax[1].plot(energies, np.load(cc_Am241_4w[0]), color=colors[1])
ax[1].plot(energies, np.load(sec_Am241_1w[0]), color=colors[2])
ax[1].plot(energies, np.load(sec_Am241_4w[0]), color=colors[3])
ax[1].legend(['cc 1w', 'cc 4w', 'sec 1w', 'sec 4w'])
ax[1].set_xlabel('Energy (AU)')
ax[1].set_ylabel('Counts')
ax[1].set_title(titles[1])

#plt.show()


#%%