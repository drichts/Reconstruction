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

fig, ax = plt.subplots(4, 1, figsize=(10, 4))
path = directory + folder

sec_spectra1w = glob.glob(directory + folder + 'SEC*mA*1w*')
cc_spectra1w = glob.glob(directory + folder + 'CC*mA*1w*')
print(cc_spectra1w)
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
    ax[order[0], order[1]].legend(['cc 1w', 'cc 4w', 'sec 1w', 'sec 4w'], fontsize=18)
    ax[order[0], order[1]].set_xlabel('Energy (AU)', fontsize=20)
    ax[order[0], order[1]].set_ylabel('Counts', fontsize=20)
    ax[order[0], order[1]].set_title(titles[idx], fontsize=20)
    ax[order[0], order[1]].tick_params(labelsize=18)
    ax[order[0], order[1]].set_ylim([0, np.max(np.load(cc_spectra1w[idx]))*1.1])
    ax[order[0], order[1]].set_xlim([0, len(np.load(cc_spectra1w[idx]))])
plt.subplots_adjust(top=0.95, bottom=0.1, left=0.1, right=0.95, hspace=0.40, wspace=0.40)
plt.show()
#plt.savefig('C:/Users/10376/Documents/IEEE Abstract/Figures/Spectra/xray_spectra.png', dpi=fig.dpi)

#%% Plot spectra for the two isotopes

folder = '/Spectra/'
sns.set_style('whitegrid')
colors = ['black', 'mediumseagreen', 'mediumblue', 'crimson']
titles = ['Co57 Spectrum', 'Am241 Spectrum']

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
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

plt.show()
#plt.savefig('C:/Users/10376/Documents/IEEE Abstract/Figures/Spectra/ri_spectra.png', dpi=fig.dpi)


#%%  Flat field
sns.set_style('white')
folder = 'C:/Users/10376/Documents/IEEE Abstract/Analysis Data/Flat Field/'

x = np.load(folder + 'flatfield_4wA0.npy')

plt.imshow(x[7, 100])
plt.show()


