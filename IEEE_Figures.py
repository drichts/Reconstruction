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
path = directory + folder + 'Energy/'

sec_energies1w = glob.glob(directory + folder + '/Energy/SEC*mA*1w*')
cc_energies1w = glob.glob(directory + folder + '/Energy/CC*mA*1w*')

sec_spectra1w = glob.glob(directory + folder + '/Spectra/SEC*mA*1w*')
cc_spectra1w = glob.glob(directory + folder + '/Spectra/CC*mA*1w*')

sec_energies4w = glob.glob(directory + folder + '/Energy/SEC*mA*4w*')
cc_energies4w = glob.glob(directory + folder + '/Energy/CC*mA*4w*')

sec_spectra4w = glob.glob(directory + folder + '/Spectra/SEC*mA*4w*')
cc_spectra4w = glob.glob(directory + folder + '/Spectra/CC*mA*4w*')

orders = np.array([[1, 0], [1, 1], [0, 0], [0, 1]])

for idx, order in enumerate(orders):
    print(idx)
    print(order)
    ax[order[0], order[1]].plot(np.load(cc_energies1w[idx]), np.load(cc_spectra1w[idx]), color=colors[0])
    ax[order[0], order[1]].plot(np.load(cc_energies4w[idx]), np.load(cc_spectra4w[idx]), color=colors[1])
    ax[order[0], order[1]].plot(np.load(sec_energies1w[idx]), np.load(sec_spectra1w[idx]), color=colors[2])
    ax[order[0], order[1]].plot(np.load(sec_energies4w[idx]), np.load(sec_spectra4w[idx]), color=colors[3])
    ax[order[0], order[1]].legend(['cc 1w', 'cc 4w', 'sec 1w', 'cc 4w'])
    ax[order[0], order[1]].set_xlabel('Energy (AU)')
    ax[order[0], order[1]].set_ylabel('Counts')
    ax[order[0], order[1]].set_title(titles[idx])

#plt.show()

