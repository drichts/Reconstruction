import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import matplotlib.lines as mlines
import string


directory = r'C:\Users\10376\Documents\IEEE Abstract\Analysis Data'


#%% Plot spectra for all tube currents

folder = '/Spectra/'
sns.set_style('whitegrid')
colors = ['black', 'mediumseagreen', 'mediumblue', 'crimson']
titles = ['10mA Spectrum', '25mA Spectrum', '2mA Spectrum', '5mA Spectrum']

fig, ax = plt.subplots(4, 1, figsize=(5, 12))
ax1 = fig.add_subplot(111, frameon=False)
ax1.grid(False)
# Hide axes ticks
ax1.set_xticks([])
ax1.set_yticks([])
path = directory + folder

sec_spectra1w = glob.glob(directory + folder + 'SEC*mA*1w*')
cc_spectra1w = glob.glob(directory + folder + 'CC*mA*1w*')
print(cc_spectra1w)
sec_spectra4w = glob.glob(directory + folder + 'SEC*mA*4w*')
cc_spectra4w = glob.glob(directory + folder + 'CC*mA*4w*')

energies = np.arange(221)

#orders = np.array([[1, 0], [1, 1], [0, 0], [0, 1]])
orders = np.array([2, 3, 0, 1])

for idx, order in enumerate(orders):
    print(idx)
    print(order)
    ax[order].plot(energies, np.load(cc_spectra1w[idx]), color=colors[0])
    ax[order].plot(energies, np.load(cc_spectra4w[idx]), color=colors[1])
    ax[order].plot(energies, np.load(sec_spectra1w[idx]), color=colors[2])
    ax[order].plot(energies, np.load(sec_spectra4w[idx]), color=colors[3])
    #ax[order].legend(['cc 1w', 'cc 4w', 'sec 1w', 'sec 4w'], fontsize=16)
    ax[order].set_xlabel('Energy (AU)', fontsize=14)
    ax[order].set_ylabel('Counts', fontsize=14)
    ax[order].set_title(titles[idx], fontsize=14)
    ax[order].tick_params(labelsize=13)
    ax[order].set_ylim([0, np.max(np.load(cc_spectra1w[idx]))*1.1])
    ax[order].set_xlim([0, len(np.load(cc_spectra1w[idx]))])
    ax[order].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

bluepatch = mlines.Line2D([0], [0], color=colors[2], label='SEC 1w')
redpatch = mlines.Line2D([0], [0], color=colors[3], label='SEC 4w')
greenpatch = mlines.Line2D([0], [0], color=colors[1], label='CC 4w')
blackpatch = mlines.Line2D([0], [0], color=colors[0], label='CC 1w')
leg1 = plt.legend(handles=[blackpatch, greenpatch, bluepatch, redpatch], bbox_to_anchor=(0.45, -0.17),
                  loc='lower center', fancybox=True, ncol=4, shadow=False, fontsize=13,
                  handlelength=2, handletextpad=0.5, columnspacing=1)

# Add letters for subplots
ax = ax.flat
for n, ax in enumerate(ax):
    ax.text(-0.1, 1.1, string.ascii_lowercase[n], transform=ax.transAxes,
            size=20, weight='bold')

ax1.add_artist(leg1)
plt.subplots_adjust(top=0.95, bottom=0.15, left=0.15, right=0.90, hspace=1)
plt.show()
#plt.savefig('C:/Users/10376/Documents/IEEE Abstract/Figures/Spectra/xray_spectra.png', dpi=fig.dpi)

#%% Plot spectra for the two isotopes

folder = '/Spectra/'
sns.set_style('whitegrid')
colors = ['black', 'mediumseagreen', 'mediumblue', 'crimson']
titles = ['Co-57 Spectrum', 'Am-241 Spectrum']

fig, ax = plt.subplots(2, 1, figsize=(5, 6))
ax1 = fig.add_subplot(111, frameon=False)
ax1.grid(False)
# Hide axes ticks
ax1.set_xticks([])
ax1.set_yticks([])
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
ax[0].set_xlabel('Energy (AU)', fontsize=14)
ax[0].set_ylabel('Counts', fontsize=14)
ax[0].set_title(titles[0], fontsize=14)
ax[0].tick_params(labelsize=13)
ax[0].set_ylim([0, np.max(np.load(cc_Co57_1w[0]))*1.1])
ax[0].set_xlim([0, len(np.load(cc_Co57_1w[0]))])

energies = np.arange(127)

ax[1].plot(energies, np.load(cc_Am241_1w[0]), color=colors[0])
ax[1].plot(energies, np.load(cc_Am241_4w[0]), color=colors[1])
ax[1].plot(energies, np.load(sec_Am241_1w[0]), color=colors[2])
ax[1].plot(energies, np.load(sec_Am241_4w[0]), color=colors[3])
ax[1].set_xlabel('Energy (AU)', fontsize=14)
ax[1].set_ylabel('Counts', fontsize=14)
ax[1].set_title(titles[1], fontsize=14)
ax[1].tick_params(labelsize=13)
ax[1].set_ylim([0, np.max(np.load(cc_Am241_1w[0]))*1.1])
ax[1].set_xlim([0, len(np.load(cc_Am241_1w[0]))])

bluepatch = mlines.Line2D([0], [0], color=colors[2], label='SEC 1w')
redpatch = mlines.Line2D([0], [0], color=colors[3], label='SEC 4w')
greenpatch = mlines.Line2D([0], [0], color=colors[1], label='CC 4w')
blackpatch = mlines.Line2D([0], [0], color=colors[0], label='CC 1w')
leg1 = plt.legend(handles=[blackpatch, greenpatch, bluepatch, redpatch], bbox_to_anchor=(0.45, -0.23),
                  loc='lower center', fancybox=True, ncol=4, shadow=False, fontsize=13,
                  handlelength=2, handletextpad=0.5, columnspacing=1)
ax1.add_artist(leg1)

# Add letters for subplots
ax = ax.flat
for n, ax in enumerate(ax):
    ax.text(-0.1, 1.1, string.ascii_lowercase[n], transform=ax.transAxes,
            size=20, weight='bold')

plt.show()
plt.subplots_adjust(top=0.92, bottom=0.2, left=0.15, right=0.90, hspace=0.6)
#plt.savefig('C:/Users/10376/Documents/IEEE Abstract/Figures/Spectra/ri_spectra.png', dpi=fig.dpi)

#%% Plot 2 mA, 25 mA, and the two isotopes
folder = '/Spectra/'
sns.set_style('whitegrid')
colors = ['black', 'mediumseagreen', 'mediumblue', 'crimson']
titles = ['2mA Spectrum', '25mA Spectrum', 'Co-57 Spectrum', 'Am-241 Spectrum']

fig, ax = plt.subplots(4, 1, figsize=(5, 11))
ax1 = fig.add_subplot(111, frameon=False)
ax1.grid(False)
# Hide axes ticks
ax1.set_xticks([])
ax1.set_yticks([])
path = directory + folder

sec_021w = np.load(glob.glob(directory + folder + 'SEC*2mA*1w*')[0])
cc_021w = np.load(glob.glob(directory + folder + 'CC*2mA*1w*')[0])
sec_024w = np.load(glob.glob(directory + folder + 'SEC*2mA*4w*')[0])
cc_024w = np.load(glob.glob(directory + folder + 'CC*2mA*4w*')[0])

sec_251w = np.load(glob.glob(directory + folder + 'SEC*25mA*1w*')[0])
cc_251w = np.load(glob.glob(directory + folder + 'CC*25mA*1w*')[0])
sec_254w = np.load(glob.glob(directory + folder + 'SEC*25mA*4w*')[0])
cc_254w = np.load(glob.glob(directory + folder + 'CC*25mA*4w*')[0])

sec_Co57_1w = np.load(glob.glob(directory + folder + '*SEC*Co57*1w*')[0])
cc_Co57_1w = np.load(glob.glob(directory + folder + '*CC*Co57*1w*')[0])
sec_Co57_4w = np.load(glob.glob(directory + folder + '*SEC*Co57*4w*')[0])
cc_Co57_4w = np.load(glob.glob(directory + folder + '*CC*Co57*4w*')[0])

sec_Am241_1w = np.load(glob.glob(directory + folder + '*SEC*Am241*1w*')[0])
cc_Am241_1w = np.load(glob.glob(directory + folder + '*CC*Am241*1w*')[0])
sec_Am241_4w = np.load(glob.glob(directory + folder + '*SEC*Am241*4w*')[0])
cc_Am241_4w = np.load(glob.glob(directory + folder + '*CC*Am241*4w*')[0])

energies = np.arange(len(cc_021w))
ax[0].plot(energies, cc_021w, color=colors[0])
ax[0].plot(energies,  cc_024w, color=colors[1])
ax[0].plot(energies, sec_021w, color=colors[2])
ax[0].plot(energies, sec_024w, color=colors[3])
ax[0].set_xlabel('Energy (AU)', fontsize=14)
ax[0].set_ylabel('Counts', fontsize=14)
ax[0].set_title(titles[0], fontsize=14)
ax[0].tick_params(labelsize=13)
ax[0].set_ylim([0, np.max(cc_021w)*1.1])
ax[0].set_xlim([0, len(cc_021w)])
ax[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

ax[1].plot(energies, cc_251w, color=colors[0])
ax[1].plot(energies,  cc_254w, color=colors[1])
ax[1].plot(energies, sec_251w, color=colors[2])
ax[1].plot(energies, sec_254w, color=colors[3])
ax[1].set_xlabel('Energy (AU)', fontsize=14)
ax[1].set_ylabel('Counts', fontsize=14)
ax[1].set_title(titles[1], fontsize=14)
ax[1].tick_params(labelsize=13)
ax[1].set_ylim([0, np.max(cc_251w)*1.1])
ax[1].set_xlim([0, len(cc_251w)])
ax[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

energies = np.arange(len(cc_Co57_1w))
ax[2].plot(energies, cc_Co57_1w, color=colors[0])
ax[2].plot(energies,  cc_Co57_4w, color=colors[1])
ax[2].plot(energies, sec_Co57_1w, color=colors[2])
ax[2].plot(energies, sec_Co57_4w, color=colors[3])
ax[2].set_xlabel('Energy (AU)', fontsize=14)
ax[2].set_ylabel('Counts', fontsize=14)
ax[2].set_title(titles[2], fontsize=14)
ax[2].tick_params(labelsize=13)
ax[2].set_ylim([0, np.max(cc_Co57_1w)*1.1])
ax[2].set_xlim([0, len(cc_Co57_1w)])
ax[2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

energies = np.arange(len(cc_Am241_1w))
ax[3].plot(energies, cc_Am241_1w, color=colors[0])
ax[3].plot(energies,  cc_Am241_4w, color=colors[1])
ax[3].plot(energies, sec_Am241_1w, color=colors[2])
ax[3].plot(energies, sec_Am241_4w, color=colors[3])
ax[3].set_xlabel('Energy (AU)', fontsize=14)
ax[3].set_ylabel('Counts', fontsize=14)
ax[3].set_title(titles[3], fontsize=14)
ax[3].tick_params(labelsize=13)
ax[3].set_ylim([0, np.max(cc_Am241_1w)*1.1])
ax[3].set_xlim([0, len(cc_Am241_1w)])
ax[3].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

bluepatch = mlines.Line2D([0], [0], color=colors[2], label='SEC 1w')
redpatch = mlines.Line2D([0], [0], color=colors[3], label='SEC 4w')
greenpatch = mlines.Line2D([0], [0], color=colors[1], label='CC 4w')
blackpatch = mlines.Line2D([0], [0], color=colors[0], label='CC 1w')
leg1 = plt.legend(handles=[blackpatch, greenpatch, bluepatch, redpatch], bbox_to_anchor=(0.45, -0.15),
                  loc='lower center', fancybox=True, ncol=4, shadow=False, fontsize=13, title_fontsize=14,
                  handlelength=2, handletextpad=0.5, columnspacing=1)

# Add letters for subplots
ax = ax.flat
for n, ax in enumerate(ax):
    ax.text(-0.1, 1.1, string.ascii_lowercase[n], transform=ax.transAxes,
            size=20, weight='bold')

ax1.add_artist(leg1)
plt.subplots_adjust(top=0.95, bottom=0.15, left=0.15, right=0.90, hspace=0.85)
plt.show()
#plt.savefig('C:/Users/10376/Documents/IEEE Abstract/Figures/Spectra/mixed_spectra.png', dpi=fig.dpi)

#%%  Flat field
#sns.set_style('white')
folder = 'C:/Users/10376/Documents/IEEE Abstract/Raw Data/Flat Field/'

x = np.load(folder + 'plexiglass_4w.npy')

plt.imshow(x[0, 7, 100])
plt.show()


