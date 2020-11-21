import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines

directory = 'D:/Research/Python Data/Spectral CT/'
folders = ['Al_2.0_8-14-19', 'Al_2.0_10-17-19_3P', 'Al_2.0_10-17-19_1P',
           'Cu_0.5_8-14-19', 'Cu_0.5_9-13-19', 'Cu_0.5_10-17-19',
           'Cu_1.0_8-14-19', 'Cu_1.0_9-13-19', 'Cu_1.0_10-17-19',
           'Cu_0.5_Time_0.5_11-11-19', 'Cu_0.5_Time_0.1_11-4-19',
           'AuGd_width_5_12-2-19', 'AuGd_width_10_12-2-19', 'AuGd_width_14_12-2-19', 'AuGd_width_20_12-9-19']

good_slices = [[5, 19], [10, 18], [11, 18],
               [4, 15], [7, 15], [12, 19],
               [4, 14], [5, 16], [10, 19],
               [10, 19], [10, 18],
               [11, 19], [11, 19], [11, 19], [11, 19]]


#%% Figure 1 Signal (now plots 1 bin, but all contrast agents, each in a separate plot with all filters)

# Choose the bin
b = 4

concentration = np.array([5, 3, 1, 0, 0, 0])
# Calculate y points from the fit above
xpts = np.linspace(6, -0.5, 50)
sns.set()

fig, ax = plt.subplots(1, 5, sharey=True, figsize=(14, 6))
ax1 = fig.add_subplot(111, frameon=False)
ax1.grid(False)
# Hide axes ticks
ax1.set_xticks([])
ax1.set_yticks([])


colors = ['orange', 'crimson', 'mediumseagreen', 'darkorchid', 'dodgerblue']
titles = ['Gold', 'Dysprosium', 'Lutetium', 'Gadolinium', 'Iodine']
line_styles = ['--', '-', ':']

for x in np.arange(3):
    mean_signal = np.empty([3, 5, 6])
    std_signal = np.empty([3, 5, 6])
    for i, folder in enumerate(folders[3*x:3*x+3]):

        mean_signal[i, :, :] = np.load(directory + folder + '/Mean_Signal.npy')
        std_signal[i, :, :] = np.load(directory + folder + '/Std_Signal.npy')


    zeros_mean = mean_signal[:, b, 0]
    zeros_std = std_signal[:, b, 0]

    for v in np.arange(1, 6):
        curr_plot_mean = np.concatenate((mean_signal[:, b, v], zeros_mean))

        coeffs = np.polyfit(concentration, curr_plot_mean, 1)
        p = np.poly1d(coeffs)


        curr_plot_std = np.concatenate((std_signal[:, b, v], zeros_std))
        #ax[v-1].errorbar(concentration, curr_plot_mean, yerr=curr_plot_std, color='black', capsize=3,
        #                  fmt='none')
        ax[v-1].plot(xpts, p(xpts), color='black', ls=line_styles[x])
        ax[v-1].set_xlim([-0.5, 6])
        ax[v-1].tick_params(labelsize=13)

        ax[v-1].set_title(titles[v-1], fontsize=15)


linepatch = mlines.Line2D([0], [0], color='black', lw=2, linestyle='-', label='0.5 mm Cu')
dashpatch = mlines.Line2D([0], [0], color='black', lw=2, linestyle='--', label='2.0 mm Al')
dotpatch = mlines.Line2D([0], [0], color='black', lw=2, linestyle=':', label='1.0 mm Cu')
ax1.legend(handles=[dashpatch, linepatch, dotpatch], fancybox=True, shadow=False, bbox_to_anchor=(0.5, -0.3),
           loc='lower center', ncol=3, fontsize=15)

ax1.set_ylabel('Signal (HU)', fontsize=20, labelpad=50)
ax1.set_xlabel('Contrast Concentration (wt%)', fontsize=20, labelpad=30)
plt.subplots_adjust(top=0.90, bottom=0.20, hspace=0.3)
plt.show()
#plt.savefig(directory + 'Paper 1 Figures/Figure1.png', dpi=1000)

#%% Figure 1 K-Edge CNR

xpts = np.linspace(-0.5, 6, 50)
concentration = np.array([5, 3, 1, 0])
# Calculate y points from the fit above
sns.set()

fig, ax = plt.subplots(3, 4, sharey=True, figsize=(15, 9))
ax1 = fig.add_subplot(111, frameon=False)
ax1.grid(False)
# Hide axes ticks
ax1.set_xticks([])
ax1.set_yticks([])

# Au, Lu, Dy, Gd
colors = ['orange', 'mediumseagreen', 'crimson', 'darkorchid']
titles = ['Au', 'Lu', 'Dy', 'Gd']

for x in np.arange(3):
    mean_signal = np.empty([3, 8])
    std_signal = np.empty([3, 8])

    for i, folder in enumerate(folders[3*x:3*x+3]):

        mean_signal[i, :] = np.load(directory + folder + '/Mean_Kedge_CNR_Filter.npy')
        std_signal[i, :] = np.load(directory + folder + '/Std_Kedge_CNR_Filter.npy')

    for b in np.arange(4):
        zero_mean = mean_signal[0, b]
        zero_std = std_signal[0, b]

        curr_mean = np.concatenate((mean_signal[:, b + 4], [zero_mean]))

        coeffs = np.polyfit(concentration, curr_mean, 1)
        p = np.poly1d(coeffs)

        curr_std = np.concatenate((std_signal[:, b], [zero_std]))

        ax[x, b].errorbar(concentration, curr_mean, yerr=curr_std, color=colors[b], capsize=3, fmt='none')
        ax[x, b].plot(xpts, p(xpts), color=colors[b])
        #ax[x, b].set_ylim([-0.5, 6])
        ax[x, b].set_xlim([-0.5, 6])
        if x == 0:
            ax[x, b].set_title(titles[b], fontsize=20)
        ax[x, b].tick_params(labelsize=15)

ax[0, 0].set_ylabel('2.0 mm Al', fontsize=25, labelpad=50)
ax[1, 0].set_ylabel('0.5 mm Cu', fontsize=25, labelpad=50)
ax[2, 0].set_ylabel('1.0 mm Cu', fontsize=25, labelpad=50)

ax1.set_ylabel('CNR (HU)', fontsize=20, labelpad=30)
ax1.set_xlabel('Contrast Concentration (wt%)', fontsize=20, labelpad=30)
plt.subplots_adjust(top=0.90, bottom=0.15, hspace=0.35)
ax1.set_title('K-Edge CNR', fontsize=25, pad=35)
plt.show()
plt.savefig(directory + 'Paper 1 Figures/Figure1_K-Edge_CNR.png', dpi=1000)

#%% Figure 2
import matplotlib.patches as mpatches

concentration = np.array([0, 0.5, 3])
# Calculate y points from the fit above
xpts = np.linspace(-0.5, 3.5, 50)
sns.set()

fig, ax = plt.subplots(4, 4, sharey=True, figsize=(12, 12))
ax1 = fig.add_subplot(111, frameon=False)
ax1.grid(False)
# Hide axes ticks
ax1.set_xticks([])
ax1.set_yticks([])


colors = ['steelblue', 'darkorchid']
titles = [['45-50 keV', '50-55 keV', '76-81 keV', '81-86 keV'],
          ['40-50 keV', '50-60 keV', '71-81 keV', '81-91 keV'],
          ['45-50 keV', '50-64 keV', '67-81 keV', '81-95 keV'],
          ['42-50 keV', '50-58 keV', '61-81 keV', '81-101 keV']]

# This array goes in the order, 0%, 0.5% Au, 3% Au, 0.5% Gd, 3% Gd

for i, folder in enumerate(folders[11:]):

    mean_signal = np.load(directory + folder + '/Mean_Signal_BinWidth.npy')
    std_signal = np.load(directory + folder + '/Std_Signal_BinWidth.npy')

    for b in np.arange(5):
        if b == 2:
            continue

        zero_mean = mean_signal[b, 0]
        zero_std = std_signal[b, 0]

        if b < 2:
            curr_plot_mean = np.concatenate(([zero_mean], mean_signal[b, 3:]))
            curr_plot_std = np.concatenate(([zero_mean], std_signal[b, 3:]))
        else:
            curr_plot_mean = np.concatenate(([zero_mean], mean_signal[b, 1:3]))
            curr_plot_std = np.concatenate(([zero_mean], std_signal[b, 1:3]))

        coeffs = np.polyfit(concentration, curr_plot_mean, 1)
        p = np.poly1d(coeffs)

        if b < 2:
            ax[i, b].errorbar(concentration, curr_plot_mean, yerr=curr_plot_std, color=colors[1], capsize=3,
                              fmt='none')
            ax[i, b].plot(xpts, p(xpts), color=colors[1])
            ax[i, b].set_xlim([-0.5, 3.5])
            #ax[i, b].set_ylim([-1000, 10000])
            ax[i, b].set_title(titles[i][b], fontsize=15)

        else:
            ax[i, b-1].errorbar(concentration, curr_plot_mean, yerr=curr_plot_std, color=colors[0], capsize=3,
                              fmt='none')
            ax[i, b-1].plot(xpts, p(xpts), color=colors[0])
            ax[i, b-1].set_xlim([-0.5, 3.5])
            #ax[i, b-1].set_ylim([-1000, 10000])
            ax[i, b-1].set_title(titles[i][b-1], fontsize=15)

ax[0, 0].set_ylabel('5 keV \nWidth', fontsize=20, labelpad=50)
ax[1, 0].set_ylabel('10 keV \nWidth', fontsize=20, labelpad=50)
ax[2, 0].set_ylabel('14 keV \nWidth', fontsize=20, labelpad=50)
ax[3, 0].set_ylabel('8/20 keV \nWidth', fontsize=20, labelpad=50)

purplepatch = mpatches.Patch(color='darkorchid', label='Gd (Z=64)')
orangepatch = mpatches.Patch(color='steelblue', label='Au (Z=79)')
ax1.legend(handles=[purplepatch, orangepatch], loc='upper center',
           bbox_to_anchor=(0.5, -0.12), fancybox=True, shadow=True, ncol=5, fontsize=15)
ax1.set_ylabel('Signal (HU)', fontsize=20, labelpad=50)
ax1.set_xlabel('Contrast Concentration (wt%)', fontsize=20, labelpad=30)
plt.subplots_adjust(top=0.90, bottom=0.20, hspace=0.5, left=0.19)
plt.show()
#plt.savefig(directory + 'Paper 1 Figures/Figure2_Signal.png', dpi=1000)

#%% Figure 2 CNR

import matplotlib.patches as mpatches

concentration = np.array([0, 0.5, 3])
# Calculate y points from the fit above
xpts = np.linspace(-0.5, 3.5, 50)
sns.set()

fig, ax = plt.subplots(2, 4, sharey=True, figsize=(12, 7))
ax1 = fig.add_subplot(111, frameon=False)
ax1.grid(False)
# Hide axes ticks
ax1.set_xticks([])
ax1.set_yticks([])


colors = ['steelblue', 'darkorchid']
titles = [['5 keV Bin Width', '10 keV Bin Width', '14 keV Bin Width', '20 keV Bin Width'],
          ['5 keV Bin Width', '10 keV Bin Width', '14 keV Bin Width', '8 keV Bin Width']]

for i, folder in enumerate(folders[11:]):
    # These arrays go in the order: 0%, 0.5% Au, 3% Au, 0%, 0.5% Gd, 3% Gd
    mean_CNR = np.load(directory + folder + '/Mean_Signal_BinWidth_CNR.npy')
    std_CNR = np.load(directory + folder + '/Std_Signal_BinWidth_CNR.npy')

    for b in np.arange(2):

        curr_plot_mean = mean_CNR[b*3:b*3+3]
        curr_plot_std = std_CNR[b*3:b*3+3]

        coeffs = np.polyfit(concentration, curr_plot_mean, 1)
        p = np.poly1d(coeffs)

        if b == 1:
            if i == 1:
                j = 2
            elif i == 2:
                j = 3
            elif i == 3:
                j = 1
            else:
                j = 0
            ax[b, j].errorbar(concentration, curr_plot_mean, yerr=curr_plot_std, color=colors[b], capsize=3,
                              fmt='none')
            ax[b, j].plot(xpts, p(xpts), color=colors[b])
            ax[b, j].set_xlim([-0.5, 3.5])
            #ax[b, i].set_ylim([-1000, 10000])
            ax[b, j].tick_params(labelsize=15)
            ax[b, j].set_xticks([0, 1, 2, 3])
            ax[b, j].set_title(titles[b][i], fontsize=15)
        else:
            ax[b, i].errorbar(concentration, curr_plot_mean, yerr=curr_plot_std, color=colors[b], capsize=3,
                              fmt='none')
            ax[b, i].plot(xpts, p(xpts), color=colors[b])
            ax[b, i].set_xlim([-0.5, 3.5])
            # ax[b, i].set_ylim([-1000, 10000])
            ax[b, i].tick_params(labelsize=15)
            ax[b, i].set_xticks([0, 1, 2, 3])
            ax[b, i].set_title(titles[b][i], fontsize=15)

purplepatch = mpatches.Patch(color=colors[1], label='Gd (Z=64)')
orangepatch = mpatches.Patch(color=colors[0], label='Au (Z=79)')
ax1.legend(handles=[purplepatch, orangepatch], loc='upper center',
           bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=5, fontsize=15)
ax1.set_ylabel('K-Edge CNR', fontsize=20, labelpad=50)
ax1.set_xlabel('Contrast Concentration (wt%)', fontsize=20, labelpad=35)
ax1.set_title('K-Edge CNR', fontsize=25, pad=30)
plt.subplots_adjust(top=0.88, bottom=0.20, hspace=0.5)
plt.show()
#plt.savefig(directory + 'Paper 1 Figures/Figure2_CNR.png', dpi=1000)

#%% Figure 3 CNR

xpts = np.array([1, 2, 3, 4, 5])
# Calculate y points from the fit above
sns.set()

fig, ax = plt.subplots(3, 5, sharey=True, figsize=(15, 9))
ax1 = fig.add_subplot(111, frameon=False)
ax1.grid(False)
# Hide axes ticks
ax1.set_xticks([])
ax1.set_yticks([])

colors = ['orange', 'crimson', 'mediumseagreen', 'darkorchid', 'dodgerblue']
titles = ['16 or 35-50 keV', '50-54 keV', '54-64 keV', '64-81 keV', '81-120 keV']

for i, folder in enumerate(np.concatenate(([folders[4]], folders[9:11]))):

    mean_CNR = np.load(directory + folder + '/Mean_CNR_Time.npy')
    std_CNR = np.load(directory + folder + '/Std_CNR_Time.npy')

    for b in np.arange(5):

        ax[i, b].bar(xpts, mean_CNR[b, 1:], color=colors, yerr=std_CNR[b, 1:])
        if i == 0:
            ax[i, b].set_title(titles[b], fontsize=15)
        ax[i, b].set_xticks(xpts)
        ax[i, b].set_xticklabels(['Au', 'Dy', 'Lu', 'Gd', 'I'])
        ax[i, b].tick_params(labelsize=15)

ax[0, 0].set_ylabel('1 s', fontsize=25, labelpad=50)
ax[1, 0].set_ylabel('0.5 s', fontsize=25, labelpad=50)
ax[2, 0].set_ylabel('0.1 s', fontsize=25, labelpad=50)

ax1.set_ylabel('CNR (HU)', fontsize=20, labelpad=30)
ax1.set_xlabel('Contrast Element', fontsize=20, labelpad=30)
ax1.set_title('CT CNR', fontsize=25, pad=35)
plt.subplots_adjust(top=0.90, bottom=0.15, hspace=0.35)
plt.show()
#plt.savefig(directory + 'Paper 1 Figures/Figure3_CNR.png', dpi=1000)

#%% Figure 3 Noise

import matplotlib.patches as mpatches

xpts = np.array([1, 2, 3, 4, 5])
# Calculate y points from the fit above
sns.set()

fig, ax = plt.subplots(1, 3, sharey=True, figsize=(15, 7))
ax1 = fig.add_subplot(111, frameon=False)
ax1.grid(False)
# Hide axes ticks
ax1.set_xticks([])
ax1.set_yticks([])

titles = ['1 s', '0.5 s', '0.1 s']
bins = ['35-50', '\n50-54', '54-64', '\n64-81', '81-120']

for i, folder in enumerate(np.concatenate(([folders[4]], folders[9:11]))):

    mean_noise = np.load(directory + folder + '/Mean_Noise_Time.npy')
    std_noise = np.load(directory + folder + '/Std_Noise_Time.npy')

    ax[i].bar(xpts, mean_noise, yerr=std_noise)
    ax[i].set_xticks(xpts)
    ax[i].set_xticklabels(bins)
    ax[i].tick_params(labelsize=15)
    ax[i].set_title(titles[i], fontsize=20)

ax[0].set_xticklabels(['16-50', '\n50-54', '54-64', '\n64-81', '81-120'], fontsize=15)
ax1.set_ylabel('Noise (HU)', fontsize=20, labelpad=50)
ax1.set_xlabel('Energy Range (keV)', fontsize=20, labelpad=50)
plt.subplots_adjust(top=0.85, bottom=0.20)
plt.show()
#plt.savefig(directory + 'Paper 1 Figures/Figure3_Noise.png', dpi=1000)

#%% Figure 3 K-Edge CNR

xpts = np.array([1, 2, 3, 4])
# Calculate y points from the fit above
sns.set()

fig, ax = plt.subplots(1, 3, sharey=True, figsize=(15, 9))
ax1 = fig.add_subplot(111, frameon=False)
ax1.grid(False)
# Hide axes ticks
ax1.set_xticks([])
ax1.set_yticks([])

# Au, Lu, Dy, Gd
colors = ['orange', 'mediumseagreen', 'crimson', 'darkorchid']
titles = ['1 s', '0.5s', '0.1 s']

for i, folder in enumerate(np.concatenate(([folders[4]], folders[9:11]))):

    mean_CNR = np.load(directory + folder + '/Mean_Kedge_CNR_Time.npy')
    std_CNR = np.load(directory + folder + '/Std_Kedge_CNR_Time.npy')

    ax[i].bar(xpts, mean_CNR, color=colors, yerr=std_CNR)
    ax[i].set_xticks(xpts)
    ax[i].set_xticklabels(['Au', 'Lu', 'Dy', 'Gd'])
    ax[i].set_title(titles[i], fontsize=20)
    ax[i].tick_params(labelsize=15)

ax1.set_ylabel('CNR (HU)', fontsize=20, labelpad=30)
ax1.set_xlabel('Contrast Element', fontsize=20, labelpad=30)
plt.subplots_adjust(top=0.85, bottom=0.20)
ax1.set_title('K-Edge CNR', fontsize=25, pad=35)
plt.show()
#plt.savefig(directory + 'Paper 1 Figures/Figure3_K-Edge_CNR.png', dpi=1000)

#%% Figure 4

concentration = np.array([5, 3, 1, 0])
# Calculate y points from the fit above
xpts = np.linspace(6, -0.5, 50)
sns.set()


fig, ax = plt.subplots(1, 4, sharey=True, figsize=(15, 5))
ax1 = fig.add_subplot(111, frameon=False)
ax1.grid(False)
# Hide axes ticks
ax1.set_xticks([])
ax1.set_yticks([])

colors = ['steelblue', 'mediumseagreen', 'crimson']
titles = ['Gold (Z=79)', 'Lutetium (Z=71)', 'Dysprosium (Z=66)', 'Gadolinium (Z=64)']


for x in np.arange(3):
    mean_signal = np.empty([3, 8])
    std_signal = np.empty([3, 8])
    for i, folder in enumerate(folders[3*x:3*x+3]):

        mean_signal[i, :] = np.load(directory + folder + '/Mean_Kedge.npy')
        std_signal[i, :] = np.load(directory + folder + '/Std_Kedge.npy')

    for b in np.arange(4):
        zero_mean = mean_signal[0, b]
        zero_std = std_signal[0, b]

        curr_mean = np.concatenate((mean_signal[:, b+4], [zero_mean]))

        coeffs = np.polyfit([zero_mean, curr_mean[0]], [0, 5], 1)
        p = np.poly1d(coeffs)

        k_edge_concentration = p(curr_mean)

        curr_std = np.concatenate((std_signal[:, b], [zero_std]))
        curr_std = np.multiply(curr_std, coeffs[0])

        ax[b].errorbar(concentration, k_edge_concentration, yerr=curr_std, color=colors[x], capsize=3, fmt='none')
        ax[b].set_ylim([-0.5, 6])
        ax[b].set_xlim([-0.5, 6])
        ax[b].set_title(titles[b], fontsize=20)
        ax[b].tick_params(labelsize=15)

ax[0].plot(xpts, xpts, color='black', lw=1)
ax[1].plot(xpts, xpts, color='black', lw=1)
ax[2].plot(xpts, xpts, color='black', lw=1)
ax[3].plot(xpts, xpts, color='black', lw=1)

bluepatch = mpatches.Patch(color='steelblue', label='2.0 mm Al')
greenpatch = mpatches.Patch(color='mediumseagreen', label='0.5 mm Cu')
redpatch = mpatches.Patch(color='crimson', label='1.0 mm Cu')
ax1.legend(handles=[bluepatch, greenpatch, redpatch], loc='upper center',
           bbox_to_anchor=(0.5, -0.22), fancybox=True, shadow=True, ncol=5, fontsize=15)
ax1.set_ylabel('K-Edge Concentration (wt%)', fontsize=20, labelpad=50)
ax1.set_xlabel('Contrast Concentration (wt%)', fontsize=20, labelpad=30)
ax1.set_title('K-Edge Concentration vs. Concentration', fontsize=25, pad=35)
plt.subplots_adjust(top=0.85, bottom=0.25)

plt.show()
#plt.savefig(directory + 'Paper 1 Figures/Figure4.png', dpi=1000)