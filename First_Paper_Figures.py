import numpy as np
import matplotlib.pyplot as plt
import sCT_Analysis as sct
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

# Create the colormaps
nbins = 100
c1 = (1, 0, 1)
c2 = (0, 1, 0)
c3 = (1, 0.843, 0)
c4 = (1, 0, 0)

gray_val = 0
gray_list = (gray_val, gray_val, gray_val)

c1_rng = [gray_list, c1]
cmap1 = colors.LinearSegmentedColormap.from_list('Purp', c1_rng, N=nbins)
c2_rng = [gray_list, c2]
cmap2 = colors.LinearSegmentedColormap.from_list('Gree', c2_rng, N=nbins)
c3_rng = [gray_list, c3]
cmap3 = colors.LinearSegmentedColormap.from_list('G78', c3_rng, N=nbins)
c4_rng = [gray_list, c4]
cmap4 = colors.LinearSegmentedColormap.from_list('Redd8', c4_rng, N=nbins)

#%% Figure 1

concentration = np.array([5, 3, 1, 0, 0, 0])
# Calculate y points from the fit above
xpts = np.linspace(6, -0.5, 50)
sns.set_style('whitegrid')

fig, ax = plt.subplots(1, 4, figsize=(13, 6), sharey=True)
ax1 = fig.add_subplot(111, frameon=False)
ax1.grid(False)
# Hide axes ticks
ax1.set_xticks([])
ax1.set_yticks([])

# Colors for the different contrast materials
colors = ['orange', 'crimson', 'mediumseagreen', 'darkorchid', 'dodgerblue']

# Filter colors
titles = ['16-50 keV', '50-54 keV', '54-64 keV', '64-81 keV', '81-120 keV']

mean_2Al = np.empty([3, 5, 6])
std_2Al = np.empty([3, 5, 6])

mean_05Cu = np.empty([3, 5, 6])
std_05Cu = np.empty([3, 5, 6])

mean_1Cu = np.empty([3, 5, 6])
std_1Cu = np.empty([3, 5, 6])

# Get Al data
for i, folder in enumerate(folders[0:3]):
    mean_2Al[i, :, :] = np.load(directory + folder + '/Mean_Signal.npy')
    std_2Al[i, :, :] = np.load(directory + folder + '/Std_Signal.npy')

# Get 0.5 Cu data
for i, folder in enumerate(folders[3:6]):
    mean_05Cu[i, :, :] = np.load(directory + folder + '/Mean_Signal.npy')
    std_05Cu[i, :, :] = np.load(directory + folder + '/Std_Signal.npy')

# Get 1.0 Cu data
for i, folder in enumerate(folders[6:9]):
    mean_1Cu[i, :, :] = np.load(directory + folder + '/Mean_Signal.npy')
    std_1Cu[i, :, :] = np.load(directory + folder + '/Std_Signal.npy')

# First plot 0.5 Cu 16-50 keV, Signal vs. Concentration
zeros_mean = mean_05Cu[:, 0, 0]
zeros_std = std_05Cu[:, 0, 0]
for v in np.arange(1, 5):
    curr_plot_mean = np.concatenate((mean_05Cu[:, 0, v], zeros_mean))

    coeffs = np.polyfit(concentration, curr_plot_mean, 1)
    p = np.poly1d(coeffs)

    curr_plot_std = np.concatenate((std_05Cu[:, 0, v], zeros_std))
    ax[0].errorbar(concentration, curr_plot_mean, yerr=curr_plot_std, color=colors[v - 1], capsize=3, fmt='none')
    ax[0].plot(xpts, p(xpts), color=colors[v - 1])
    ax[0].set_xlim([-0.2, 6])
    ax[0].set_ylim([-60, 2650])
    ax[0].tick_params(labelsize=13)
    ax[0].set_title('0.5 mm Cu \n 16-50 keV', fontsize=15)

# Second plot 0.5 Cu 81-120 keV, Signal vs. Concentration
for v in np.arange(1, 5):
    curr_plot_mean = np.concatenate((mean_05Cu[:, 4, v], zeros_mean))

    coeffs = np.polyfit(concentration, curr_plot_mean, 1)
    p = np.poly1d(coeffs)

    curr_plot_std = np.concatenate((std_05Cu[:, 4, v], zeros_std))
    ax[1].errorbar(concentration, curr_plot_mean, yerr=curr_plot_std, color=colors[v - 1], capsize=3, fmt='none')
    ax[1].plot(xpts, p(xpts), color=colors[v - 1])
    ax[1].set_xlim([-0.2, 6])
    ax[1].tick_params(labelsize=13)
    ax[1].set_title('0.5 mm Cu \n 81-120 keV', fontsize=15)

# Third plot: all filters, Gd only, 16-50 keV
zeros_mean = mean_2Al[:, 0, 0]
zeros_std = std_2Al[:, 0, 0]

curr_plot_mean = np.concatenate((mean_2Al[:, 0, 2], zeros_mean))

coeffs = np.polyfit(concentration, curr_plot_mean, 1)
p = np.poly1d(coeffs)

curr_plot_std = np.concatenate((std_2Al[:, 0, 2], zeros_std))
ax[2].errorbar(concentration, curr_plot_mean, yerr=curr_plot_std, color=colors[1], capsize=3, fmt='none')
ax[2].plot(xpts, p(xpts), color=colors[1], ls='--')
ax[2].set_xlim([-0.2, 6])
ax[2].tick_params(labelsize=13)
ax[2].set_title('Dysprosium, 16-50 keV \n Filter dependence', fontsize=15)

zeros_mean = mean_05Cu[:, 0, 0]
zeros_std = std_05Cu[:, 0, 0]

curr_plot_mean = np.concatenate((mean_05Cu[:, 0, 2], zeros_mean))

coeffs = np.polyfit(concentration, curr_plot_mean, 1)
p = np.poly1d(coeffs)

curr_plot_std = np.concatenate((std_05Cu[:, 0, 2], zeros_std))
ax[2].errorbar(concentration, curr_plot_mean, yerr=curr_plot_std, color=colors[1], capsize=3, fmt='none')
ax[2].plot(xpts, p(xpts), color=colors[1], ls='-')

zeros_mean = mean_1Cu[:, 0, 0]
zeros_std = std_1Cu[:, 0, 0]

curr_plot_mean = np.concatenate((mean_1Cu[:, 0, 2], zeros_mean))

coeffs = np.polyfit(concentration, curr_plot_mean, 1)
p = np.poly1d(coeffs)

curr_plot_std = np.concatenate((std_1Cu[:, 0, 2], zeros_std))
ax[2].errorbar(concentration, curr_plot_mean, yerr=curr_plot_std, color=colors[1], capsize=3, fmt='none')
ax[2].plot(xpts, p(xpts), color=colors[1], ls=':')

# Fourth plot: all filters, Au only, 81-120 keV
zeros_mean = mean_2Al[:, 0, 0]
zeros_std = std_2Al[:, 0, 0]

curr_plot_mean = np.concatenate((mean_2Al[:, 0, 1], zeros_mean))

coeffs = np.polyfit(concentration, curr_plot_mean, 1)
p = np.poly1d(coeffs)

curr_plot_std = np.concatenate((std_2Al[:, 0, 1], zeros_std))
ax[3].errorbar(concentration, curr_plot_mean, yerr=curr_plot_std, color=colors[0], capsize=3, fmt='none')
ax[3].plot(xpts, p(xpts), color=colors[0], ls='--')
ax[3].set_xlim([-0.2, 6])
ax[3].tick_params(labelsize=13)
ax[3].set_title('Gold, 16-50 keV \n Filter dependence', fontsize=15)

zeros_mean = mean_05Cu[:, 0, 0]
zeros_std = std_05Cu[:, 0, 0]

curr_plot_mean = np.concatenate((mean_05Cu[:, 0, 1], zeros_mean))

coeffs = np.polyfit(concentration, curr_plot_mean, 1)
p = np.poly1d(coeffs)

curr_plot_std = np.concatenate((std_05Cu[:, 0, 1], zeros_std))
ax[3].errorbar(concentration, curr_plot_mean, yerr=curr_plot_std, color=colors[0], capsize=3, fmt='none')
ax[3].plot(xpts, p(xpts), color=colors[0], ls='-')

zeros_mean = mean_1Cu[:, 0, 0]
zeros_std = std_1Cu[:, 0, 0]

curr_plot_mean = np.concatenate((mean_1Cu[:, 0, 1], zeros_mean))

coeffs = np.polyfit(concentration, curr_plot_mean, 1)
p = np.poly1d(coeffs)

curr_plot_std = np.concatenate((std_1Cu[:, 0, 1], zeros_std))
ax[3].errorbar(concentration, curr_plot_mean, yerr=curr_plot_std, color=colors[0], capsize=3, fmt='none')
ax[3].plot(xpts, p(xpts), color=colors[0], ls=':')

bluepatch = mpatches.Patch(color='dodgerblue', label='I (Z=53)')
purplepatch = mpatches.Patch(color='darkorchid', label='Gd (Z=64)')
redpatch = mpatches.Patch(color='crimson', label='Dy (Z=66)')
greenpatch = mpatches.Patch(color='mediumseagreen', label='Lu (Z=71)')
orangepatch = mpatches.Patch(color='orange', label='Au (Z=79)')
linepatch = mlines.Line2D([0], [0], color='black', lw=2, linestyle='-', label='0.5 mm Cu')
dashpatch = mlines.Line2D([0], [0], color='black', lw=2, linestyle='--', label='2.0 mm Al')
dotpatch = mlines.Line2D([0], [0], color='black', lw=2, linestyle=':', label='1.0 mm Cu')
leg1 = plt.legend(handles=[purplepatch, redpatch, greenpatch, orangepatch], bbox_to_anchor=(0.5, -0.47),
                  loc='lower center', fancybox=True, shadow=False, ncol=4, fontsize=15)
leg2 = plt.legend(handles=[dashpatch, linepatch, dotpatch], bbox_to_anchor=(0.5, -0.63), fancybox=True, shadow=False,
                  loc='lower center', ncol=3, fontsize=15)
ax1.add_artist(leg1)
ax1.add_artist(leg2)
ax1.set_ylabel('Signal (HU)', fontsize=18, labelpad=50)
ax1.set_xlabel('Contrast Concentration (wt%)', fontsize=18, labelpad=30)
plt.subplots_adjust(top=0.80, bottom=0.35, hspace=0.2, wspace=0.3, left=0.09, right=0.93)
plt.show()
#plt.savefig(directory + 'Paper 1 Figures/Figure1.png', dpi=500)

#%% Figure 2

# Calculate y points from the fit above
xpts = np.linspace(0, 25, 50)
sns.set_style('whitegrid')

fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
ax1 = fig.add_subplot(111, frameon=False)
ax1.grid(False)
# Hide axes ticks
ax1.set_xticks([])
ax1.set_yticks([])

# Au, Gd
colors = ['orange', 'darkorchid']
Au_widths = np.array([5, 10, 14, 20])
Gd_widths = np.array([5, 8, 10, 14])

titles = [['5 keV Bin Width', '10 keV Bin Width', '14 keV Bin Width', '20 keV Bin Width'],
          ['5 keV Bin Width', '10 keV Bin Width', '14 keV Bin Width', '8 keV Bin Width']]

Au_CNR = np.zeros(4)
Gd_CNR_temp = np.zeros(4)
Au_std = np.zeros(4)
Gd_std_temp = np.zeros(4)

for i, folder in enumerate(folders[11:]):
    # These arrays go in the order: 0%, 0.5% Au, 3% Au, 0%, 0.5% Gd, 3% Gd
    mean_CNR = np.load(directory + folder + '/Mean_Signal_BinWidth_CNR.npy')
    std_CNR = np.load(directory + folder + '/Std_Signal_BinWidth_CNR.npy')

    # Get the 3% values
    Au_CNR[i] = mean_CNR[2]
    Au_std[i] = std_CNR[2]

    Gd_CNR_temp[i] = mean_CNR[5]
    Gd_std_temp[i] = std_CNR[5]

# Move the 8 bin width and remove it from the end
Gd_CNR = np.insert(Gd_CNR_temp, 1, Gd_CNR_temp[3])
Gd_std = np.insert(Gd_std_temp, 1, Gd_std_temp[3])

Gd_CNR = np.delete(Gd_CNR, 4)
Gd_std = np.delete(Gd_std, 4)

#Au_CNR = np.insert(Au_CNR, 0, 0)
#Au_std = np.insert(Au_std, 0, 0)
#Gd_CNR = np.insert(Gd_CNR, 0, 0)
#Gd_std = np.insert(Gd_std, 0, 0)

# Create a quadratic fit of the data
Au_coeffs = np.polyfit(Au_widths, Au_CNR, 2)
Gd_coeffs = np.polyfit(Gd_widths, Gd_CNR, 2)

# Fit (equation)
p_Au = np.poly1d(Au_coeffs)
p_Gd = np.poly1d(Gd_coeffs)

# Plot the points
ax[1].plot(xpts, p_Au(xpts), color=colors[0])
ax[0].plot(xpts, p_Gd(xpts), color=colors[1])

ax[1].errorbar(Au_widths, Au_CNR, yerr=Au_std, fmt='none', capsize=4, color=colors[0])
ax[0].errorbar(Gd_widths, Gd_CNR, yerr=Gd_std, fmt='none', capsize=4, color=colors[1])

ax[0].set_xlim([4, 15])
ax[1].set_xlim([4, 21])
ax[0].set_ylim([0, 30])

ax[0].set_title('a) Gadolinum', fontsize=20)
ax[1].set_title('b) Gold', fontsize=20)

ax[0].tick_params(labelsize=15)
ax[1].tick_params(labelsize=15)



#orangepatch = mpatches.Patch(color=colors[0], label='Au (Z=79)')
#purplepatch = mpatches.Patch(color=colors[1], label='Gd (Z=64)')
#leg = plt.legend(handles=[purplepatch, orangepatch], loc='lower center', bbox_to_anchor=(0.5, -0.28), ncol=2,
#                 fancybox=True, fontsize=20)
#ax1.add_artist(leg)
ax1.set_ylabel('K-Edge CNR', fontsize=18, labelpad=40)
ax1.set_xlabel('Bin Width (keV)', fontsize=18, labelpad=35)
plt.subplots_adjust(bottom=0.23, wspace=0.3)
plt.show()
#plt.savefig(directory + 'Paper 1 Figures/Figure2.png', dpi=500)

#%% Figure 3

xpts = np.array([1, 2, 3, 4])
# Calculate y points from the fit above
sns.set_style('whitegrid')

fig, ax = plt.subplots(1, 3, sharey=True, figsize=(14, 6))
ax1 = fig.add_subplot(111, frameon=False)
ax1.grid(False)
# Hide axes ticks
ax1.set_xticks([])
ax1.set_yticks([])

# Order is Au, Lu, Dy, Gd
colors = ['orange', 'mediumseagreen', 'crimson', 'darkorchid']
titles = ['a) 1 s', 'b) 0.5 s', 'c) 0.1 s']

for i, folder in enumerate(np.concatenate(([folders[4]], folders[9:11]))):

    mean_CNR = np.load(directory + folder + '/Mean_Kedge_CNR_Time.npy')
    std_CNR = np.load(directory + folder + '/Std_Kedge_CNR_Time.npy')

    ax[i].bar(xpts, mean_CNR, color=colors, yerr=std_CNR, capsize=4)
    ax[i].set_xticks(xpts)
    ax[i].set_xticklabels(['Au', 'Lu', 'Dy', 'Gd'])
    ax[i].set_title(titles[i], fontsize=20)
    ax[i].tick_params(labelsize=15)

ax1.set_ylabel('K-Edge CNR', fontsize=18, labelpad=40)
ax1.set_xlabel('Contrast Element (3% Concentration)', fontsize=18, labelpad=40)
plt.subplots_adjust(top=0.85, bottom=0.20, wspace=0.3)
plt.show()
#plt.savefig(directory + 'Paper 1 Figures/Figure3.png', dpi=500)

#%% Figure 4

concentration = np.array([5, 3, 1, 0])
# Calculate y points from the fit above
xpts = np.linspace(6, -0.5, 50)
sns.set_style('whitegrid')

# set width of bar
barWidth = 0.3

fig = plt.figure(figsize=(8, 6))

# Order is Au, Lu, Dy, Gd
colors = ['orange', 'mediumseagreen', 'crimson', 'darkorchid']
labels = ['Gold (Z=79)', 'Lutetium (Z=71)', 'Dysprosium (Z=66)', 'Gadolinium (Z=64)']

mean_signal = np.empty([3, 8])
std_signal = np.empty([3, 8])
# Only want Cu_0.5
for i, folder in enumerate(folders[3:6]):

    mean_signal[i, :] = np.load(directory + folder + '/Mean_Kedge.npy')
    std_signal[i, :] = np.load(directory + folder + '/Std_Kedge.npy')


bars = np.zeros([4, 4])
stddev = np.zeros([4, 4])

# Get the values in order 0, 1, 3, 5 for each element
for b in np.arange(4):
    zero_mean = mean_signal[0, b]
    zero_std = std_signal[0, b]

    curr_mean = np.concatenate((mean_signal[:, b+4], [zero_mean]))

    coeffs = np.polyfit([zero_mean, curr_mean[0]], [0, 5], 1)
    p = np.poly1d(coeffs)

    k_edge_concentration = p(curr_mean)
    # Flip the order
    k_edge_concentration = np.flip(k_edge_concentration)

    curr_std = np.concatenate((std_signal[:, b], [zero_std]))
    curr_std = np.multiply(curr_std, coeffs[0])
    # Flip the order
    curr_std = np.flip(curr_std)

    bars[b, :] = k_edge_concentration
    stddev[b, :] = curr_std

# Positions for the bars on the x-axis
concs = [0, 1.5, 3, 4.5]
r2 = np.subtract(concs, barWidth/2)
r1 = [x - barWidth for x in r2]
r3 = np.add(concs, barWidth/2)
r4 = [x + barWidth for x in r3]

# Make the plot
bar_sub = 0.03
plt.bar(r1, bars[0], yerr=stddev[0], color=colors[0], width=barWidth-bar_sub, edgecolor=colors[0], label=labels[0], capsize=3)
plt.bar(r2, bars[1], yerr=stddev[1], color=colors[1], width=barWidth-bar_sub, edgecolor=colors[1], label=labels[1], capsize=3)
plt.bar(r3, bars[2], yerr=stddev[2], color=colors[2], width=barWidth-bar_sub, edgecolor=colors[2], label=labels[2], capsize=3)
plt.bar(r4, bars[3], yerr=stddev[3], color=colors[3], width=barWidth-bar_sub, edgecolor=colors[3], label=labels[3], capsize=3)

plt.xticks(concs, ['0%', '1%', '3%', '5%'])
plt.legend(loc='upper left', fancybox=True, shadow=True, fontsize=15)
plt.ylabel('K-Edge Concentration (%)', fontsize=20, labelpad=25)
plt.xlabel('True Concentration', fontsize=20, labelpad=15)
plt.tick_params(labelsize=15)
plt.subplots_adjust(top=0.95, bottom=0.135, left=0.125, right=0.945)
plt.show()
#plt.savefig(directory + 'Paper 1 Figures/Figure4.png', dpi=500)

#%% Setup Figure

lan = mpimg.imread('C:/Users/drich/OneDrive/Pictures/Multi-contrast imaging with PCCT/Lanthanide_Paper1.png')
augd = mpimg.imread('C:/Users/drich/OneDrive/Pictures/Multi-contrast imaging with PCCT/AuGd_Phantom.png')
setup = mpimg.imread('C:/Users/drich/OneDrive/Pictures/Multi-contrast imaging with PCCT/Setup_Labeled.png')
dimen = mpimg.imread('C:/Users/drich/OneDrive/Pictures/Multi-contrast imaging with PCCT/Dimensions2.png')

# Plot figure with subplots of different sizes
fig = plt.figure(figsize=(9, 9))
# set up subplot grid
gridspec.GridSpec(3, 3)

# large subplot
plt.subplot2grid((3, 3), (0, 0), colspan=3, rowspan=2, frameon=False)
plt.imshow(setup)
plt.grid(False)
plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, labelbottom=False, labelleft=False)
plt.title('a) Experimental Setup', fontsize=15)

# small subplot 1
plt.subplot2grid((3, 3), (2, 0), frameon=False)
plt.imshow(lan)
plt.grid(False)
plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, labelbottom=False, labelleft=False)
plt.annotate('Au', (368, 780), xycoords='data', fontsize=15)
plt.annotate('Dy', (368, 1410), xycoords='data', fontsize=15)
plt.annotate('Lu', (915, 1710), xycoords='data', fontsize=15)
plt.annotate('Gd', (1430, 1400), xycoords='data', fontsize=15)
plt.annotate('I', (1525, 785), xycoords='data', fontsize=15)
plt.annotate(r'$H_2 O$', (840, 470), xycoords='data', fontsize=13)
plt.annotate(r'$H_2 O$', (840, 1090), xycoords='data', fontsize=13)
plt.title('b) Lanthanide Phantom', fontsize=15)

# small subplot 2
plt.subplot2grid((3, 3), (2, 1), frameon=False)
plt.imshow(augd)
plt.grid(False)
plt.title('c) AuGd Phantom', fontsize=15)
plt.annotate('3%', (350, 785), xycoords='data', fontsize=15)
plt.annotate('0.5%', (315, 1400), xycoords='data', fontsize=12)
plt.annotate('Mix', (875, 1710), xycoords='data', fontsize=15)
plt.annotate('3%', (1425, 1400), xycoords='data', fontsize=15)
plt.annotate('0.5%', (1375, 775), xycoords='data', fontsize=12)
plt.annotate(r'$H_2 O$', (840, 470), xycoords='data', fontsize=13)
plt.annotate(r'$H_2 O$', (840, 1090), xycoords='data', fontsize=13)
plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, labelbottom=False, labelleft=False)

# small subplot 2
plt.subplot2grid((3, 3), (2, 2), frameon=False)
plt.imshow(dimen)
plt.grid(False)
plt.title('d) Phantom Dimensions', fontsize=15)
plt.annotate('0.6 cm', (1850, 160), xycoords='data', fontsize=12)
plt.annotate('2.5\ncm', (0, 1100), xycoords='data', fontsize=12)
plt.annotate('3 cm', (1100, 1940), xycoords='data', fontsize=12)
plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, labelbottom=False, labelleft=False)

plt.show()
#plt.savefig(directory + 'Paper 1 Figures/Setup_Figure.png', dpi=500)

#%% Figure 1 CT Images
colors = ['orange', 'mediumseagreen', 'crimson', 'darkorchid', 'dodgerblue']
sns.set_style('ticks')
fig, ax = plt.subplots(1, 4, figsize=(13, 6))
ax1 = fig.add_subplot(111, frameon=False)
ax1.grid(False)
# Hide axes ticks
ax1.set_xticks([])
ax1.set_yticks([])

# Window and Level
min_HU, max_HU = -800, 2000

# 16-50 keV, Cu 0.5
img0 = np.load(directory + folders[4] + '/Slices/Bin0_Slice9.npy')
ax[0].imshow(img0, cmap='gray', vmin=min_HU, vmax=max_HU)
ax[0].grid(False)
ax[0].set_xticks([4, 57.5, 111])
ax[0].set_xticklabels([-1.5, 0, 1.5])
ax[0].set_yticks([6, 59.5, 113])
ax[0].set_yticklabels([-1.5, 0, 1.5])
ax[0].tick_params(labelsize=15)
ax[0].set_title('0.5 mm Cu\n16-50 keV', fontsize=18)
ax[0].annotate('Au', (118, 232), xycoords='figure points', fontsize=15, color=colors[0])
ax[0].annotate('Dy', (117, 186), xycoords='figure points', fontsize=15, color=colors[2])
ax[0].annotate('Lu', (156, 163), xycoords='figure points', fontsize=15, color=colors[1])
ax[0].annotate('Gd', (193, 184), xycoords='figure points', fontsize=15, color=colors[3])
ax[0].annotate('I', (203, 230), xycoords='figure points', fontsize=15, color=colors[4])
ax[0].annotate(r'$H_2 O$', (155, 208), xycoords='figure points', fontsize=11, color='white')
ax[0].annotate(r'$H_2 O$', (155, 253), xycoords='figure points', fontsize=11, color='white')

img1 = np.load(directory + folders[4] + '/Slices/Bin4_Slice9.npy')
ax[1].imshow(img1, cmap='gray', vmin=min_HU, vmax=max_HU)
ax[1].grid(False)
ax[1].set_xticks([4, 57.5, 111])
ax[1].set_xticklabels([-1.5, 0, 1.5])
ax[1].set_yticks([6, 59.5, 113])
ax[1].set_yticklabels([-1.5, 0, 1.5])
ax[1].tick_params(labelsize=15)
ax[1].set_title('0.5 mm Cu\n81-120 keV', fontsize=18)

img2 = np.load(directory + folders[1] + '/Slices/Bin0_Slice9.npy')
ax[2].imshow(img2, cmap='gray', vmin=min_HU, vmax=max_HU)
ax[2].grid(False)
ax[2].set_xticks([4, 57.5, 111])
ax[2].set_xticklabels([-1.5, 0, 1.5])
ax[2].set_yticks([6, 59.5, 113])
ax[2].set_yticklabels([-1.5, 0, 1.5])
ax[2].tick_params(labelsize=15)
ax[2].set_title('2.0 mm Al\n16-50 keV', fontsize=18)

img3 = np.load(directory + folders[7] + '/Slices/Bin0_Slice9.npy')
ax[3].imshow(img3, cmap='gray', vmin=min_HU, vmax=max_HU)
ax[3].grid(False)
ax[3].set_xticks([4, 57.5, 111])
ax[3].set_xticklabels([-1.5, 0, 1.5])
ax[3].set_yticks([6, 59.5, 113])
ax[3].set_yticklabels([-1.5, 0, 1.5])
ax[3].tick_params(labelsize=15)
ax[3].set_title('1.0 mm Cu \n 16-50 keV', fontsize=18)

ax1.set_xlabel('x (cm)', fontsize=18, labelpad=-50)
ax1.set_ylabel('y (cm)', fontsize=18, labelpad=30)
plt.subplots_adjust(wspace=0.30, left=0.09, right=0.93)
plt.show()
#plt.savefig(directory + 'Paper 1 Figures/Figure1_Images.png', dpi=500)

#%% Mass Attenuation Coefficient Figures

folder = 'D:/Research/Bin Optimization/'
colors = ['orange', 'crimson', 'mediumseagreen', 'darkorchid', 'dodgerblue']  # Au, Dy, Lu, Gd, I

energies = np.load(folder + 'corrected-spectrum_120kV.npy')
au = np.load(folder + 'Au.npy')
dy = np.load(folder + 'Dy.npy')
lu = np.load(folder + 'Lu.npy')
gd = np.load(folder + 'Gd.npy')
iod = np.load(folder + 'I.npy')
h2o = np.load(folder + 'H2O.npy')

# Get only the energy values
energies = energies[:, 0]
energies = 1000*energies  # Convert from MeV to keV

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax1 = fig.add_subplot(111, frameon=False)
ax1.grid(False)
# Hide axes ticks
ax1.set_xticks([])
ax1.set_yticks([])

# Plot the elements

ax[0].semilogy(energies, au, color=colors[0])
ax[0].semilogy(energies, dy, color=colors[1])
ax[0].semilogy(energies, lu, color=colors[2])
ax[0].semilogy(energies, gd, color=colors[3])
ax[0].semilogy(energies, iod, color=colors[4])
ax[0].semilogy(energies, h2o, color='black')

# Plot vertical lines at the energy thresholds
ones = np.ones(50)
y_vals = np.linspace(-5, 1E4, 50)

ax[0].plot(16*ones, y_vals, color='black', ls='--')
ax[0].plot(50*ones, y_vals, color='black', ls='--')
ax[0].plot(54*ones, y_vals, color='black', ls='--')
ax[0].plot(63.5*ones, y_vals, color='black', ls='--')
ax[0].plot(81*ones, y_vals, color='black', ls='--')

ax[0].annotate('16 keV', (0.025, 0.84), xycoords='axes fraction', fontsize=13, rotation=90)
ax[0].annotate('50 keV', (0.29, 0.84), xycoords='axes fraction', fontsize=13, rotation=90)
ax[0].annotate('54 keV', (0.385, 0.84), xycoords='axes fraction', fontsize=13, rotation=90)
ax[0].annotate('64 keV', (0.475, 0.84), xycoords='axes fraction', fontsize=13, rotation=90)
ax[0].annotate('81 keV', (0.645, 0.84), xycoords='axes fraction', fontsize=13, rotation=90)

bluepatch = mpatches.Patch(color='dodgerblue', label='I')
purplepatch = mpatches.Patch(color='darkorchid', label='Gd')
redpatch = mpatches.Patch(color='crimson', label='Dy')
greenpatch = mpatches.Patch(color='mediumseagreen', label='Lu')
orangepatch = mpatches.Patch(color='orange', label='Au')
blackpatch = mpatches.Patch(color='black', label='H2O')

ax[0].legend(handles=[bluepatch, purplepatch, redpatch, greenpatch, orangepatch, blackpatch], fancybox=True, shadow=False,
           fontsize=15)
ax[0].set_xlabel('Energy (keV)', fontsize=18, labelpad=5)
ax[0].set_ylabel(r"$\mu / \rho$ $(cm^2 / g)$", fontsize=18)
ax[0].set_xlim([15, 120])
ax[0].set_ylim([1E-1, 1E3])
ax[0].tick_params(labelsize=15)
ax[0].set_title('a) Mass Attenuation', fontsize=20)

# All Spectra with Filtration

# 2.0 mm Al spectrum
import matplotlib.lines as mlines

folder = 'D:/Research/Bin Optimization/'
folder1 = folder + '/Npy Attenuation/'

spectrum = np.load(folder + 'Beam Spectrum/corrected-spectrum_120kV.npy')

energies = 1000*spectrum[:, 0]
spectrum = spectrum[:, 1]

Al_spectrum = np.load(folder + 'Al2.0_spectrum.npy')
Cu05_spectrum = np.load(folder + 'Cu0.5_spectrum.npy')
Cu1_spectrum = np.load(folder + 'Cu1.0_spectrum.npy')

ax[1].plot(energies, Al_spectrum, ls='-', color='black')
ax[1].plot(energies, Cu05_spectrum*2.25, ls='--', color='black')
ax[1].plot(energies, Cu1_spectrum*4.75, ls=':', color='black')

linepatch = mlines.Line2D([0], [0], color='black', lw=2, linestyle='-', label='2.0 mm Al')
dashpatch = mlines.Line2D([0], [0], color='black', lw=2, linestyle='--', label='0.5 mm Cu')
dotpatch = mlines.Line2D([0], [0], color='black', lw=2, linestyle=':', label='1.0 mm Cu')

ax[1].legend(handles=[linepatch, dashpatch, dotpatch], fancybox=True, shadow=False, fontsize=15)

ax[1].set_xlabel('Energy (keV)', fontsize=18, labelpad=5)
ax[1].set_ylabel('Relative Weight', fontsize=18, labelpad=5)
ax[1].set_xlim([0, 120])
ax[1].set_ylim([0, 7E-7])
ax[1].set_title('b) Beam Spectra', fontsize=20)
#plt.ylim([0, 3.2E-7])
ax[1].tick_params(labelsize=15)
ax[1].set_yticks([])
plt.subplots_adjust(left=0.1, right=0.95, wspace=0.25)
plt.show()
#plt.savefig(directory + 'Paper 1 Figures/AttenuationCoeffs.png', dpi=500)

#%% Figure 2 Bin Width
m = 15
mmin, mmax = -0.1, 4

imgAu = np.load(directory + folders[13] + '/Normed K-Edge/Bin4-3_Slice' + str(m) + '.npy')
imgGd = np.load(directory + folders[12] + '/Normed K-Edge/Bin1-0_Slice' + str(m) + '.npy')

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax1 = fig.add_subplot(111, frameon=False)
ax1.grid(False)
# Hide axes ticks
ax1.set_xticks([])
ax1.set_yticks([])

im0 = ax[0].imshow(imgGd, cmap=cmap1, vmin=mmin, vmax=mmax)
ax[0].grid(False)
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[0].set_title('c) Gadolinium', fontsize=20)

im1 = ax[1].imshow(imgAu, cmap=cmap3, vmin=mmin, vmax=mmax)
ax[1].grid(False)
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[1].set_title('d) Gold', fontsize=20)

# Add colorbars
d0 = make_axes_locatable(ax[0])
d1 = make_axes_locatable(ax[1])

cax0 = d0.append_axes("right", size="5%", pad=0.05)
cax0.tick_params(labelsize=15)
plt.colorbar(im0, cax=cax0)
h0 = cax0.set_ylabel('Concentration', fontsize=18, labelpad=25)
h0.set_rotation(-90)

cax1 = d1.append_axes("right", size="5%", pad=0.05)
cax1.tick_params(labelsize=15)
plt.colorbar(im1, cax=cax1)
h1 = cax1.set_ylabel('Concentration', fontsize=18, labelpad=25)
h1.set_rotation(-90)

plt.subplots_adjust(bottom=0.23, wspace=0.3)#, left=0.05, right=0.90, wspace=0.32)
plt.show()
#plt.savefig(directory + 'Paper 1 Figures/K-Edge-Bin.png', dpi=500)

#%% Figure 3 Time Acquisition
ked = '4-3'
m = 17
mmin, mmax = 0, 4

img1s = np.load(directory + folders[4] + '/Normed K-Edge/Bin' + ked + '_Slice' + str(m-9) + '.npy')
img05s = np.load(directory + folders[9] + '/Normed K-Edge/Bin' + ked + '_Slice' + str(m) + '.npy')
img01s = np.load(directory + folders[10] + '/Normed K-Edge/Bin' + ked + '_Slice' + str(m) + '.npy')

fig, ax = plt.subplots(1, 3, figsize=(14, 6))
ax1 = fig.add_subplot(111, frameon=False)
ax1.grid(False)
# Hide axes ticks
ax1.set_xticks([])
ax1.set_yticks([])

im0 = ax[0].imshow(img1s, cmap=cmap3, vmin=mmin, vmax=mmax)
ax[0].grid(False)
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[0].set_title('d) 1 s', fontsize=20)

im1 = ax[1].imshow(img05s, cmap=cmap3, vmin=mmin, vmax=mmax)
ax[1].grid(False)
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[1].set_title('e) 0.5 s', fontsize=20)

im2 = ax[2].imshow(img01s, cmap=cmap3, vmin=mmin, vmax=mmax)
ax[2].grid(False)
ax[2].set_xticks([])
ax[2].set_yticks([])
ax[2].set_title('f) 0.1 s', fontsize=20)

# Add colorbars
d0 = make_axes_locatable(ax[0])
d1 = make_axes_locatable(ax[1])
d2 = make_axes_locatable(ax[2])

cax0 = d0.append_axes("right", size="5%", pad=0.05)
cax0.tick_params(labelsize=15)
plt.colorbar(im0, cax=cax0)
h0 = cax0.set_ylabel('Concentration', fontsize=18, labelpad=25)
h0.set_rotation(-90)

cax1 = d1.append_axes("right", size="5%", pad=0.05)
cax1.tick_params(labelsize=15)
plt.colorbar(im1, cax=cax1)
h1 = cax1.set_ylabel('Concentration', fontsize=18, labelpad=25)
h1.set_rotation(-90)

cax2 = d2.append_axes("right", size="5%", pad=0.05)
cax2.tick_params(labelsize=15)
plt.colorbar(im2, cax=cax2)
h2 = cax2.set_ylabel('Concentration', fontsize=18, labelpad=25)
h2.set_rotation(-90)

plt.subplots_adjust(top=0.85, bottom=0.20, left=0.01, right=0.94, wspace=0.3)
plt.show()
#plt.savefig(directory + 'Paper 1 Figures/K-Edge-Time.png', dpi=500)

#%% Figure 4
zed = '2-1'
m = 10
mmin, mmax = 0, 4

img5p = np.load(directory + folders[1] + '/Normed K-Edge/Bin' + zed + '_Slice' + str(m+8) + '.npy')
img3p = np.load(directory + folders[4] + '/Normed K-Edge/Bin' + zed + '_Slice' + str(m-3) + '.npy')
img1p = np.load(directory + folders[7] + '/Normed K-Edge/Bin' + zed + '_Slice' + str(m-3) + '.npy')

fig, ax = plt.subplots(2, 3, figsize=(14, 10))
ax1 = fig.add_subplot(111, frameon=False)
ax1.grid(False)
# Hide axes ticks
ax1.set_xticks([])
ax1.set_yticks([])

im0 = ax[0, 0].imshow(img5p, cmap=cmap4, vmin=mmin, vmax=mmax)
ax[0, 0].grid(False)
ax[0, 0].set_xticks([])
ax[0, 0].set_yticks([])
ax[0, 0].set_title('a) 2.0 mm Al', fontsize=20)

im1 = ax[0, 1].imshow(img3p, cmap=cmap4, vmin=mmin, vmax=mmax)
ax[0, 1].grid(False)
ax[0, 1].set_xticks([])
ax[0, 1].set_yticks([])
ax[0, 1].set_title('b) 0.5 mm Cu', fontsize=20)

im2 = ax[0, 2].imshow(img1p, cmap=cmap4, vmin=mmin, vmax=mmax)
ax[0, 2].grid(False)
ax[0, 2].set_xticks([])
ax[0, 2].set_yticks([])
ax[0, 2].set_title('c) 1.0 mm Cu', fontsize=20)


# Add colorbars
d0 = make_axes_locatable(ax[0, 0])
d1 = make_axes_locatable(ax[0, 1])
d2 = make_axes_locatable(ax[0, 2])

cax0 = d0.append_axes("right", size="5%", pad=0.05)
cax0.tick_params(labelsize=15)
plt.colorbar(im0, cax=cax0)
h0 = cax0.set_ylabel('Concentration', fontsize=18, labelpad=25)
h0.set_rotation(-90)

cax1 = d1.append_axes("right", size="5%", pad=0.05)
cax1.tick_params(labelsize=18)
plt.colorbar(im1, cax=cax1)
h1 = cax1.set_ylabel('Concentration', fontsize=18, labelpad=25)
h1.set_rotation(-90)

cax2 = d2.append_axes("right", size="5%", pad=0.05)
cax2.tick_params(labelsize=15)
plt.colorbar(im2, cax=cax2)
h2 = cax2.set_ylabel('Concentration', fontsize=18, labelpad=25)
h2.set_rotation(-90)

zed = '4-3'

img5p = np.load(directory + folders[1] + '/Normed K-Edge/Bin' + zed + '_Slice' + str(m+8) + '.npy')
img3p = np.load(directory + folders[4] + '/Normed K-Edge/Bin' + zed + '_Slice' + str(m-3) + '.npy')
img1p = np.load(directory + folders[7] + '/Normed K-Edge/Bin' + zed + '_Slice' + str(m-3) + '.npy')

im3 = ax[1, 0].imshow(img5p, cmap=cmap3, vmin=mmin, vmax=mmax)
ax[1, 0].grid(False)
ax[1, 0].set_xticks([])
ax[1, 0].set_yticks([])
ax[1, 0].set_title('d) 2.0 mm Al', fontsize=20)

im4 = ax[1, 1].imshow(img3p, cmap=cmap3, vmin=mmin, vmax=mmax)
ax[1, 1].grid(False)
ax[1, 1].set_xticks([])
ax[1, 1].set_yticks([])
ax[1, 1].set_title('e) 0.5 mm Cu', fontsize=20)

im5 = ax[1, 2].imshow(img1p, cmap=cmap3, vmin=mmin, vmax=mmax)
ax[1, 2].grid(False)
ax[1, 2].set_xticks([])
ax[1, 2].set_yticks([])
ax[1, 2].set_title('f) 1.0 mm Cu', fontsize=20)


# Add colorbars
d3 = make_axes_locatable(ax[1, 0])
d4 = make_axes_locatable(ax[1, 1])
d5 = make_axes_locatable(ax[1, 2])

cax3 = d3.append_axes("right", size="5%", pad=0.05)
cax3.tick_params(labelsize=15)
plt.colorbar(im3, cax=cax3)
h3 = cax3.set_ylabel('Concentration', fontsize=18, labelpad=25)
h3.set_rotation(-90)

cax4 = d4.append_axes("right", size="5%", pad=0.05)
cax4.tick_params(labelsize=15)
plt.colorbar(im4, cax=cax4)
h4 = cax4.set_ylabel('Concentration', fontsize=18, labelpad=25)
h4.set_rotation(-90)

cax5 = d5.append_axes("right", size="5%", pad=0.05)
cax5.tick_params(labelsize=15)
plt.colorbar(im5, cax=cax5)
h5 = cax5.set_ylabel('Concentration', fontsize=18, labelpad=25)
h5.set_rotation(-90)

plt.annotate('Dysprosium', (0.4055, 0.935), xycoords='figure fraction', fontsize=25)
plt.annotate('Gold', (0.445, 0.44), xycoords='figure fraction', fontsize=25)

plt.subplots_adjust(bottom=0, wspace=0.29, left=0.02, right=0.94, top=0.91, hspace=0.18)

plt.show()
#plt.savefig(directory + 'Paper 1 Figures/K-Edge-Filter.png', dpi=500)
