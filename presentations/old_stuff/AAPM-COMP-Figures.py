import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
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
folder2 = 'Cu_0.5_Time_1.0_02-20-20'
folder3 = 'Cu_0.5_Time1.0_Uniformity_02-25-20'
gs2 = [12, 18]

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

#%% Figure 2

# Calculate y points from the fit above
xpts = np.linspace(0, 25, 50)
sns.set_style('whitegrid')

fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
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
ax[1].plot(xpts, p_Au(xpts), color=colors[0], lw=2)
ax[0].plot(xpts, p_Gd(xpts), color=colors[1], lw=2)

ax[1].errorbar(Au_widths, Au_CNR, yerr=Au_std, fmt='none', capsize=4, color=colors[0])
ax[0].errorbar(Gd_widths, Gd_CNR, yerr=Gd_std, fmt='none', capsize=4, color=colors[1])

ax[0].set_xlim([4, 15])
ax[1].set_xlim([4, 21])
ax[0].set_ylim([0, 30])

ax[0].set_title('a) Gadolinum', fontsize=18)
ax[1].set_title('b) Gold', fontsize=18)

ax[0].tick_params(labelsize=16)
ax[1].tick_params(labelsize=16)

#orangepatch = mpatches.Patch(color=colors[0], label='Au (Z=79)')
#purplepatch = mpatches.Patch(color=colors[1], label='Gd (Z=64)')
#leg = plt.legend(handles=[purplepatch, orangepatch], loc='lower center', bbox_to_anchor=(0.5, -0.28), ncol=2,
#                 fancybox=True, fontsize=20)
#ax1.add_artist(leg)
ax1.set_ylabel('K-Edge CNR', fontsize=18, labelpad=40)
ax1.set_xlabel('Bin Width (keV)', fontsize=18, labelpad=35)
plt.subplots_adjust(bottom=0.23, wspace=0.1)
plt.show()
#plt.savefig(directory + 'AAPM COMP 2020 Figures/Figure2.png', dpi=500)

#%% Find the bin width energy that maximizes K-edge CNR

xpts = np.linspace(4, 22, 1000)

au = p_Au(xpts)
gd = p_Gd(xpts)

au_ind = np.argmax(au)
gd_ind = np.argmax(gd)

print(xpts[au_ind])
print(xpts[gd_ind])

#%% Figure 3

xpts = np.array([1, 2, 3, 4])
# Calculate y points from the fit above
sns.set_style('whitegrid')

fig, ax = plt.subplots(1, 2, sharey=True, figsize=(8, 3))
ax1 = fig.add_subplot(111, frameon=False)
ax1.grid(False)
# Hide axes ticks
ax1.set_xticks([])
ax1.set_yticks([])

# Order is Au, Lu, Dy, Gd
colors = ['orange', 'mediumseagreen', 'crimson', 'darkorchid']
titles = ['a) 1 s', 'b) 0.5 s', 'c) 0.1 s']

for i, folder in enumerate(np.concatenate(([folder3], [folders[4]]))):

    mean_CNR = np.load(directory + folder + '/Mean_Kedge_CNR_Time.npy')
    std_CNR = np.load(directory + folder + '/Std_Kedge_CNR_Time.npy')

    ax[i].bar(xpts, mean_CNR, color=colors, yerr=std_CNR, capsize=4, edgecolor='black')
    ax[i].set_xticks(xpts)
    ax[i].set_xticklabels(['Au', 'Lu', 'Dy', 'Gd'])
    ax[i].set_title(titles[i], fontsize=11)
    ax[i].tick_params(labelsize=10)
    #ax[i].set_ylim([0, 24])

ax1.set_ylabel('K-Edge CNR', fontsize=11, labelpad=30)
ax1.set_xlabel('Contrast Element (3% Concentration)', fontsize=11, labelpad=30)
plt.subplots_adjust(top=0.88, bottom=0.21, left=0.125, right=0.9, hspace=0.2, wspace=0.2)
plt.show()
#plt.savefig(directory + 'AAPM COMP 2020 Figures/Figure3.png', dpi=500)

#%% Figure 4

concentration = np.array([5, 3, 1, 0])
# Calculate y points from the fit above
xpts = np.linspace(6, -0.5, 50)
sns.set_style('whitegrid')

# set width of bar
barWidth = 0.3

fig = plt.figure(figsize=(7, 5))

# Order is Au, Lu, Dy, Gd
colors = ['orange', 'mediumseagreen', 'crimson', 'darkorchid']
textures = ['//', '\ \ ', '--', 'xx']
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
plt.bar(r1, bars[0], yerr=stddev[0], color=colors[0], width=barWidth-bar_sub, edgecolor='black', label=labels[0],
        capsize=3)
plt.bar(r2, bars[1], yerr=stddev[1], color=colors[1], width=barWidth-bar_sub, edgecolor='black', label=labels[1],
        capsize=3)
plt.bar(r3, bars[2], yerr=stddev[2], color=colors[2], width=barWidth-bar_sub, edgecolor='black', label=labels[2],
        capsize=3)
plt.bar(r4, bars[3], yerr=stddev[3], color=colors[3], width=barWidth-bar_sub, edgecolor='black', label=labels[3],
        capsize=3)

purplepatch = mpatches.Patch(facecolor='darkorchid', label='Gd (Z=64)', edgecolor='black')
redpatch = mpatches.Patch(facecolor='crimson', label='Dy (Z=66)', edgecolor='black')
greenpatch = mpatches.Patch(facecolor='mediumseagreen', label='Lu (Z=71)', edgecolor='black')
orangepatch = mpatches.Patch(facecolor='orange', label='Au (Z=79)', edgecolor='black')

plt.xticks(concs, ['0%', '1%', '3%', '5%'])
plt.legend(handles=[purplepatch, redpatch, greenpatch, orangepatch], loc='upper left', fancybox=True, shadow=True,
           fontsize=14)
plt.ylabel('K-Edge Concentration (%)', fontsize=16, labelpad=10)
plt.xlabel('True Concentration', fontsize=16, labelpad=5)
plt.tick_params(labelsize=14)
plt.subplots_adjust(top=0.97,
bottom=0.181,
left=0.139,
right=0.979,
hspace=0.2,
wspace=0.2)
plt.show()
#plt.savefig(directory + 'AAPM COMP 2020 Figures/Figure4.png', dpi=500)

#%% Setup Figure

lan = mpimg.imread('C:/Users/drich/OneDrive/Pictures/Multi-contrast imaging with PCCT/Lanthanide_Paper1.png')
augd = mpimg.imread('C:/Users/drich/OneDrive/Pictures/Multi-contrast imaging with PCCT/AuGd_Phantom.png')
setup = mpimg.imread('C:/Users/drich/OneDrive/Pictures/Multi-contrast imaging with PCCT/Setup_Labeled.png')
dimen = mpimg.imread('C:/Users/drich/OneDrive/Pictures/Multi-contrast imaging with PCCT/Dimensions2.png')

# Plot figure with subplots of different sizes
fig = plt.figure(figsize=(9, 9))
# set up subplot grid
gridspec.GridSpec(3, 6)

# large subplot
plt.subplot2grid((3, 6), (0, 0), colspan=5, rowspan=3, frameon=False)
plt.imshow(setup)
plt.grid(False)
plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, labelbottom=False, labelleft=False)
plt.title('a) Experimental Setup', fontsize=15)

# small subplot 1
plt.subplot2grid((3, 6), (0, 5), frameon=False)
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
plt.title('b) Lanthanide\nPhantom', fontsize=15)

# small subplot 2
plt.subplot2grid((3, 6), (1, 5), frameon=False)
plt.imshow(augd)
plt.grid(False)
plt.title('c) AuGd\nPhantom', fontsize=15)
plt.annotate('3%', (350, 785), xycoords='data', fontsize=15)
plt.annotate('0.5%', (315, 1400), xycoords='data', fontsize=12)
plt.annotate('Mix', (875, 1710), xycoords='data', fontsize=15)
plt.annotate('3%', (1425, 1400), xycoords='data', fontsize=15)
plt.annotate('0.5%', (1375, 775), xycoords='data', fontsize=12)
plt.annotate(r'$H_2 O$', (840, 470), xycoords='data', fontsize=13)
plt.annotate(r'$H_2 O$', (840, 1090), xycoords='data', fontsize=13)
plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, labelbottom=False, labelleft=False)

# small subplot 2
plt.subplot2grid((3, 6), (2, 5), frameon=False)
plt.imshow(dimen)
plt.grid(False)
plt.title('d) Phantom\nDimensions', fontsize=15)
plt.annotate('0.6 cm', (1850, 160), xycoords='data', fontsize=12)
plt.annotate('2.5\ncm', (0, 1100), xycoords='data', fontsize=12)
plt.annotate('3 cm', (1100, 1940), xycoords='data', fontsize=12)
plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, labelbottom=False, labelleft=False)

#plt.subplots_adjust()

plt.show()
#plt.savefig(directory + 'AAPM COMP 2020 Figures/Setup_Figure.png', dpi=500)

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
#plt.savefig(directory + 'AAPM COMP 2020 Figures/K-Edge-Bin.png', dpi=500)

#%% Figure 3 Time Acquisition
ked = '4-3'
m = 17
mmin, mmax = 0, 4

img1s = np.load(directory + folder3 + '/Normed K-Edge/Bin' + ked + '_Slice' + str(m+1) + '.npy')
img05s = np.load(directory + folders[9] + '/Normed K-Edge/Bin' + ked + '_Slice' + str(m-2) + '.npy')
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
h0 = cax0.set_ylabel('Concentration', fontsize=13, labelpad=25)
h0.set_rotation(-90)

cax1 = d1.append_axes("right", size="5%", pad=0.05)
cax1.tick_params(labelsize=15)
plt.colorbar(im1, cax=cax1)
h1 = cax1.set_ylabel('Concentration', fontsize=18, labelpad=25)
h1.set_rotation(-90)

cax2 = d2.append_axes("right", size="5%", pad=0.05)
cax2.tick_params(labelsize=15)
plt.colorbar(im2, cax=cax2)
h2 = cax2.set_ylabel('Concentration', fontsize=13, labelpad=25)
h2.set_rotation(-90)

plt.subplots_adjust(top=0.85, bottom=0.20, left=0.01, right=0.94, wspace=0.3)
plt.show()
#plt.savefig(directory + 'AAPM COMP 2020 Figures/K-Edge-Time.png', dpi=500)

#%% Figure 4
zed = '2-1'
m = 10
mmin, mmax = -0.05, 3.6

img5p = np.load(directory + folders[1] + '/Normed K-Edge/Bin' + zed + '_Slice' + str(m+8) + '.npy')
img3p = np.load(directory + folders[4] + '/Normed K-Edge/Bin' + zed + '_Slice' + str(m-3) + '.npy')
img1p = np.load(directory + folders[7] + '/Normed K-Edge/Bin' + zed + '_Slice' + str(m-3) + '.npy')

fig, ax = plt.subplots(2, 3, figsize=(9, 6))
ax1 = fig.add_subplot(111, frameon=False)
ax1.grid(False)
# Hide axes ticks
ax1.set_xticks([])
ax1.set_yticks([])

im0 = ax[0, 0].imshow(img5p, cmap=cmap4, vmin=mmin, vmax=mmax)
ax[0, 0].grid(False)
ax[0, 0].set_xticks([])
ax[0, 0].set_yticks([])
ax[0, 0].set_title('a) 2.0 mm Al', fontsize=14)
ax[0, 0].set_ylabel('Dysprosium', fontsize=16, labelpad=5)

im1 = ax[0, 1].imshow(img3p, cmap=cmap4, vmin=mmin, vmax=mmax)
ax[0, 1].grid(False)
ax[0, 1].set_xticks([])
ax[0, 1].set_yticks([])
ax[0, 1].set_title('b) 0.5 mm Cu', fontsize=14)

im2 = ax[0, 2].imshow(img1p, cmap=cmap4, vmin=mmin, vmax=mmax)
ax[0, 2].grid(False)
ax[0, 2].set_xticks([])
ax[0, 2].set_yticks([])
ax[0, 2].set_title('c) 1.0 mm Cu', fontsize=14)

zed = '4-3'

img5p = np.load(directory + folders[1] + '/Normed K-Edge/Bin' + zed + '_Slice' + str(m+8) + '.npy')
img3p = np.load(directory + folders[4] + '/Normed K-Edge/Bin' + zed + '_Slice' + str(m-3) + '.npy')
img1p = np.load(directory + folders[7] + '/Normed K-Edge/Bin' + zed + '_Slice' + str(m-3) + '.npy')

im3 = ax[1, 0].imshow(img5p, cmap=cmap3, vmin=mmin, vmax=mmax)
ax[1, 0].grid(False)
ax[1, 0].set_xticks([])
ax[1, 0].set_yticks([])
ax[1, 0].set_title('d) 2.0 mm Al', fontsize=14)
ax[1, 0].set_ylabel('Gold', fontsize=16, labelpad=5)

im4 = ax[1, 1].imshow(img3p, cmap=cmap3, vmin=mmin, vmax=mmax)
ax[1, 1].grid(False)
ax[1, 1].set_xticks([])
ax[1, 1].set_yticks([])
ax[1, 1].set_title('e) 0.5 mm Cu', fontsize=14)

im5 = ax[1, 2].imshow(img1p, cmap=cmap3, vmin=mmin, vmax=mmax)
ax[1, 2].grid(False)
ax[1, 2].set_xticks([])
ax[1, 2].set_yticks([])
ax[1, 2].set_title('f) 1.0 mm Cu', fontsize=14)


cbar_ax1 = fig.add_axes([0.90, 0.522, 0.03, 0.353])
cbar_ax1.tick_params(labelsize=15)
fig.colorbar(im2, cax=cbar_ax1)
h1 = cbar_ax1.set_ylabel('Concentration (%)', fontsize=14, labelpad=25)
h1.set_rotation(-90)

cbar_ax2 = fig.add_axes([0.90, 0.11, 0.03, 0.353])
cbar_ax2.tick_params(labelsize=15)
fig.colorbar(im5, cax=cbar_ax2)
h2 = cbar_ax2.set_ylabel('Concentration (%)', fontsize=14, labelpad=25)
h2.set_rotation(-90)

plt.subplots_adjust(top=0.875,
bottom=0.11,
left=0.13,
right=0.89,
hspace=0.17,
wspace=0.0)

plt.show()
plt.savefig(directory + 'AAPM COMP 2020 Figures/K-Edge-Filter.png', dpi=500)
