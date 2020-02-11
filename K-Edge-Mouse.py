import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable


# Things to enter
directory = 'D:/Research/Python Data/Spectral CT/'
folder = 'AuGdLu_5-27-19/'

# Load the two bin matrices in question
img1 = np.load(directory + folder + 'K-Edge/Bin2-1_Slice14.npy')
img2 = np.load(directory + folder + 'K-Edge/Bin3-2_Slice14.npy')
img3 = np.load(directory + folder + 'K-Edge/Bin4-3_Slice14.npy')
img6 = np.load(directory + folder + 'Slices/Bin6_Slice14.npy')


#gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])
#fig, ax = plt.subplots(2, 2, figsize=(12, 12), gridspec_kw={'height_ratios': [1, 1], 'width_ratios': [1, 1]})
#plt.setp(ax, xticks=[4, 60, 116], xticklabels=['-1.5', '0.0', '1.5'],
#         yticks=[4, 60, 116], yticklabels=['-1.5', '0.0', '1.5'])

# Create the colormaps
nbins = 100
c1 = (1, 0, 1)
c2 = (0, 1, 0)
c3 = (1, 0.843, 0)
c4 = (0, 0, 1)

c1_rng = [(0, 0, 0), c1]
cmap1 = colors.LinearSegmentedColormap.from_list('Purp', c1_rng, N=nbins)
c2_rng = [(0, 0, 0), c2]
cmap2 = colors.LinearSegmentedColormap.from_list('Gree', c2_rng, N=nbins)
c3_rng = [(0, 0, 0), c3]
cmap3 = colors.LinearSegmentedColormap.from_list('G78', c3_rng, N=nbins)
c4_rng = [(0, 0, 0), c4]
cmap4 = colors.LinearSegmentedColormap.from_list('Blu8', c4_rng, N=nbins)

gray_val = 0.34
gray_list = (gray_val, gray_val, gray_val)

cmap5 = colors.LinearSegmentedColormap.from_list('1ab', [gray_list, c1], N=nbins)
cmap5.set_bad('white', alpha=0)
cmap5.set_over(c1, alpha=1)
cmap5.set_under('white', alpha=0)
cmap6 = colors.LinearSegmentedColormap.from_list('2ab', [gray_list, c2], N=nbins)
cmap6.set_bad('white', alpha=0)
cmap6.set_over(c2, alpha=1)
cmap6.set_under('white', alpha=0)
cmap7 = colors.LinearSegmentedColormap.from_list('3ab', [gray_list, c3], N=nbins)
cmap7.set_bad('white', alpha=0)
cmap7.set_over(c3, alpha=1)
cmap7.set_under('white', alpha=0)
cmap8 = colors.LinearSegmentedColormap.from_list('4ab', [gray_list, c4], N=nbins)
cmap8.set_bad('white', alpha=0)
cmap8.set_over(c4, alpha=1)
cmap8.set_under('white', alpha=0)

fig, ax = plt.subplots(2, 2, figsize=(12, 12), gridspec_kw={'height_ratios': [1, 1], 'width_ratios': [1, 1]})
plt.setp(ax, xticks=[4, 60, 116], xticklabels=['-1.5', '0.0', '1.5'],
         yticks=[4, 60, 116], yticklabels=['-1.5', '0.0', '1.5'])

low = 20
high = 350

ax[0, 0].imshow(img1, cmap=cmap1, vmin=low, vmax=high)
ax[0, 0].set_title('a) Gadolinium (2%)', fontsize=35)
ax[0, 0].tick_params(labelsize=25)
ax[0, 0].set_xlabel('x (cm)', fontsize=30)
ax[0, 0].set_ylabel('y (cm)', fontsize=30)

ax[0, 1].imshow(img2, cmap=cmap2, vmin=low, vmax=high)
ax[0, 1].set_title('b) Lutetium (2%)', fontsize=35)
ax[0, 1].tick_params(labelsize=25)
ax[0, 1].set_xlabel('x (cm)', fontsize=30)
ax[0, 1].set_ylabel('y (cm)', fontsize=30)

ax[1, 0].imshow(img3, cmap=cmap3, vmin=low, vmax=high)
ax[1, 0].set_title('c) Gold (2%)', fontsize=35)
ax[1, 0].tick_params(labelsize=25)
ax[1, 0].set_xlabel('x (cm)', fontsize=30)
ax[1, 0].set_ylabel('y (cm)', fontsize=30)

im = ax[1, 1].imshow(img6, cmap='gray', vmin=-500, vmax=1000)
ax[1, 1].imshow(img1, cmap=cmap5, norm=colors.Normalize(vmin=75, vmax=200), alpha=1)
ax[1, 1].imshow(img2, cmap=cmap6, norm=colors.Normalize(vmin=75, vmax=200), alpha=1)
ax[1, 1].imshow(img3, cmap=cmap7, norm=colors.Normalize(vmin=75, vmax=200), alpha=1)
ax[1, 1].set_title('d) Composite Image', fontsize=35)
ax[1, 1].tick_params(labelsize=25)
ax[1, 1].set_xlabel('x (cm)', fontsize=30)
ax[1, 1].set_ylabel('y (cm)', fontsize=30)
d4 = make_axes_locatable(ax[1, 1])
cax4 = d4.append_axes("right", size="5%", pad=0.05)
cax4.tick_params(labelsize=25)
plt.colorbar(im, cax=cax4)
h4 = cax4.set_ylabel('HU', fontsize=30, labelpad=10)
h4.set_rotation(-90)

plt.show()

