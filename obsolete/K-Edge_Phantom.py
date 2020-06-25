import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import operator
from PIL import Image
from PIL import ImageDraw

# Things to enter
directory = 'D:/Research/Python Data/Spectral CT/'
folder = 'Al_2.0_10-17-19_3P/'
append = folder[0:6]
percent = 3
st_per = str(percent) + '%'
title = append + ', ' + st_per
title = folder[0:15] + ', ' + st_per
# Load the two bin matrices in question
img1_0 = np.load(directory + folder + 'K-Edge/Bin1-0_Slice14.npy')
img2_1 = np.load(directory + folder + 'K-Edge/Bin2-1_Slice14.npy')
img3_2 = np.load(directory + folder + 'K-Edge/Bin3-2_Slice14.npy')
img4_3 = np.load(directory + folder + 'K-Edge/Bin4-3_Slice14.npy')
img6 = np.load(directory + folder + 'Slices/Bin6_Slice14.npy')

masks = np.load(directory + folder + 'Vial_Masks.npy')
v1 = masks[1]
v2 = masks[2]
v3 = masks[3]
v4 = masks[4]
#v1 = np.load(directory + folder + 'Vial1_MaskMatrix.npy')
#v2 = np.load(directory + folder + 'Vial2_MaskMatrix.npy')
#v3 = np.load(directory + folder + 'Vial3_MaskMatrix.npy')
#v4 = np.load(directory + folder + 'Vial4_MaskMatrix.npy')

if percent is 5:
    val4 = np.nanmean(v1*img4_3)
    val2 = np.nanmean(v2*img2_1)
    val3 = np.nanmean(v3*img3_2)
    val1 = np.nanmean(v4*img1_0)
    values = np.array([val1, val2, val3, val4])
    np.save(directory + append + '_K-edge_upper_thresholds.npy', values)
else:
    values = np.load(directory + append + '_K-edge_upper_thresholds.npy')
    val1 = values[0]
    val2 = values[1]
    val3 = values[2]
    val4 = values[3]

print(val1, val2, val3, val4)

# Create the colormaps
nbins = 100
c1 = (1, 0, 1)
c2 = (0, 1, 0)
c3 = (1, 0.843, 0)
c4 = (0, 0, 1)

gray_val = 0
gray_list = (gray_val, gray_val, gray_val)

c1_rng = [gray_list, c1]
cmap1 = colors.LinearSegmentedColormap.from_list('Purp', c1_rng, N=nbins)
c2_rng = [gray_list, c2]
cmap2 = colors.LinearSegmentedColormap.from_list('Gree', c2_rng, N=nbins)
c3_rng = [gray_list, c3]
cmap3 = colors.LinearSegmentedColormap.from_list('G78', c3_rng, N=nbins)
c4_rng = [gray_list, c4]
cmap4 = colors.LinearSegmentedColormap.from_list('Blu8', c4_rng, N=nbins)

fig, ax = plt.subplots(2, 2, figsize=(12, 12), gridspec_kw={'height_ratios': [1, 1], 'width_ratios': [1, 1]})
plt.setp(ax, xticks=[4, 60, 116], xticklabels=['-1.5', '0.0', '1.5'],
         yticks=[4, 60, 116], yticklabels=['-1.5', '0.0', '1.5'])
fig.add_subplot(111, frameon=False, xticklabels=[], yticklabels=[])
plt.title(title, fontsize=30, pad=40)
divisor = 100
low1 = val1/divisor
low2 = val2/divisor
low3 = val3/divisor
low4 = val4/divisor

ax[0, 0].imshow(img1_0, cmap=cmap1, vmin=low1, vmax=val1)
ax[0, 0].set_title('a) Gadolinium (' + st_per + ')', fontsize=25)
ax[0, 0].tick_params(labelsize=20)
ax[0, 0].set_xlabel('x (cm)', fontsize=20)
ax[0, 0].set_ylabel('y (cm)', fontsize=20)

ax[0, 1].imshow(img2_1, cmap=cmap2, vmin=low2, vmax=val2)
ax[0, 1].set_title('b) Dysprosium (' + st_per + ')', fontsize=25)
ax[0, 1].tick_params(labelsize=20)
ax[0, 1].set_xlabel('x (cm)', fontsize=20)
ax[0, 1].set_ylabel('y (cm)', fontsize=20)

ax[1, 0].imshow(img3_2, cmap=cmap4, vmin=low3, vmax=val3)
ax[1, 0].set_title('c) Lutetium (' + st_per + ')', fontsize=25)
ax[1, 0].tick_params(labelsize=20)
ax[1, 0].set_xlabel('x (cm)', fontsize=20)
ax[1, 0].set_ylabel('y (cm)', fontsize=20)

ax[1, 1].imshow(img4_3, cmap=cmap3, vmin=low4, vmax=val4)
ax[1, 1].set_title('d) Gold (' + st_per + ')', fontsize=25)
ax[1, 1].tick_params(labelsize=20)
ax[1, 1].set_xlabel('x (cm)', fontsize=20)
ax[1, 1].set_ylabel('y (cm)', fontsize=20)

plt.subplots_adjust(hspace=0.4)

plt.show()
#plt.close()


#%%
gray_val = 0.2
gray_list = (gray_val, gray_val, gray_val)

cmap5 = colors.LinearSegmentedColormap.from_list('1ab', [gray_list, c1], N=nbins)
cmap5.set_bad('black', alpha=0)
cmap5.set_over(c1, alpha=1)
cmap5.set_under('black', alpha=0)
cmap6 = colors.LinearSegmentedColormap.from_list('2ab', [gray_list, c2], N=nbins)
cmap6.set_bad('black', alpha=0)
cmap6.set_over(c2, alpha=1)
cmap6.set_under('black', alpha=0)
cmap7 = colors.LinearSegmentedColormap.from_list('3ab', [gray_list, c3], N=nbins)
cmap7.set_bad('black', alpha=0)
cmap7.set_over(c3, alpha=1)
cmap7.set_under('black', alpha=0)
cmap8 = colors.LinearSegmentedColormap.from_list('4ab', [gray_list, c4], N=nbins)
cmap8.set_bad('black', alpha=0)
cmap8.set_over(c4, alpha=1)
cmap8.set_under('black', alpha=0)

plt.imshow(img6, cmap='gray', vmin=-500, vmax=1000, alpha=1)
#plt.imshow(img1_0, cmap=cmap5, norm=colors.Normalize(vmin=low1, vmax=val1), alpha=0.8)
#plt.imshow(img2_1, cmap=cmap6, norm=colors.Normalize(vmin=low2, vmax=val2), alpha=0.8)
#plt.imshow(img3_2, cmap=cmap8, norm=colors.Normalize(vmin=low3, vmax=val3), alpha=0.8)
#plt.imshow(img4_3, cmap=cmap7, norm=colors.Normalize(vmin=low4, vmax=val4), alpha=0.8)

plt.imshow(img1_0, cmap=cmap5, vmin=low1, vmax=val1, alpha=1)
plt.imshow(img2_1, cmap=cmap6, vmin=low2, vmax=val2, alpha=1)
plt.imshow(img3_2, cmap=cmap8, vmin=low3, vmax=val3, alpha=1)
plt.imshow(img4_3, cmap=cmap7, vmin=low4, vmax=val4, alpha=1)
plt.title(title + ' Composite Image', fontsize=40, pad=20)
plt.xticks((4, 60, 116), ('-1.5', '0.0', '1.5'), fontsize=30)
plt.yticks((4, 60, 116), ('-1.5', '0.0', '1.5'), fontsize=30)
plt.xlabel('x (cm)', fontsize=30)
plt.ylabel('y (cm)', fontsize=30)
plt.show()
#plt.close()
