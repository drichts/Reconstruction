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
import generateROImask as grm
import sCT_Analysis as sct

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

#%% Air masks

# Calculate the air masks

# Go through each of the folders
#for i, folder in enumerate(folders[9:11]):
#    image = np.load(directory + folder + '/Slices/Bin6_Slice13.npy')

#    continue_flag = True
#    while continue_flag:
#        air = grm.air_mask(image)
#        val = input('Were the ROIs acceptable? (y/n)')
#        if val is 'y':
#            continue_flag = False

    # Save if desired
#    np.save(directory + folder + '/Air_Mask.npy', air)

#%% Calculate the average CNR in K-edge images for filter and time

# Select the folder from the folders list
folder_num = 10
folder = folders[folder_num]
#folder = folder3
low_z, high_z = good_slices[folder_num][0], good_slices[folder_num][1]
#low_z, high_z = gs2[0], gs2[1]

# The K-edge subtractions
k_edges = ['1-0', '2-1', '3-2', '4-3']

# The masks
vials = np.load(directory + folder + '/Vial_Masks.npy')
air = np.load(directory + folder + '/Phantom_Mask.npy')

# The order of the vials based on K-edge
order = [4, 2, 3, 1]

# Go through each of the k-edges
for i, k in enumerate(k_edges):
    mean_conc = np.zeros(high_z - low_z + 1)  # Hold the CNR of each slice
    std_conc = np.zeros(high_z - low_z + 1)  # Hold the std. dev of the CNR
    vial = vials[order[i]]

    # Go through all the of the good slices
    for j in np.arange(low_z, high_z+1):
        image = np.load(directory + folder + '/K-Edge/Bin' + k + '_Slice' + str(j) + '.npy')
        mean_conc[j - low_z], std_conc[j - low_z] = sct.cnr(image, vial, air)

    print(k, np.mean(mean_conc), np.mean(std_conc))

#%% Calculate the average CNR in K-edge images for bin width

# Select the folder from the folders list
folder_num = 12
folder = folders[folder_num]
#folder = folder3
low_z, high_z = good_slices[folder_num][0], good_slices[folder_num][1]
#low_z, high_z = gs2[0], gs2[1]

# The K-edge subtractions
k_edges = ['1-0', '4-3']

# The masks
vials = np.load(directory + folder + '/Vial_Masks.npy')
air = np.load(directory + folder + '/Phantom_Mask.npy')

mean_conc = np.zeros(high_z - low_z + 1)  # Hold the CNR of each slice
std_conc = np.zeros(high_z - low_z + 1)  # Hold the std. dev of the CNR

# Go through all the of the good slices for '1-0'
for j in np.arange(low_z, high_z+1):
    image = np.load(directory + folder + '/K-Edge/Bin' + k_edges[0] + '_Slice' + str(j) + '.npy')
    mean_conc[j - low_z], std_conc[j - low_z] = sct.cnr(image, vials[4], air)

print(k_edges[0], np.mean(mean_conc), np.mean(std_conc))

mean_conc = np.zeros(high_z - low_z + 1)  # Hold the CNR of each slice
std_conc = np.zeros(high_z - low_z + 1)  # Hold the std. dev of the CNR

# Go through all the of the good slices for '1-0'
for j in np.arange(low_z, high_z + 1):
    image = np.load(directory + folder + '/K-Edge/Bin' + k_edges[1] + '_Slice' + str(j) + '.npy')
    mean_conc[j - low_z], std_conc[j - low_z] = sct.cnr(image, vials[1], air)

print(k_edges[1], np.mean(mean_conc), np.mean(std_conc))

#%% Calculate the average concentration in K-edge images for bin width

# Select the folder from the folders list
folder_num = 12
folder = folders[folder_num]
#folder = folder3
low_z, high_z = good_slices[folder_num][0], good_slices[folder_num][1]
#low_z, high_z = gs2[0], gs2[1]

# The K-edge subtractions
k_edges = ['1-0', '4-3']

# The masks
vials = np.load(directory + folder + '/Vial_Masks.npy')
air = np.load(directory + folder + '/Phantom_Mask.npy')

mean_conc = np.zeros(high_z - low_z + 1)  # Hold the CNR of each slice
std_conc = np.zeros(high_z - low_z + 1)  # Hold the std. dev of the CNR

# Go through all the of the good slices for '1-0'
for j in np.arange(low_z, high_z+1):
    image = np.load(directory + folder + '/Normed K-Edge/Bin' + k_edges[0] + '_Slice' + str(j) + '.npy')
    mean_conc[j - low_z] = np.nanmean(image*vials[4])
    std_conc[j - low_z] = np.nanstd(image*vials[4])

print(k_edges[0], np.mean(mean_conc), np.mean(std_conc))

mean_conc = np.zeros(high_z - low_z + 1)  # Hold the CNR of each slice
std_conc = np.zeros(high_z - low_z + 1)  # Hold the std. dev of the CNR

# Go through all the of the good slices for '1-0'
for j in np.arange(low_z, high_z + 1):
    image = np.load(directory + folder + '/Normed K-Edge/Bin' + k_edges[1] + '_Slice' + str(j) + '.npy')
    mean_conc[j - low_z] = np.nanmean(image * vials[1])
    std_conc[j - low_z] = np.nanstd(image * vials[1])

print(k_edges[1], np.mean(mean_conc), np.mean(std_conc))