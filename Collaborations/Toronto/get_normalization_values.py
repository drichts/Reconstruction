import os
import numpy as np
import matplotlib.pyplot as plt

directory = r'D:\OneDrive - University of Victoria\Research\LDA Data\21-03-24_CT_NP_nophantom_normalization_ring_corr'

folder = 'phantom_scan'
# Get air and water values from raw CT data images
data = np.load(os.path.join(directory, folder, 'CT', 'CT.npy'))
water_mask = np.load(os.path.join(directory, folder, 'water_mask.npy'))
air_mask = np.load(os.path.join(directory, folder, 'air_mask.npy'))
contrast_masks = np.load(os.path.join(directory, folder, 'contrast_masks_Au.npy'))

water_values = np.zeros(3)
air_values = np.zeros(3)

# Get the values for the 2 bins around the K-edge and the summed bin
for i in range(3):
    water_values[i] = np.nanmean(data[i, 12]*water_mask)
    air_values[i] = np.nanmean(data[i, 12]*air_mask)

# Get the same from the K-edge images
data_k = data[1] - data[0]
k_edge_high = np.nanmean(data_k[12]*contrast_masks[0])
k_edge_low = np.nanmean(data_k[12]*water_mask)

np.save(os.path.join(directory, folder, 'CT_water_values.npy'), water_values)
np.save(os.path.join(directory, folder, 'CT_air_values.npy'), air_values)
np.save(os.path.join(directory, folder, 'K_edge_values.npy'), np.array([k_edge_high, k_edge_low]))