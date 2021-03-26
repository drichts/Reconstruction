import os
import numpy as np
import matplotlib.pyplot as plt

directory = r'D:\OneDrive - University of Victoria\Research\LDA Data\21-03-11_CT_AuNPs'

folder = 'phantom_scan_1'
# Get air and water values from raw CT data images
data = np.load(os.path.join(directory, folder, 'CT', 'CT.npy'))
water_mask = np.load(os.path.join(directory, folder, 'water_mask.npy'))
air_mask = np.load(os.path.join(directory, folder, 'air_mask.npy'))
contrast_masks = np.load(os.path.join(directory, folder, 'contrast_masks_Au.npy'))

water_values = np.zeros((3, 4))
air_values = np.zeros((3, 4))

# Get the values for the 2 bins around the K-edge and the summed bin
for i in range(3):
    for j in range(4):
        water_values[i, j] = np.nanmean(data[i, j+11]*water_mask)
        air_values[i, j] = np.nanmean(data[i, j+11]*air_mask)

water_values = np.mean(water_values, axis=1)
air_values = np.mean(air_values, axis=1)

# Get the same from the K-edge images
data_k = data[1] - data[0]

k_edge_high = np.zeros(4)
k_edge_low = np.zeros(4)
for i in range(4):
    k_edge_high[i] = np.nanmean(data_k[i+11]*contrast_masks[0])
    k_edge_low[i] = np.nanmean(data_k[i+11]*water_mask)

k_edge_high = np.mean(k_edge_high)
k_edge_low = np.mean(k_edge_low)

np.save(os.path.join(directory, folder, 'CT_water_values.npy'), water_values)
np.save(os.path.join(directory, folder, 'CT_air_values.npy'), air_values)
np.save(os.path.join(directory, folder, 'K_edge_values.npy'), np.array([k_edge_high, k_edge_low]))