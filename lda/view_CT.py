import os
import numpy as np
import matplotlib.pyplot as plt

directory = r'D:\OneDrive - University of Victoria\Research\LDA Data'
# folder = '21-05-05_CT_metal_artifact'
folder = '21-05-05_CT_metal_30keV_bins'
sub_folder = '3_metal'  # water_only, 2_metal_1and2, 2_metal_3and4, 3_metal

# Parameters
thresholds = [70, 80, 90, 100, 110, 120]
bin_num = 2  # 0, 1, 2, 3, 4, 5  (There is no pile-up bin)
z_slice = 13  # The slice number
low = -400  # Low window level
high = 200  # High window level

data = np.load(os.path.join(directory, folder, sub_folder, 'Norm CT', 'CT_norm.npy'))

# Plot the desired data
fig = plt.figure(figsize=(7, 7))
plt.axis('off')
plt.imshow(data[bin_num, z_slice, 138:440, 138:440], vmin=low, vmax=high, cmap='gray')
if bin_num < 5:
    plt.title(f'{thresholds[bin_num]}-{thresholds[bin_num+1]} keV')
    plt.title(f'{thresholds[bin_num]}-{thresholds[bin_num+2]}')
else:
    plt.title('16-120 keV')
plt.show()
