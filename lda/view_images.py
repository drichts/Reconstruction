import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import general_functions as gen
import mask_functions as msk

directory = r'D:\OneDrive - University of Victoria\Research\LDA Data\21-04_15_Probes'

folder = 'BCF_60_Thresholds_25_30_35_40_45_50_EC_CC'
data = np.load(os.path.join(directory, folder, 'Data', 'data.npy'))
air = np.load(os.path.join(directory, 'airscan_60s_50kVp_5mA_Thresholds_25_30_35_40_45_50_EC_CC', 'Data', 'data.npy')) / 6
dark = np.load(os.path.join(directory, 'darkscan_60s_50kVp_5mA_Thresholds_25_30_35_40_45_50_EC_CC', 'Data', 'data.npy')) / 6

air = np.sum(air[:, :, 0:2], axis=2)
dark = np.sum(dark[:, :, 0:2], axis=2)
data = np.sum(data[:, :, 0:2], axis=2)

# air = air[:, :, 0]
# dark = dark[:, :, 0]
# data = data[:, :, 0]

# data = np.sum(data1, axis=0)
proj = np.log(air-dark) - np.log(data-dark)


fig = plt.figure(figsize=(12, 4))
plt.imshow(proj, vmin=0.09, vmax=0.105)
plt.title('kV stationary')
plt.show()
