import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import mask_functions as msk
import general_functions as gen


directory = '/home/knoll/LDAData'
folder1 = '21-02-26_CT_min_Au_SEC'
folder2 = '21-03-01_CT_min_Gd_3mA_SEC'

data1_k = np.load(os.path.join(directory, folder1, 'phantom_scan', 'Norm CT', 'K-edge.npy'))

data2_k = np.load(os.path.join(directory, folder2, 'phantom_scan', 'Norm CT', 'K-edge.npy'))

water1 = np.load(os.path.join(directory, folder1, 'phantom_scan', 'water_mask.npy'))
air1 = np.load(os.path.join(directory, folder1, 'phantom_scan', 'air_mask.npy'))
contrast1 = np.load(os.path.join(directory, folder1, 'phantom_scan', 'contrast_masks.npy'))

water2 = np.load(os.path.join(directory, folder2, 'phantom_scan', 'water_mask.npy'))
air2 = np.load(os.path.join(directory, folder2, 'phantom_scan', 'air_mask.npy'))
contrast2 = np.load(os.path.join(directory, folder2, 'phantom_scan', 'contrast_masks.npy'))

cnr1 = np.zeros((24, 2, 7))
cnr2 = np.zeros((24, 2, 9))

for i in range(24):
    for j in range(6):
        cnr1[i, :, j] = gen.cnr(data1_k[i], contrast1[j], water1)
    cnr1[i, :, -1] = gen.cnr(data1_k[i], water1, water1)
    for j in range(8):
        cnr2[i, :, j] = gen.cnr(data2_k[i], contrast2[j], water2)
    cnr2[i, :, -1] = gen.cnr(data2_k[i], water2, water2)

np.save(os.path.join(directory, folder1, 'phantom_scan', 'CNR.npy'), cnr1)
np.save(os.path.join(directory, folder2, 'phantom_scan', 'CNR.npy'), cnr2)