import numpy as np
import matplotlib.pyplot as plt
import os
from redlen.uniformity_analysis import AnalyzeUniformity
import mask_functions as grm
import _pickle as pickle
from glob import glob
from scipy.io import loadmat

folders = ['many_thresholds_BB4mm', 'many_thresholds_BB2mm', 'many_thresholds_BB1mm',
           'many_thresholds_glass2mm', 'many_thresholds_glass1mm',
           'many_thresholds_steel07mm', 'many_thresholds_steel2mm',
           'many_thresholds_PP']
airfolder = 'many_thresholds_airscan'
#folder = 'energy_bin_check_PP'
#airfolder = 'energy_bin_check_airscan'
folders2 = ['many_thresholds_BB4mm', 'many_thresholds_BB2mm', 'many_thresholds_BB1mm', 'many_thresholds_glass1mm']

mm = 'M20358_Q20'
#for folder in folders:
#for i in np.arange(1, 18):
#for folder in folders2:
for i in np.arange(1, 18):
    a = AnalyzeUniformity(folder, airfolder, test_num=i)
    a.analyze_cnr_noise(redo=True)
# x = a.stitch_a0a1()
# y = a.air_data.stitch_a0a1()
# corr = -1*np.log(x/y)
#
# plt.imshow(np.sum(corr[12], axis=0))
