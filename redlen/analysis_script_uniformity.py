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

mm = 'M20358_Q20'
for folder in folders:
    for i in np.arange(1, 18):
        a = AnalyzeUniformity(folder, airfolder, test_num=i)
