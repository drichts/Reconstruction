import numpy as np
import matplotlib.pyplot as plt
import os
from redlen.uniformity_analysis import AnalyzeUniformity
import mask_functions as grm
import _pickle as pickle

folder = 'multiple_energy_thresholds_1w'
airfolder = 'multiple_energy_thresholds_flatfield_1w'

mm = 'M20358_Q20'

for i in np.arange(1, 8):
    a = AnalyzeUniformity(folder, airfolder, test_num=i, mm=mm)
