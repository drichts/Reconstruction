import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from scipy.interpolate import make_interp_spline as spline
from redlen.uniformity_analysis import AnalyzeUniformity
from redlen.uniformity_analysis_add_bins import AddBinsUniformity
from redlen.visualize import VisualizeUniformity
from redlen.visualize_add_bins import AddBinsVisualize
from redlen.visualize_3windows import Visualize3Windows
from general_functions import load_object

titles_mult = [['20-30', '30-50', '50-70', '70-90', '90-120', 'EC'],
              ['20-30', '30-40', '40-50', '50-60', '60-70', 'EC'],
              ['20-35', '35-50', '50-65', '65-80', '80-90', 'EC'],
              ['25-35', '35-45', '45-55', '55-65', '65-75', 'EC'],
              ['25-40', '40-55', '55-70', '70-80', '80-95', 'EC'],
              ['30-45', '45-60', '60-75', '75-85', '85-95', 'EC'],
              ['20-30', '30-70', '70-85', '85-100', '100-120', 'EC']]
              #['20-90', '90-100', '100-105', '105-110', '110-120', 'EC']]

titles_energy_check = [['20-30', '30-50', '50-70', '70-90', '90-120', 'EC'],
                       ['20-50', '50-70', '70-100', '100-110', '110-120', 'EC'],
                       ['20-40', '40-50', '50-70', '70-80', '80-120', 'EC']]

titles_many = [['20-30', '30-50', '50-70', '70-90', '90-120', 'EC'],
               ['20-40', '40-60', '60-80', '80-100', '100-120', 'EC'],
               ['25-45', '45-65', '65-85', '85-105', '105-120', 'EC'],
               ['20-35', '35-55', '55-75', '75-95', '95-120', 'EC'],
               ['20-50', '50-80', '80-100', '100-110', '110-120', 'EC'],
               ['20-30', '30-60', '60-90', '90-100', '100-120', 'EC'],
               ['20-40', '40-70', '70-100', '100-110', '110-120', 'EC'],
               ['20-60', '60-70', '70-80', '80-90', '90-120', 'EC'],
               ['20-70', '70-80', '80-90', '90-100', '100-120', 'EC'],
               ['20-80', '80-90', '90-100', '100-110', '110-120', 'EC'],
               ['20-90', '90-100', '100-105', '105-110', '110-120', 'EC'],
               ['20-30', '30-70', '70-80', '80-90', '90-120', 'EC'],
               ['20-30', '30-80', '80-90', '90-100', '100-120', 'EC'],
               ['20-30', '30-90', '90-100', '100-110', '110-120', 'EC'],
               ['20-40', '40-80', '80-90', '90-100', '100-120', 'EC'],
               ['20-40', '40-90', '90-100', '100-110', '110-120', 'EC'],
               ['20-50', '50-90', '90-100', '100-110', '110-120', 'EC']]

folders = ['many_thresholds_BB4mm', 'many_thresholds_BB2mm', 'many_thresholds_BB1mm',
           'many_thresholds_glass2mm', 'many_thresholds_glass1mm',
           'many_thresholds_steel07mm', 'many_thresholds_steel2mm',
           'many_thresholds_PP']
airfolder = 'many_thresholds_airscan'

folders2 = ['many_thresholds_BB4mm', 'many_thresholds_BB2mm', 'many_thresholds_BB1mm', 'many_thresholds_glass1mm']

folders3 = ['multiple_energy_thresholds_1w', 'multiple_energy_thresholds_3w']
airfolders3 = ['multiple_energy_thresholds_flatfield_1w', 'multiple_energy_thresholds_flatfield_3w']

f = 0
t = 1
b = 1
a1 = AddBinsUniformity(folders[f], airfolder, test_num=t)
# a2 = AddBinsUniformity(folders[7], airfolder)
a1.add_adj_bins([1, 3])
a1.analyze_cnr_noise()

# a2.analyze_cnr_noise()
v = VisualizeUniformity(a1)
v.titles = titles_many[t-1]
for p in [1, 2, 3]:
    v.blank_vs_time_single_bin(b, pixel=p, save=True)
# v = AddBinsVisualize('30-90', a2, AnalyzeUniformity=a1)
# v.plot_comparison(1, 1, save=True)

# a1 = AnalyzeUniformity(folders[6], airfolder)
# a2 = AnalyzeUniformity(folders[7], airfolder)
# a1.analyze_cnr_noise()
# a2.analyze_cnr_noise()
#
# v1 = VisualizeUniformity(a1)
# v2 = VisualizeUniformity(a2)
#
# v1.noise_vs_counts_six_bins(save=True)
# v2.noise_vs_counts_six_bins(save=True)

#v = Visualize3Windows(a1, a2)
#v.plot_cnr_vs_time(cnr_or_noise=1, save=True)
#v.plot_cnr_vs_time(save=True)

# for j, folder in enumerate(folders[1:]):
#     for i in np.arange(1, 18):
#         a1 = AnalyzeUniformity(folder, airfolder, test_num=i)
#         a1.analyze_cnr_noise()
#         v1 = VisualizeUniformity(a1)
#         v1.titles = titles_many[i-1]
#         for p in a1.pxp[0:6]:
#             v1.blank_vs_time_six_bins(cnr_or_noise=0, pixel=p, save=True)
#             v1.blank_vs_time_six_bins(cnr_or_noise=1, pixel=p, save=True)
        # if i == 1:
        #     # if folder is folders[5]:
        #     #     v1.blank_vs_pixels_six_bins(time=2, y_lim=80, save=True)
        #     #     v1.blank_vs_pixels_six_bins(time=5, y_lim=80, save=True)
        #     #     v1.blank_vs_pixels_six_bins(time=10, y_lim=80, save=True)
        #     #     v1.blank_vs_pixels_six_bins(time=25, y_lim=80, save=True)
        #     # else:
        #     v1.blank_vs_pixels_six_bins(cnr_or_noise=0, time=2, save=True)
        #     v1.blank_vs_pixels_six_bins(cnr_or_noise=0, time=5, save=True)
        #     v1.blank_vs_pixels_six_bins(cnr_or_noise=0, time=10, save=True)
        #     v1.blank_vs_pixels_six_bins(cnr_or_noise=0, time=25, save=True)

