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

# for folder in folders:
#     #for i in np.arange(1, 18):
#     i = 1
#     a1 = AnalyzeUniformity(folder, airfolder, test_num=i)
#     a1.analyze_cnr_noise()
#     v1 = VisualizeUniformity(a1)
#     v1.titles = titles_many[i-1]
#     if folder is folders[5]:
#         v1.blank_vs_pixels_six_bins(time=5, y_lim=80, save=True)
#     else:
#         v1.blank_vs_pixels_six_bins(time=5, save=True)

#%%

a1 = AnalyzeUniformity(folders[5], airfolder)
v1 = VisualizeUniformity(a1)
v1.pixel_images(save=True)

# a1 = AnalyzeUniformity('many_thresholds_BB4mm', 'many_thresholds_airscan')
# a2 = AnalyzeUniformity('many_thresholds_BB2mm', 'many_thresholds_airscan')
# a3 = AnalyzeUniformity('many_thresholds_BB1mm', 'many_thresholds_airscan')
# a4 = AnalyzeUniformity('many_thresholds_glass2mm', 'many_thresholds_airscan')
# a5 = AnalyzeUniformity('many_thresholds_glass1mm', 'many_thresholds_airscan')
# a6 = AnalyzeUniformity('many_thresholds_steel2mm', 'many_thresholds_airscan')
# a7 = AnalyzeUniformity('many_thresholds_steel07mm', 'many_thresholds_airscan')
#
# a8 = AnalyzeUniformity('many_thresholds_PP', 'many_thresholds_airscan')
# a9 = AnalyzeUniformity('energy_bin_check_PP', 'energy_bin_check_airscan')
# a10 = AnalyzeUniformity('multiple_energy_thresholds_1w', 'multiple_energy_thresholds_flatfield_1w')
#
# a11 = AnalyzeUniformity('energy_bin_check_PP', 'energy_bin_check_airscan', test_num=2)
# a12 = AnalyzeUniformity('energy_bin_check_PP', 'energy_bin_check_airscan', test_num=3)
#
# a1.analyze_cnr_noise()
# a2.analyze_cnr_noise()
# a3.analyze_cnr_noise()
# a4.analyze_cnr_noise()
# a5.analyze_cnr_noise()
# a6.analyze_cnr_noise()
# a7.analyze_cnr_noise()
# a8.analyze_cnr_noise()
# a9.analyze_cnr_noise()
# a10.analyze_cnr_noise()
#
# a11.analyze_cnr_noise()
# a12.analyze_cnr_noise()

# v1 = VisualizeUniformity(a1)
# v1.titles = titles_all[0]
# v2 = VisualizeUniformity(a2)
# v2.titles = titles_all[0]
# v3 = VisualizeUniformity(a3)
# v3.titles = titles_all[0]
# v4 = VisualizeUniformity(a4)
# v4.titles = titles_all[0]
# v5 = VisualizeUniformity(a5)
# v5.titles = titles_all[0]
# v6 = VisualizeUniformity(a6)
# v6.titles = titles_all[0]
# v7 = VisualizeUniformity(a7)
# v7.titles = titles_all[0]
# v8 = VisualizeUniformity(a8)
# v8.titles = titles_all[0]
# v9 = VisualizeUniformity(a9)
# v9.titles = titles_all[0]
# v10 = VisualizeUniformity(a10)
# v10.titles = titles_all[0]
#
# v11 = VisualizeUniformity(a11)
# v11.titles = titles_all[0]
# v12 = VisualizeUniformity(a12)
# v12.titles = titles_all[0]

# v1.blank_vs_time_six_bins()
# v2.blank_vs_time_six_bins()
# v3.blank_vs_time_six_bins()
# v4.blank_vs_time_six_bins()
# v5.blank_vs_time_six_bins()
# v6.blank_vs_time_six_bins()
# v7.blank_vs_time_six_bins()
# v8.blank_vs_time_six_bins()
# v9.blank_vs_time_six_bins()
# v10.blank_vs_time_six_bins()
#v11.blank_vs_time_six_bins()
#v12.blank_vs_time_six_bins()


# v3 = VisualizeUniformity(a3)
#
# v = Visualize3Windows(a1, a2, a3)
#
# v.plot_cnr_vs_time(save=True)
# v.plot_cnr_vs_time(cnr_or_noise=1, save=True)
# v1.blank_vs_time_six_bins()
# v2.plot_bin(1)
# v2.blank_vs_time_six_bins(save=True)
# v3.blank_vs_time_six_bins(save=True)
# v1.blank_vs_time_six_bins(cnr_or_noise=1, save=True)
# v2.blank_vs_time_six_bins(cnr_or_noise=1, save=True)
# v3.blank_vs_time_six_bins(cnr_or_noise=1, save=True)

