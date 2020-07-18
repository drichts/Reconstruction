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

titles_all = [['20-30', '30-50', '50-70', '70-90', '90-120', 'EC'],
              ['20-30', '30-40', '40-50', '50-60', '60-70', 'EC'],
              ['20-35', '35-50', '50-65', '65-80', '80-90', 'EC'],
              ['25-35', '35-45', '45-55', '55-65', '65-75', 'EC'],
              ['25-40', '40-55', '55-70', '70-80', '80-95', 'EC'],
              ['30-45', '45-60', '60-75', '75-85', '85-95', 'EC'],
              ['20-30', '30-70', '70-85', '85-100', '100-120', 'EC']]
              #['20-90', '90-100', '100-105', '105-110', '110-120', 'EC']]

a1 = AnalyzeUniformity('multiple_energy_thresholds_3w', 'multiple_energy_thresholds_flatfield_3w', mm='M20358_Q20')
a2 = AnalyzeUniformity('multiple_energy_thresholds_3w', 'multiple_energy_thresholds_flatfield_3w', test_num=2, mm='M20358_Q20')
a3 = AnalyzeUniformity('multiple_energy_thresholds_3w', 'multiple_energy_thresholds_flatfield_3w', test_num=3, mm='M20358_Q20')
a4 = AnalyzeUniformity('multiple_energy_thresholds_3w', 'multiple_energy_thresholds_flatfield_3w', test_num=4, mm='M20358_Q20')
a5 = AnalyzeUniformity('multiple_energy_thresholds_3w', 'multiple_energy_thresholds_flatfield_3w', test_num=5, mm='M20358_Q20')
a6 = AnalyzeUniformity('multiple_energy_thresholds_3w', 'multiple_energy_thresholds_flatfield_3w', test_num=6, mm='M20358_Q20')
a7 = AnalyzeUniformity('multiple_energy_thresholds_3w', 'multiple_energy_thresholds_flatfield_3w', test_num=7, mm='M20358_Q20')

a1.analyze_cnr_noise()
a2.analyze_cnr_noise()
a3.analyze_cnr_noise()
a4.analyze_cnr_noise()
a5.analyze_cnr_noise()
a6.analyze_cnr_noise()
a7.analyze_cnr_noise()

v1 = VisualizeUniformity(a1)
v1.titles = titles_all[0]
v2 = VisualizeUniformity(a2)
v2.titles = titles_all[1]
v3 = VisualizeUniformity(a3)
v3.titles = titles_all[2]
v4 = VisualizeUniformity(a4)
v4.titles = titles_all[3]
v5 = VisualizeUniformity(a5)
v5.titles = titles_all[4]
v6 = VisualizeUniformity(a6)
v6.titles = titles_all[5]
v7 = VisualizeUniformity(a7)
v7.titles = titles_all[6]

v1.blank_vs_time_six_bins(save=True)
v2.blank_vs_time_six_bins(save=True)
v3.blank_vs_time_six_bins(save=True)
v4.blank_vs_time_six_bins(save=True)
v5.blank_vs_time_six_bins(save=True)
v6.blank_vs_time_six_bins(save=True)
v7.blank_vs_time_six_bins(save=True)


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

