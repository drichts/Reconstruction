import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from scipy.interpolate import make_interp_spline as spline
from redlen.uniformity_analysis import AnalyzeUniformity
from redlen.visualize import VisualizeUniformity
from redlen.visualize_3windows import Visualize3Windows
from general_functions import load_object

a1 = AnalyzeUniformity('bluebelt_small_1w_deadtime_32ns', 'airscan_1w_deadtime_32ns')
a2 = AnalyzeUniformity('bluebelt_small_3w_deadtime_32ns', 'airscan_3w_deadtime_32ns')
a3 = AnalyzeUniformity('bluebelt_small_8w_deadtime_32ns', 'airscan_8w_deadtime_32ns')

a1.analyze_cnr_noise()
a2.analyze_cnr_noise()
a3.analyze_cnr_noise()

v1 = VisualizeUniformity(a1)
v2 = VisualizeUniformity(a2)
v3 = VisualizeUniformity(a3)

v = Visualize3Windows(a1, a2, a3)

v.plot_cnr_vs_time(save=True)
v.plot_cnr_vs_time(cnr_or_noise=1, save=True)
v1.blank_vs_time_six_bins(save=True)
v2.blank_vs_time_six_bins(save=True)
v3.blank_vs_time_six_bins(save=True)
v1.blank_vs_time_six_bins(cnr_or_noise=1, save=True)
v2.blank_vs_time_six_bins(cnr_or_noise=1, save=True)
v3.blank_vs_time_six_bins(cnr_or_noise=1, save=True)

