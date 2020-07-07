from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
from redlen.uniformity_analysis import AnalyzeUniformity
from redlen.contrast_materials import ContrastMaterial
from general_functions import load_object

# a1 = AnalyzeUniformity('polyprop_3w_deadtime_32ns', 'airscan_3w_deadtime_32ns')
# a1.analyze_cnr_noise()
# a1.non_uniformity((11, 17))
# a1.mean_counts()
# a1.avg_contrast_over_all_frames()
# a1.save_object('Noise_model.pk1')

a1 = load_object(r'C:\Users\10376\Documents\Phantom Data\Uniformity\polyprop_3w_deadtime_32ns\Noise_model.pk1')

b1 = ContrastMaterial('PMMA', 1.18)
c1 = ContrastMaterial('PP', 0.946)

