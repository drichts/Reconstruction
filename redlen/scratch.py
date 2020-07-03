from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
from redlen.uniformity_analysis import AnalyzeUniformity

a1 = AnalyzeUniformity('polyprop_3w_deadtime_32ns', 'airscan_3w_deadtime_32ns')
u1 = a1.non_uniformity((11, 17))
n1 = a1.noise_time[0]
m1 = a1.mean_counts()

u2 = np.mean(u1, axis=1)
m2 = np.divide(np.sqrt(m1), m1)
n2 = n1[:, 0]

# def time_to_counts(t, a):
#     return a*t*m1[12, 0]

def noise_fit(x, a, b):
    c = np.divide(np.sqrt(x*11157.386373737374), x*11157.386373737374)
    return a*c + b*0.09578305164677405

