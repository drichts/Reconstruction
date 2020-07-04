from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
from redlen.uniformity_analysis import AnalyzeUniformity
from general_functions import load_object

# a1 = AnalyzeUniformity('polyprop_3w_deadtime_32ns', 'airscan_3w_deadtime_32ns')
# a1.analyze_cnr_noise()
# a1.non_uniformity((11, 17))
# a1.mean_counts()
# a1.save_object('Noise_model.pk1')

a1 = load_object(r'C:\Users\10376\Documents\Phantom Data\Uniformity\polyprop_3w_deadtime_32ns\Noise_model.pk1')

#%%
mods = a1.stitch_a0a1()
air_mods = a1.air_data.stitch_a0a1()

def noise_test(x, a, b, mca, rua):
    c = np.divide(np.sqrt(x * mca), x * mca)
    return a*c + rua*b


def non_uniform(x, a, b, rua):
    return a*np.exp(b*x) + rua


u1 = a1.rel_uniformity
n1 = a1.noise_time[0]
m1 = a1.counts

m2 = np.divide(np.sqrt(m1), m1)
n2 = n1[:, 0]
u2 = n2 - m2

frames = a1.frames

bin_num = 3

mc = m1[bin_num, 0]
ru = u1[bin_num, 0]
print(mc)
print(ru)
coeffs, covar = curve_fit(lambda x, a, b: noise_test(x, a, b, mc, ru), frames, n2[12])
ypts = noise_test(frames, coeffs[0], coeffs[1], mc, ru)

#c2, blah = curve_fit(lambda x, a, b: non_uniform(x, a, b, ru), frames, u1[bin_num])

fig = plt.figure(figsize=(6, 6))
plt.plot(frames, n2[bin_num])
plt.plot(frames, m2[bin_num])
plt.plot(frames, u2[bin_num])
#plt.plot(frames, ypts)
#plt.plot(frames, non_uniform(frames, 1, 1, ru))
plt.legend(['Actual Noise', 'Theoretical', 'Non-uniformity', 'Fit', 'NU Fit'])
# for i in np.arange(13):
#     plt.plot(frames, n2[i])
# plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'])






