from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
from redlen.uniformity_analysis import AnalyzeUniformity
from redlen.spectrum_analysis import AnalyzeSpectrum
from redlen.contrast_materials import ContrastMaterial
from general_functions import load_object

# a1 = AnalyzeUniformity('polyprop_1w_deadtime_32ns', 'airscan_1w_deadtime_32ns')
# a1.analyze_cnr_noise()
# a1.non_uniformity((11, 17))
# a1.mean_counts()
# a1.avg_contrast_over_all_frames()
# a1.save_object('Noise_model.pk1')


def find_nearest(array, value):
    """

    :param array:
    :param value:
    :return:
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


def get_bin_spectra(low_e, high_e, spectrum_obj):

    low_e, low_i = find_nearest(spectrum_obj.energy, low_e)
    high_e, high_i = find_nearest(spectrum_obj.energy, high_e)
    high_i += 1

    return spectrum_obj.energy[low_i:high_i], spectrum_obj.spectrum[low_i:high_i]/np.sum(spectrum_obj.spectrum[low_i:high_i])


def calc_weighted_attenuation(energy, spectrum, contrast_obj):
    atten_coeffs = contrast_obj.interp_val(energy)
    atten_coeffs = np.multiply(atten_coeffs, spectrum)
    atten_coeff = np.sum(atten_coeffs)

    return atten_coeff


a1 = load_object(r'C:\Users\10376\Documents\Phantom Data\Uniformity\polyprop_1w_deadtime_32ns\Noise_model.pk1')
s1 = load_object(r'C:\Users\10376\Documents\Phantom Data\Spectrum\2ma_1w\Run0Spectrum.pk1')

b1 = ContrastMaterial('PMMA', 1.18)
c1 = ContrastMaterial('PP', 0.946)



bin_num = 8
low_e, high_e = 50, 70
en, spec = get_bin_spectra(low_e, high_e, s1)

b1_att = calc_weighted_attenuation(en, spec, b1)
c1_att = calc_weighted_attenuation(en, spec, c1)

con_thi = 0.4
b1_signal = b1_att*b1.density*2
c1_signal = c1_att*c1.density*con_thi + b1_att*b1.density*(2-con_thi)

print(f'Calculated BG Signal: {b1_signal}')
print(f'Actual BG Signal: {np.mean(a1.bg_signal[bin_num, 0])}')
print(f'Actual BG Signal Error: {np.mean(a1.bg_signal[bin_num, 1])}')

print(f'Calculated Contrast Signal: {c1_signal}')
print(f'Actual Contrast Signal: {np.mean(a1.signal[bin_num, 0])}')
print(f'Actual Contrast Error: {np.mean(a1.signal[bin_num, 1])}')
