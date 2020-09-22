import numpy as np
from redlen.uniformity_analysis import AnalyzeUniformity
from redlen.visualize import VisualizeUniformity
from pathos.helpers import mp
import matplotlib.pyplot as plt

def run():
    a1 = AnalyzeUniformity('many_thresholds_BB1mm', 'many_thresholds_airscan', test_num=1)
    a2 = AnalyzeUniformity('many_thresholds_BB1mm', 'many_thresholds_airscan', test_num=2)

    a1.analyze_cnr_noise()
    a2.analyze_cnr_noise()

    v1 = VisualizeUniformity(a1)
    v2 = VisualizeUniformity(a2)

    # process = [mp.Process(target=a1.analyze_cnr_noise, kwargs={'redo': True}),
    #            mp.Process(target=a2.analyze_cnr_noise,  kwargs={'redo': True})]

    process = [mp.Process(target=v2.blank_vs_time_six_bins),
               mp.Process(target=v1.blank_vs_time_six_bins)]
    r1 = map(lambda p: p.start(), process)
    r2 = map(lambda p: p.join(), process)
    r1 = list(r1)
    r1 = list(r2)


def check_mask(num, pixel):
    folders = ['many_thresholds_BB4mm', 'many_thresholds_BB2mm',
               'many_thresholds_glass2mm', 'many_thresholds_glass1mm',
               'many_thresholds_steel07mm', 'many_thresholds_steel2mm',
               'many_thresholds_PP']
    airfolder = 'many_thresholds_airscan'

    a1 = AnalyzeUniformity(folders[num], airfolder)
    data = np.load(a1.data_a0)
    airdata = np.load(a1.air_data.data_a0)

    if pixel == 1:
        data = np.sum(data[12], axis=0)
        airdata = np.sum(airdata[12], axis=0)
    else:
        data = np.sum(a1.sumpxp(data, pixel)[12], axis=0)
        airdata = np.sum(a1.sumpxp(airdata, pixel)[12], axis=0)

    corr = a1.intensity_correction(data, airdata)

    plt.imshow(corr, vmin=0.4, vmax=0.6)
    #plt.imshow(a1.masks[np.squeeze(np.argwhere(a1.pxp == pixel))], alpha=0.5)
    #plt.imshow(a1.bg[np.squeeze(np.argwhere(a1.pxp == pixel))], alpha=0.5)
    plt.show()


if __name__ == '__main__':
    check_mask(6, 1)
    #run()