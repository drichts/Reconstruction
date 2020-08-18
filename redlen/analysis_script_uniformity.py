import numpy as np
import matplotlib.pyplot as plt
import os
from redlen.uniformity_analysis import AnalyzeUniformity
import mask_functions as grm
import _pickle as pickle
from glob import glob
from scipy.io import loadmat
from pathos.helpers import mp
import time

folders = ['many_thresholds_BB4mm', 'many_thresholds_BB2mm',
           'many_thresholds_glass2mm', 'many_thresholds_glass1mm',
           'many_thresholds_steel07mm', 'many_thresholds_steel2mm',
           'many_thresholds_PP']
airfolder = 'many_thresholds_airscan'


def redo_cnr_noise(num):
    for i in np.arange(1, 18):
        a = AnalyzeUniformity(folders[num], airfolder, test_num=i)
        a.analyze_cnr_noise(redo=True)


if __name__ == '__main__':

    # for folder in folders:
    #     a1 = AnalyzeUniformity(folder, airfolder)
    #     a1.redo_masks(pixels=[6, 8, 12])

    process = [mp.Process(target=redo_cnr_noise, args=(0, )),
               mp.Process(target=redo_cnr_noise, args=(1, )),
               mp.Process(target=redo_cnr_noise, args=(2, )),
               mp.Process(target=redo_cnr_noise, args=(3, ))]
    r1 = map(lambda p: p.start(), process)
    r2 = map(lambda p: p.join(), process)
    r1 = list(r1)
    r1 = list(r2)

    process = [mp.Process(target=redo_cnr_noise, args=(4, )),
               mp.Process(target=redo_cnr_noise, args=(5, )),
               mp.Process(target=redo_cnr_noise, args=(6, ))]
    r1 = map(lambda p: p.start(), process)
    r2 = map(lambda p: p.join(), process)
    r1 = list(r1)
    r1 = list(r2)
