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

u_folders = ['NDT_BB4mm', 'NDT_BB2mm',
             'NDT_glass2mm',
             'NDT_steel07mm', 'NDT_steel2mm']
u_airfolder = 'NDT_airscan'
u_directory = r'C:\Users\10376\Documents\Phantom Data\UVic'


def redo_cnr_noise(num, pixels=None):
    for i in np.arange(1, 2):
        # a = AnalyzeUniformity(u_folders[num], u_airfolder, mm='M15691', test_num=i, load_dir=u_directory)
        a = AnalyzeUniformity(folders[num], airfolder, test_num=i)
        a.analyze_cnr_noise(redo=True)
        a.mean_signal_all_pixels(redo=True)


if __name__ == '__main__':

    # for folder in u_folders[1:]:
    #     a1 = AnalyzeUniformity(folder, u_airfolder, mm='M15691', load_dir=u_directory)
    #     #a1.redo_masks(pixels=[8])

    process = [mp.Process(target=redo_cnr_noise, args=(0, ), kwargs={'pixels': [1, 2, 3, 4, 6, 8]}),
               mp.Process(target=redo_cnr_noise, args=(1, ), kwargs={'pixels': [1, 2, 3, 4, 6, 8]}),
               mp.Process(target=redo_cnr_noise, args=(2, ), kwargs={'pixels': [1, 2, 3, 4, 6, 8]}),
               mp.Process(target=redo_cnr_noise, args=(3, ), kwargs={'pixels': [1, 2, 3, 4, 6, 8]})]
    r1 = map(lambda p: p.start(), process)
    r2 = map(lambda p: p.join(), process)
    r1 = list(r1)
    r1 = list(r2)

    process = [mp.Process(target=redo_cnr_noise, args=(4, ), kwargs={'pixels': [1, 2, 3, 4, 6, 8]}),
               mp.Process(target=redo_cnr_noise, args=(5, ), kwargs={'pixels': [1, 2, 3, 4, 6, 8]}),
               mp.Process(target=redo_cnr_noise, args=(6, ), kwargs={'pixels': [1, 2, 3, 4, 6, 8]})]
    r1 = map(lambda p: p.start(), process)
    r2 = map(lambda p: p.join(), process)
    r1 = list(r1)
    r1 = list(r2)
