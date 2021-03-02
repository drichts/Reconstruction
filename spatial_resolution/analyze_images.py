import os
import numpy as np
import matplotlib.pyplot as plt
import mask_functions as msk
from lda.analysis_LDA import ReconLDA


directory = '/home/knoll/LDAData'
folder = '/home/knoll/LDAData/Stationary_kV/kV_QC3_40kVp_5mA_stationary/Data/data.npy'


class MTFVals(ReconLDA):

    def __init__(self, folder, duration, num_patterns, airscan_time=60, reanalyze=True, directory=directory,
                 sub_folder='phantom_scan', air_folder='airscan_60s', dark_folder='darkscan_60s'):
        super().__init__(folder, duration, airscan_time=airscan_time, reanalyze=reanalyze, directory=directory,
                         sub_folder=sub_folder, air_folder=air_folder, dark_folder=dark_folder)

        self.num_patterns = num_patterns

        # Load just the EC bin, that's all we want for spatial resolution
        self.mtf_data = np.load(self.corr_data)[:, :, 6]

        self.get_ROI_vals()

    def get_ROI_vals(self):
        """
        Get the ROIs for the phantom material, background material, and the various patterns
        :return:
        """

        # Get the phantom material
        phantom = msk.square_ROI(self.mtf_data)

        # Get the background material
        background = msk.square_ROI(self.mtf_data)

        # Get the pattern rois
        std_patt = np.zeros(self.num_patterns)
        for i in range(self.num_patterns):
            mask = msk.square_ROI(self.mtf_data)
            std_patt[i] = np.nanstd(mask*self.mtf_data)

        contrast = np.abs(np.nanmean(self.mtf_data*phantom) - np.nanmean(self.mtf_data*background))
        p_noise = np.nanstd(self.mtf_data*phantom)
        b_noise = np.nanstd(self.mtf_data*background)

        np.savez(os.path.join(self.folder, 'mtf_contrast_vals.npz'), {'contrast': contrast, 'std_1': p_noise,
                                                                      'std_2': b_noise})
        np.save(os.path.join(self.folder, 'mtf_pattern_std.npy'), std_patt)
