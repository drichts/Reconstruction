import os
import numpy as np
from glob import glob


def build_lookup_table(directory, currents, t0, filtration, dpm):
    """
    This function will build the lookup table for a specific set of tube currents, frame time t0, and filtration
    :param directory: str
            The full path to the folder containing all the data files for the different current values.
            The folders contained inside should be named xxxx_current_{current}mA_xxxxx
    :param currents: list, ndarray
            List of the tube currents used, as floats or ints
    :param t0: int, float
            The exposure time per frame
    :param filtration: str
            The filtration type and thickness used, e.g. 'Al_2mm'
    :param dpm: ndarray
            The dead pixel matrix to correct for dead pixels in the data before correction. 1's everywhere except for
            dead pixels, which are set to nan
            Shape: <rows, columns>
    :return:
    """

    # The lookup table
    lookup_table = np.zeros((len(currents), 24, 576, 2))

    # Go through each current value
    for idx, current in enumerate(currents):
        folder = glob(os.path.join(directory, f'*_current_{current}mA*'))[0]
        data = np.load(os.path.join(folder, 'Data', 'data.npy'))

        # Get rid of deadpixels
        data *= dpm

        lookup_table[idx, :, :, 0] = np.nanmean(data, axis=0)
        lookup_table[idx, :, :, 1] = np.nanvar(data, axis=0)

    save_folder_file = rf'D:\OneDrive - University of Victoria\Research\LDA Data\Lookup Tables\Current' \
                  rf'{current[0]}-{current[-1]}_{filtration}_time{t0}.npy'

    np.save(save_folder_file, lookup_table)
