import numpy as np
from general_functions import correct_dead_pixels


def gi(lam1, lam2, I1, I2, t0):
    """
    This function calculates gi according to equation 4 in the paper "Non-uniformity correction for MARS
    photon-counting detectors"
    :param lam1: int, float, ndarray
            Mean counts in a pixel (or pixel matrix) from flux I1 over M frames
    :param lam2: int, float, ndarray
            Mean counts in a pixel (or pixel matrix) from flux I2 over M frames
    :param I1: int, float
            Flux 1 (in uA or mA, typically)
    :param I2: int, float
            Flux 2 (in uA or mA, typically)
    :param t0: int, float
            The exposure time per frame
    :return: float, ndarray
            gi, the linear gain coefficient of a single pixel or pixel array
    """
    return (lam2 - lam1) / ((I2 - I1) * t0)


def ki(lam1, lam2, I1, I2, t0):
    """
    This function calculates ki according to equation 5 in the paper "Non-uniformity correction for MARS
    photon-counting detectors"
    :param lam1: int, float, ndarray
            Mean counts in a pixel (or pixel matrix) from flux I1 over M frames
    :param lam2: int, float, ndarray
            Mean counts in a pixel (or pixel matrix) from flux I2 over M frames
    :param I1: int, float
            Flux 1 (in uA or mA, typically)
    :param I2: int, float
            Flux 2 (in uA or mA, typically)
    :param t0: int, float
            The exposure time per frame
    :return: float, ndarray
            ki, the gain correction ratio per pixel g_avg/gi
    """

    # Obtain gi for the pixel array
    g_i = gi(lam1, lam2, I1, I2, t0)

    return np.nanmean(g_i)/g_i


def di(lam1, lam2, I1, I2, t0):
    """
    This function calculates di according to equation 4 in the paper "Non-uniformity correction for MARS
    photon-counting detectors"
    :param lam1: int, float, ndarray
            Mean counts in a pixel (or pixel matrix) from flux I1 over M frames
    :param lam2: int, float, ndarray
            Mean counts in a pixel (or pixel matrix) from flux I2 over M frames
    :param I1: int, float
            Flux 1 (in uA or mA, typically)
    :param I2: int, float
            Flux 2 (in uA or mA, typically)
    :param t0: int, float
            The exposure time per frame
    :return: float, ndarray
            di, the nonlinear deviation from the ideal count rate of a single pixel or pixel array
    """
    return (lam1 * I2 - lam2 * I1) / ((I2 - I1) * t0)


def linear_range_corr(air1, air2, I1, I2, t0_air, t0_data, data, dpm):
    """
    If data is in the linear range of the detector, this will correct the data given for non-uniformities. I1 and I2
    can be any flux in the linear range of the detector
    :param air1: ndarray
            The open beam count data for the first flux rate (I1) in the linear range with the same bin thresholds that
            your data has, and the same time per frames that the data has. Must have at least 500 frames of data of
            length t0
            Shape: <frames, rows, columns, bin>
    :param air2: ndarray
            The open beam count data for the second flux rate (I2) in the linear range with the same bin thresholds that
            your data has, and the same time per frames that the data has. Must have at least 500 frames of data of
            length t0
            Shape: <frames, rows, columns, bin>
     :param I1: int, float
            Flux 1 (in uA or mA, typically)
    :param I2: int, float
            Flux 2 (in uA or mA, typically)
    :param t0_air: int, float
            The exposure time per frame for the correction scans
    :param t0_data: int, float
            The exposure time per frame for the data
    :param data: ndarray
            The data to correct for non-uniformities.
            Shape: <frames, rows, columns, bins>
    :param dpm: ndarray
            The dead pixel matrix to correct for dead pixels in the data before correction. 1's everywhere except for
            dead pixels, which are set to nan
            Shape: <rows, columns>
    :return: ndarray
            The data, corrected for pixel non-uniformities
    """

    # Put bin axes second <frames, bins, rows, columns>
    air1 = np.transpose(air1, axes=[0, 3, 1, 2])
    air2 = np.transpose(air2, axes=[0, 3, 1, 2])
    data = np.transpose(data, axes=[0, 3, 1, 2])

    # Set all dead pixels to nan in air1, air2, and data
    air1 *= dpm
    air2 *= dpm
    data *= dpm

    # Calculate the average value in each pixel (and each bin) over all the frames in air1 and air2
    # nanmean will overlook pixels with a single nan in one frame, but dead pixel nans will be carried through
    lam1 = np.nanmean(air1, axis=0)
    lam2 = np.nanmean(air2, axis=0)

    # Calculate k and d for each pixel in each bin
    k = ki(lam1, lam2, I1, I2, t0_air)
    d = di(lam1, lam2, I1, I2, t0_air)

    # Calculate the mean pixel values across all frames of the data
    data_mean = np.nanmean(data, axis=0)

    # Calculate g*I*t0
    gIt0 = (data_mean - d*t0_data) * k

    # Unbias the data
    data = data - data_mean

    # Variance correction
    data *= np.sqrt(gIt0/data_mean)

    # Mean correction
    data += gIt0

    return np.transpose(data, axes=(0, 2, 3, 1))


def non_linear_corr(data, lookup_table, dpm):
    """
    This function performs the non-uniformity correction when the flux used is not in the linear range of the
    detector
    :param data: ndarray
            The data to correct for non-uniformities.
            Shape: <frames, rows, columns, bins>
    :param lookup_table: ndarray
            The table containing the mean pixel values and gIt0 values from the standard beam flux points
            Shape: <Flux, rows, columns, mean/gIt0 values (2)>
    :param dpm: ndarray
            The dead pixel matrix to correct for dead pixels in the data before correction. 1's everywhere except for
            dead pixels, which are set to nan
            Shape: <rows, columns>
    :return:
    """
    # Set all dead pixels to nan in data
    data *= dpm

    # Calculate the mean for each pixel across all frames
    data_mean = np.nanmean(data, axis=0)

    # Check which array in the lookup table minimizes the difference between data_mean and the expectation value

    # Set correction based on the array as a whole
    diff = np.sum(np.abs(data_mean - lookup_table[0, :, :, 0]))
    lookup_idx = 0
    for idx, flux_table in enumerate(lookup_table):
        temp_diff = np.sum(np.abs(data_mean - flux_table[:, :, 0]))
        if temp_diff < diff:
            diff = temp_diff
            lookup_idx = 0
    corr_array = lookup_table[lookup_idx]

    # Set correction based per pixel
    # data_shape = np.shape(data)[1:-1]  # row, column
    # corr_array = lookup_table[0]
    # for flux in range(len(lookup_table[1:])):
    #     for i in range(data_shape[0]):
    #         for j in range(data_shape[1]):
    #             if np.abs(data_mean[i, j] - lookup_table[flux, i, j, 0]) < corr_array[i, j, 0]:
    #                 corr_array[i, j] = lookup_table[flux, i, j]

    # Apply the correction
    gIt0 = corr_array[:, :, 1]
    lam = corr_array[:, :, 0]

    data = np.sqrt(gIt0/lam) * (data - lam) + gIt0

    return data
