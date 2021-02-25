import numpy as np
from scipy import signal
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt


def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the beginning and end part of the output signal.
    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.
    output:
        the smoothed signal
    example:
    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    see also:
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """
    if x.ndim != 1:
        raise(ValueError, "smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise(ValueError, "Input vector needs to be bigger than window size.")
    if window_len < 3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise(ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    s = np.r_[x[window_len-1:0:-1], x, x[-2:-window_len-1:-1]]

    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.'+window+'(window_len)')

    y = np.convolve(w/w.sum(), s, mode='valid')
    return y


def pixel_corr(data, num_bins=7, type='middle' ,window='blackman'):
    """
    Jericho's correction method for correcting the variable pixel response of the detector
    :param data: ndarray
            Shape: (angles, rows, columns, bins) The data the correction matrix is calculated from (should be of a
            uniform object)
    :param num_bins: int, optional
            The number of bins in the data array. Defaults to 7
    :param window: str, optional
            The type of window. Types: 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
    :return: ndarray
            The correction matrix to multiply the data by. Shape: (rows, columns, bins)
    """
    # Go through each of the bins
    for bin_num in np.arange(num_bins):

        # Crop away the top of the image with only air
        good_data = np.nanmean(data[:, :, :, bin_num], axis=0)

        outliers = []

        # Find the mean of the data along the angle direction
        ref = np.nanmean(good_data, axis=0)

        # Now I'll discard the three biggest outliers and take that average.
        mins = np.argmax(np.abs(good_data - ref), axis=0)
        nn = 14

        for jj in range(nn):
            ref = np.nanmean(good_data, axis=0)
            mins = np.argmax(np.abs(good_data - ref), axis=0)
            outliers.append(mins)
            for ii in range(len(mins)):
                good_data[mins[ii], ii] = ref[ii]

        for jj in range(nn):
            for ii in range(len(mins)):
                good_data[outliers[jj][ii], ii] = np.nan

        # Mask invalid data, i.e. inf, nan, -inf, etc when taking the mean
        good_data = np.ma.masked_invalid(good_data)
        real_refs = np.mean(good_data, axis=0)

        smoothed = smooth(real_refs, window_len=10, window=window)

        smoothed3 = smoothed[4:-5]

        w = 0.1  # Cut-off frequency of the filter
        b, a = signal.butter(5, w, 'low')   # Numerator (b) and denominator (a) for Butterworth filter
        output = signal.filtfilt(b, a, real_refs)  # Apply the filter to the data

        smoothed3[25:-25] = output[25:-25]  # Replace filtered data in the
        correction_array = np.nanmean(data[:, :, :, bin_num], axis=0)/smoothed3
        new_data = (data[:, :, :, bin_num] / correction_array).transpose(1, 2, 0)

        new_data[new_data < -0.5] = 0
        image = new_data.copy()

        float_array = np.float32(10 * image.transpose(2, 0, 1))

        data[:, :, :, bin_num] = float_array

        data[:, :, :25, bin_num] = 0
        data[:, :, -25:, bin_num] = 0

    return data
