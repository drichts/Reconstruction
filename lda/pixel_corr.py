import numpy as np
from scipy import signal


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


def pixel_corr(data, window='blackman'):
    """
    Jericho's correction method for correcting the variable pixel response of the detector
    :param data:
    :param window:
    :return:
    """

    # Crop away the top of the image with only air
    good_data = np.mean(data[:, 9:, :, 6], 0)

    outliers = []

    # Find the mean of the data along the angle direction
    ref = np.mean(good_data, axis=0)

    # Now I'll discard the three biggest outliers and take that average.
    mins = np.argmax(np.abs(good_data - ref), axis=0)
    nn = 14

    for jj in range(nn):
        ref = np.nanmean(good_data, 0)
        mins = np.argmax(np.abs(good_data - ref), 0)
        outliers.append(mins)
        for ii in range(len(mins)):
            good_data[mins[ii], ii] = ref[ii]

    for jj in range(nn):
        for ii in range(len(mins)):
            good_data[outliers[jj][ii], ii] = np.NaN

    real_refs = np.nanmean(good_data, 0)

    smoothed = smooth(real_refs, window_len=10, window=window)

    smoothed3 = smoothed[4:-5]

    w = 0.1  # Cut-off frequency of the filter
    b, a = signal.butter(5, w, 'low')   # Numerator (b) and denominator (a) for Butterworth filter
    output = signal.filtfilt(b, a, real_refs)  # Apply the filter to the data

    smoothed3[25:-25] = output[25:-25]  # Replace filtered data in the
    correction_array3 = np.mean(data[:, 10:, :, 6], 0)/smoothed3
    X2 = (data[:, 10:, :, 6]/correction_array3).transpose(1, 2, 0)

    X2[X2 < -0.5] = 0
    image_result2 = X2.copy()

    float_array = np.float32(10*image_result2.transpose(2, 0, 1))
    one_slice = float_array
    tiled = np.pad(one_slice, [(0, 0), (25, 25), (0, 0)])
    tiled[:, :, :26] = 0
    tiled[:, :, -26:] = 0

    return tiled
