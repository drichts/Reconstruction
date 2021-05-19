import numpy as np

# These are the frequencies you would plot the mtf values against
# You can enter the freqencies (variable freq) directly if you know those values
bar_size = np.array([1, 0.75, 0.66, 0.5, 0.33, 0.25])  # Bar pattern sizes (in mm)
freq = 1 / (2 * bar_size)  # Corresponding frequencies of the bar patterns (in lp/mm)


def calc_mtf(std_vals, mat1_mean, mat2_mean, mat1_std, mat2_std):
    """
    This function takes the noise values (std) from ROIs that cover both materials in a bar pattern and calculates
    the MTF. The ROIs should all roughly encompass the same number of pixels (as close as reasonable)
    :param std_vals: list, ndarray
            List of the noise values from all the desired bar patterns
    :param mat1_mean: float
            The mean value of an ROI encompassing only 1 of the materials from a bar pattern
    :param mat2_mean: float
            The mean value of an ROI encompassing the other material from a bar pattern
    :param mat1_std: float
            The noise in an ROI encompassing only 1 of the materials from a bar pattern
    :param mat2_std: float
            The noise in an ROI encompassing the other material from a bar pattern
    :return: The MTF values for each of the values given in std_vals
    """

    # Contrast between the mean values of the 2 materials
    contrast = np.abs(mat1_mean - mat2_mean)

    # M' is the measured noise (STD) in each of the bar patterns, uncorrected
    M_prime = std_vals

    # squared noise using ROIs of materials 1 and 2
    # N^2 = (Np^2+Nw^2)/2
    noise_2 = (mat1_std ** 2 + mat2_std ** 2) / 2

    # correct std of bar patterns for noise
    M = []
    for std in M_prime:
        if std ** 2 > noise_2:
            M.append(np.sqrt(std ** 2 - noise_2))
        else:
            M.append(0)
    M = np.array(M)

    # M0 = (CT1-CT2)/2: |Air - Plastic|/2
    M0 = contrast / 2

    # MTF = (pi*sqrt(2)/4)*(M/M0)
    MTF = (np.pi * np.sqrt(2) / 4) * (M / M0)

    return MTF
