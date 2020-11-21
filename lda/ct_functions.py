import numpy as np
from numpy.fft import fft, ifft
import lda.parameters as param


def filtering(projections):
    """
    Applies a ramp filter at low frequencies and the desired high-pass filter

    :param projections: 4D numpy array
                The projection data. Shape: <counters, captures, rows, columns>

    :return: 4D numpy array
                The filtered projection data. Shape: <counters, captures, rows, columns>

    Adapted from:  Kyungsang Kim (2020). 3D Cone beam CT (CBCT) projection backprojection FDK, iterative reconstruction
    Matlab examples (https://www.mathworks.com/matlabcentral/fileexchange/35548-3d-cone-beam-ct-cbct-projection-
    backprojection-fdk-iterative-reconstruction-matlab-examples), MATLAB Central File Exchange. Retrieved May 19, 2020.
    """

    uu, vv = np.meshgrid(param.US, param.VS)  # Create a meshgrid of x, y coordinates of all pixels

    # Correction for each point based on distance from source to the coordinate
    w = param.DSD / np.sqrt(param.DSD**2 + uu**2 + vv**2)

    projections = np.multiply(projections, w)  # Correct each projection angle for detector flatness

    # Find the next highest power of 2 of number of pixels horizontally in the detector multiplied by 2
    filt_len = int(np.max([64, 2**np.ceil(np.log2(2*param.NU))]))

    ramp_kernel = ramp_flat(filt_len)  # Calculate the ramp filter kernel

    filt = filter_array(param.FILTER, ramp_kernel, filt_len)  # Calculate the full filter array

    # Copy the filter nv times (NV = number of pixels vertically)
    filt = np.tile(np.reshape(filt, (1, np.size(filt))), (param.NV, 1))

    # For each projection, filter the data
    for i, counter in enumerate(projections):
        for j, proj in enumerate(counter):

            filt_proj = np.zeros([param.NV, filt_len], dtype='float32')

            # Set proj data into the middle NU rows
            filt_proj[:, int(filt_len/2-param.NU/2):int(filt_len/2+param.NU/2)] = proj
            filt_proj = fft(filt_proj, axis=1)  # Compute the Fourier transform along each column

            filt_proj = filt_proj * filt  # Apply the filter to the Fourier transform of the data
            filt_proj = np.real(ifft(filt_proj, axis=1))  # Get only the real portion of the inverse Fourier transform

            # Apply a correction factor based on the number of projections and system geometry
            proj = filt_proj[:, int(filt_len/2-param.NU/2):int(filt_len/2+param.NU/2)] / 2 /param.PS * \
                   (2*np.pi/param.NUM_PROJ) / 2 * (param.DSD/param.DSO)

            projections[i, j] = proj  # Reassign the unfiltered data as the newly filtered data

    return projections


def ramp_flat(n):
    """
    This function creates the ramp filter array of the correct size based on the projection data

    :param n: int
                The length of the filter based on the data
    :return: 1d array
                The ramp filter of the correct size for the projection data

    Adapted from:  Kyungsang Kim (2020). 3D Cone beam CT (CBCT) projection backprojection FDK, iterative reconstruction
    Matlab examples (https://www.mathworks.com/matlabcentral/fileexchange/35548-3d-cone-beam-ct-cbct-projection-
    backprojection-fdk-iterative-reconstruction-matlab-examples), MATLAB Central File Exchange. Retrieved May 19, 2020.
    """
    nn = np.arange(-n/2, n/2)
    h = np.zeros(np.size(nn), dtype='float32')
    h[int(n/2)] = 0.25  # Set center point (0.0) equal to 1/4
    odd = np.mod(nn, 2) == 1  # odd = False, even = True
    h[odd] = -1 / (np.pi * nn[odd])**2

    return h


def filter_array(filter_type, kernel, order, d=1):
    """
    This function takes the high pass filter type, ramp filter kernel, the order, and cutoff and calculates the filter
    array to apply to the projection data

    :param filter_type: String
                High pass filter type, see Parameters.py for options under the 'filter' variable
    :param kernel: 1D numpy array
                The ramp filter kernel
    :param order: int
                The filter length depending on the data
    :param d: float, default = 1
                Cutoff for the high-pass filter. On the range [0, 1]
    :return: 1D numpy array
                The filter array

    Adapted from:  Kyungsang Kim (2020). 3D Cone beam CT (CBCT) projection backprojection FDK, iterative reconstruction
    Matlab examples (https://www.mathworks.com/matlabcentral/fileexchange/35548-3d-cone-beam-ct-cbct-projection-
    backprojection-fdk-iterative-reconstruction-matlab-examples), MATLAB Central File Exchange. Retrieved May 19, 2020.
    """
    f_kernel = np.abs(fft(kernel))*2
    filt = f_kernel[0:int(order/2+1)]
    w = 2*np.pi*np.arange(len(filt))/order  # Frequency axis up to Nyquist

    if filter_type is 'shepp-logan':
        # Be aware of your d value - do not set d equal to zero
        filt[2:] = filt[2:] * np.sin(w[2:]/(2*d)) / (w[2:]/(2*d))
    elif filter_type is 'cosine':
        filt[2:] = filt[2:] * np.cos(w[2:]/(2*d))
    elif filter_type is 'hamming':
        filt[2:] = filt[2:] * (0.54 + 0.46 * np.cos(w[2:]/d))
    elif filter_type is 'hann':
        filt[2:] = filt[2:] * (1 + np.cos(w[2:]/d)) / 2
    else:
        print(filter_type)
        raise Exception('Filter type not recognized.')

    filt[w > np.pi*d] = 0  # Crop the frequency response
    filt = np.concatenate((filt, np.flip(filt[1:-1])))  # Make the filter symmetric

    return filt
