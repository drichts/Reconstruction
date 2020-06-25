import numpy as np
import scipy.optimize as optimize
import matplotlib.pyplot as plt

# Z eff value of water
Zw = 7.733
Zw2 = Zw*Zw
Zw3 = Zw*Zw*Zw
Zw4 = Zw*Zw*Zw*Zw
atten_folder = 'D:/Research/Attenuation Data/NPY Attenuation/'
spectra_folder = 'D:/Research/Attenuation Data/NPY Spectra/'

# Matrices for the scattering data based on Z (rows) and E (columns)
example_file = np.load(atten_folder + 'Z04.npy')
Z_values = np.arange(4, 31, 1)
num_rows = len(Z_values)
num_cols = len(example_file)

# Initialize the scattering and photoelectric matrices
scatter_matrix = np.empty([num_rows, num_cols])
PE_matrix = np.empty([num_rows, num_cols])

# Load the beam energies and weights
spectra_energies = example_file[:, 0]
wts_low = np.load(spectra_folder + '40kVp_weights.npy')
wts_high = np.load(spectra_folder + '80kVp_weights.npy')

num_energies = len(spectra_energies)

# Populate the matrices with the correct data
for z in Z_values:

    # Load the file of a specific set of Z attenuation data
    if z < 10:
        data = np.load(atten_folder+'Z0' + str(z) + '.npy')
    else:
        data = np.load(atten_folder+'Z' + str(z) + '.npy')

    scatter_matrix[z-4] = np.add(data[:, 1], data[:, 2])
    PE_matrix[z-4] = data[:, 3]

# Normalize the scatter and PE values to F and G
atomic_mass = np.array([9.01218, 10.81, 12.011, 14.0067, 15.9994, 18.998403, 20.179, 22.98977, 24.305, 26.98154,
                        28.0855, 30.97376, 32.06, 35.453, 39.948, 39.0983, 40.08, 44.9559, 47.90, 50.9415, 51.996,
                        54.9380, 55.847, 58.9332, 58.70, 63.546, 65.38])

# Calculate the scatter correction for each Z-value
scatter_corr = np.divide(atomic_mass, Z_values)
PE_corr = np.divide(atomic_mass, np.power(Z_values, 5))

for i in np.arange(len(scatter_matrix[0])):
    scatter_matrix[:, i] = np.multiply(scatter_corr, scatter_matrix[:, i])
    PE_matrix[:, i] = np.multiply(PE_corr, PE_matrix[:, i])

#%%

# Create the matrices that will hold the fits for the photoelectric and scattering contributions
G = np.empty([num_energies, 3])
F = np.empty([num_energies, 5])
for i in np.arange(num_energies):

    # Each entry will hold the coefficients for the fit for a specific energy, whose entry [i] will correspond to
    # the entry [i] in the fit matrix
    G[i] = np.polyfit(Z_values, scatter_matrix[:, i], 2)
    F[i] = np.polyfit(Z_values, PE_matrix[:, i], 4)

# Calculate the Zw components of Eq. (3) and (4) from the paper (without the ratio term multiplied in)
FG_Zw1 = 0
FG_Zw2 = 0

# Sum over all i's
for i in np.arange(num_energies):

    F_G_term = (Zw4*(Zw4*F[i, 0] + Zw3*F[i, 1] + Zw2*F[i, 2] + Zw*F[i, 3] + F[i, 4]) + Zw2*G[i, 0] + Zw*G[i, 1] + G[i, 2])

    FG_Zw1 += wts_low[i]*F_G_term

    FG_Zw2 += wts_high[i]*F_G_term

num = 10
plt.plot(Z_values, PE_matrix[:, num])
plt.show()
plt.pause(3)
plt.close()

zpts = np.linspace(4, 30, 100)
fs = np.empty(len(zpts))
for i, z in enumerate(zpts):
    fs[i] = z*z*z*z*F[num, 0] + z*z*z*F[num, 1] + z*z*F[num, 2] + z*F[num, 3] + F[num, 4]
plt.plot(zpts, fs)
plt.show()
plt.pause(3)
plt.close()

#%%

def ratios_ROI(num_ROIs, folder='', filename=''):
    """
    Function that calculates the ratio between the linear attenuation coefficients of the material and water for each of
    the saved ROI
    :param folder: folder where the images matrices live
    :param filename: filename of the specific image (must be .npy)
    :param num_ROIs: number of ROIs contained in the image
    :return: two matrices, one with the high energy ratios, and one with the low energy
    """
    beam_energies = ['40kVp', '80kVp']

    # Attenuation coefficient ratios for each material and each energy
    low_ratios = np.empty(num_ROIs)
    high_ratios = np.empty(num_ROIs)

    for energy in beam_energies:

        path = folder + energy + '/'
        image = np.load(path+filename)

        for i in np.arange(num_ROIs):
            ROI = np.load(folder + 'Vial' + str(i) + '_MaskMatrix.npy')
            temp = np.multiply(image, ROI)

            # Calculate the average HU in the ROI
            HU = np.nanmean(temp)
            ratio = HU/1000 + 1

            if energy is '40kVp':
                low_ratios[i] = ratio
            else:
                high_ratios[i] = ratio

    return low_ratios, high_ratios


def ratios_pixel(folder='', filename='', crop=False):
    """
    Function that calculates the ratio between the linear attenuation coefficients of the material and water for each
    pixel in the matrix for each of the beam energies
    :param folder: folder where the images matrices live
    :param filename: filename of the specific image (must be .npy)
    :param crop: True if you have a crop mask to cut out the air around the object you're looking at
    :return: two matrices, one with the high energy ratios, and one with the low energy
    """
    beam_energies = ['40kVp', '80kVp']

    im_low = np.load(folder + beam_energies[0] + '/' + filename)  # Low energy image
    im_high = np.load(folder + beam_energies[1] + '/' + filename)  # High energy image

    # Crop background out if desired
    if crop:
        crop_mask = np.load(folder + 'Crop_Mask.npy')
        im_low = np.multiply(im_low, crop_mask)
        im_high = np.multiply(im_high, crop_mask)

    # Flatten both images to one dimensional arrays
    im_low = im_low.flatten()
    im_high = im_high.flatten()

    # Take out any nan values from the cropping
    im_low = im_low[~np.isnan(im_low)]
    im_high = im_high[~np.isnan(im_high)]

    # Attenuation coefficient ratios for each material and each energy
    low_ratios = np.empty(len(im_low))
    high_ratios = np.empty(len(im_high))

    # For every pixel in each image
    for i in np.arange(len(im_low)):
        # Get the HU value of each pixel and solve for the ratio
        ratio_low = im_low[i] / 1000 + 1
        ratio_high = im_high[i] / 1000 + 1

        high_ratios[i] = ratio_high
        low_ratios[i] = ratio_low

    return low_ratios, high_ratios


def Z_equation(Z, ratio1, ratio2):
    """
    Equation to find the root of in order to find the effective Z of the contrast material (Eq (3))
    :param Z: The current Z-value to plug into the equation
    :param ratio1: low energy ratio
    :param ratio2: high energy ratio
    :return: The value of the left hand side of equation (3) (Trying to get it to zero)
    """
    Z2 = Z*Z
    Z3 = Z*Z*Z
    Z4 = Z*Z*Z*Z

    # Calculate the second summations in each of the lines of eq (3)
    G1 = 0
    G2 = 0
    F1 = 0
    F2 = 0

    for i in np.arange(num_energies):
        G1 += wts_low[i] * (Z2*G[i, 0] + Z*G[i, 1] + G[i, 2])
        G2 += wts_high[i] * (Z2*G[i, 0] + Z*G[i, 1] + G[i, 2])
        F1 += wts_low[i] * (Z4*F[i, 0] + Z3*F[i, 1] + Z2*F[i, 2] + Z*F[i, 3] + F[i, 4])
        F2 += wts_high[i] * (Z4*F[i, 0] + Z3*F[i, 1] + Z2*F[i, 2] + Z*F[i, 3] + F[i, 4])

    solution = Z4 - ((ratio2*FG_Zw2*G1 - ratio1*FG_Zw1*G2) / (ratio1*FG_Zw1*F2 - ratio2*FG_Zw2*F1))

    return solution


def solve_Z(ratios_low, ratios_high):
    """
    Optimize Eq. (3) (Z_equation) to find the root
    :param low_ratios: The low energy ratios, must be of the form of an array
    :param high_ratios: The high energy ratios, must be of the form of an array
    :return: The Z value found for each of the given set of ratios
    """

    # Optimize the Z_equation for the given ratios
    Z_found = np.empty(len(ratios_low))

    # Store the indices of the entries that fail
    entries_to_delete = np.array([])

    for r in np.arange(len(ratios_low)):
        ratio1 = ratios_low[r]
        ratio2 = ratios_high[r]
        #print('Ratios', ratio1, ratio2)
        try:
            # Find the root of the equation between 4 and 30 with the given ratios
            Z_possible = optimize.brentq(Z_equation, 4, 30, args=(ratio1, ratio2))
        except ValueError as e:
            print(ValueError, e)
            print(r)
            entries_to_delete = np.append(entries_to_delete, int(r))
            continue
        #print('Z', Z_possible)
        Z_found[r] = Z_possible

        if len(entries_to_delete) > 0:
            Z_found = np.delete(Z_found, entries_to_delete)

        return Z_found

# Function to get the electron densities
def electron_density(ratios_low, Z_vals):
    """
    Function to find the electron densities for a given Z
    :param ratios_low: The low_ratios found from the low energy image
    :param Z_vals: The Z values found through the optimization process
    :return: The electron densities for
    """
    densities = np.empty(len(Z_vals))

    for i in np.arange(len(Z_vals)):
        Z = Z_vals[i]
        Z2 = Z*Z
        Z3 = Z*Z*Z
        Z4 = Z*Z*Z*Z
        ratio1 = ratios_low[i]
        # Calculate the denominator of the equation
        FG_Z = 0
        for j in np.arange(num_energies):
            FG_Z += wts_low[j] * (Z4*(Z4*F[j, 0] + Z3*F[j, 1] + Z2*F[j, 2] + Z*F[j, 3] + F[j, 4]) + Z2*G[j, 0] + Z*G[j, 1] + G[j, 2])

        # Calculate the density
        densities[i] = ratio1 * (FG_Zw1 / FG_Z)

    return densities

#%%
# Main file path for slices and attenuation data
slice_folder = 'D:/Research/Python Data/CBCT/Lan_7-18-19/5_Percent/'
start, stop = 125, 150
ROIs = 6

lr, hr = ratios_ROI(ROIs, folder=slice_folder, filename='volume0'+str(start)+'.npy')

#for z in zpts:
#    print(z, Z_equation(z, lr[4], hr[4]))

for i in np.arange(ROIs):
    x = solve_Z([lr[i]], [hr[i]])
    print('Z:', solve_Z([lr[i]], [hr[i]]))
    print('Density:', electron_density([lr[i]], [x]))
    print()

#%%

# Main file path for slices and attenuation data
slice_folder = 'D:/Research/Python Data/CBCT/Lan_7-18-19/3_Percent/'
start, stop = 125, 150

for i in np.arange(start, stop):
    lr, hr = ratios_ROI(6, folder=slice_folder, filename='volume0'+str(i)+'.npy')


    Zs = solve_Z(lr, hr)

    ps = electron_density(lr, Zs)
    print(Zs)
    print(ps)
