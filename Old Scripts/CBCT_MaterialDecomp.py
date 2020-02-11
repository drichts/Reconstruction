import numpy as np
import scipy.interpolate as interpolate
import scipy.optimize as optimize
import matplotlib.pyplot as plt

# Z eff value of water
Zw = 7.733
Zw4 = Zw*Zw*Zw*Zw
atten_folder = 'D:/Research/Attenuation Data/NPY Attenuation/'
spectra_folder = 'D:/Research/Attenuation Data/NPY Spectra/'

# Main file path for slices and attenuation data
slice_folder = 'D:/Research/Python Data/CBCT/Lan_7-15-19/3_Percent/'
#slice_folder = 'D:/Research/Python Data/CBCT/SARRP_7-25-19/'

energies = ['40kVp', '80kVp']  # Should be of the form '40kVp', '80kVp', etc.

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
spectra_weights_low = np.load(spectra_folder+'40kVp_weights.npy')
spectra_weights_high = np.load(spectra_folder+'80kVp_weights.npy')

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

scatter_corr = np.divide(atomic_mass, Z_values)
PE_corr = np.divide(atomic_mass, np.power(Z_values, 5))

for i in np.arange(len(scatter_matrix[0])):
    scatter_matrix[:, i] = np.multiply(scatter_corr, scatter_matrix[:, i])
    PE_matrix[:, i] = np.multiply(PE_corr, PE_matrix[:, i])

# Bins 0-4 and 6 (summed) all on one image
fig, ax = plt.subplots(1, 2, figsize=(15, 9))
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
ax[0].plot(spectra_energies, spectra_weights_low, lw=2)
ax[0].set_title('40 kVp spectra', fontsize=40)
ax[0].set_xlabel('Energy (MeV)', fontsize=35)
ax[0].set_xlim([0, 0.1])
ax[0].set_ylim([0, 0.032])
ax[0].tick_params(labelsize=25)

ax[1].plot(spectra_energies, spectra_weights_high)
ax[1].set_title('80 kVp spectra', fontsize=40)
ax[1].set_xlabel('Energy (MeV)', fontsize=35)
ax[1].set_xlim([0, 0.1])
ax[1].set_ylim([0, 0.016])
ax[1].tick_params(labelsize=25)
plt.show()
#plt.pause(1)
#plt.close()

#%%
#plt.plot(Z_values, PE_matrix[:, 99], color='green')
#plt.plot(Z_values, PE_matrix[:, 200], color='black')
#plt.title('F vs. Z')
#plt.legend(['50.25 keV', '100 keV'])
#plt.xlim([5, 15])
#plt.show()
#plt.pause(2)
#plt.close()

#plt.plot(Z_values, scatter_matrix[:, 99], color='green')
#plt.plot(Z_values, scatter_matrix[:, 200], color='black')
#plt.title('G vs. Z')
#plt.legend(['50.25 keV', '100 keV'])
#plt.xlim([5, 15])
#plt.show()
#plt.pause(2)
#plt.close()

#%%
# Function that will interpolate F (photoelectric) based on the input energy and Z-value
# Call using F(E, Z) (E and Z can be arrays)
F = interpolate.interp2d(spectra_energies, Z_values, PE_matrix, kind='cubic')
#F = np.empty((num_energies, 3))
#for i in np.arange(num_energies):
#    temp = PE_matrix[:, i]
#    F[i] = np.polyfit(Z_values, temp, 2)

# Function that will interpolate G (scattering) based on the input energy and Z-value
# Call using G(E, Z) (E and Z can be arrays)
G = interpolate.interp2d(spectra_energies, Z_values, scatter_matrix, kind='cubic')
#G = np.empty((num_energies, 2))
#for i in np.arange(num_energies):
#    temp = scatter_matrix[:, i]
#    G[i] = np.polyfit(Z_values, temp, 1)

# Calculate the Zw components of the equations
F_Zw = F(spectra_energies, Zw)
F_Zw = np.multiply(F_Zw, Zw4)
G_Zw = G(spectra_energies, Zw)

# Combine them
FG_Zw = np.add(F_Zw, G_Zw)
FG_Zw_1 = np.dot(spectra_weights_low, FG_Zw)
FG_Zw_2 = np.dot(spectra_weights_high, FG_Zw)

# Function that calculates the ratio between the linear attenuation coefficients of the material and water for each of
# the saved ROIs, or calls the generateROI function to create the ROIs necessary for each of the beam energies
# Filename must be .npy
def ratios_ROI(beam_energies=[], folder='', filename='', num_ROIs=4, radius=2):

    beam_energies = ['40kVp', '80kVp']

    # Attenuation coefficient ratios for each material and each energy
    low_ratios = np.empty(num_ROIs)
    high_ratios = np.empty(num_ROIs)

    for energy in beam_energies:

        path = folder + energy + '/'
        temp = np.load(path + filename)

        # Load all of the ROIs
        v = np.empty([num_ROIs, 120, 120])
        for k in np.arange(num_ROIs):
            v[k, :, :] = np.load(folder + 'Vial' + str(k) + '_MaskMatrix.npy')

        for i in np.arange(num_ROIs):

            # Get measured HU average and solve for the ratio
            HU = np.nanmean(temp * v[i])

            ratio = HU/1000 + 1

            if energy is beam_energies[1]:
                high_ratios[i] = ratio
            else:
                low_ratios[i] = ratio

    return low_ratios, high_ratios


# Function that calculates the ratio between the linear attenuation coefficients of the material and water for each
# pixel in the matrix for each of the beam energies
# crop is if you have generated a mask to crop out the background and want to use it
def ratios_pixel(beam_energies=[], folder='', filename='', crop=False):

    beam_energies = ['40kVp', '80kVp']

    im_low = np.load(folder + beam_energies[0] + '/' + filename)
    im_high = np.load(folder + beam_energies[1] + '/' + filename)

    # Crop background out if desired
    if crop:
        crop_mask = np.load(folder + 'Crop_Mask.npy')
        im_low = np.multiply(im_low, crop_mask)
        im_high = np.multiply(im_high, crop_mask)

    im_low = im_low.flatten()
    im_high = im_high.flatten()

    # Take out any nan values
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

# Equation to find the root of in order to find the effective Z of the contrast material
def Z_equation(Z, ratios, spectra_wts_low, spectra_wts_high, spectra_energ):

    mu1 = ratios[0]
    mu2 = ratios[1]
    Z4 = np.power(Z, 4)

    w1 = spectra_wts_low
    w2 = spectra_wts_high

    GZ = G(spectra_energ, Z)

    FZ = F(spectra_energ, Z)

    GZ1 = np.dot(w1, GZ)
    GZ2 = np.dot(w2, GZ)

    FZ1 = np.dot(w1, FZ)
    FZ2 = np.dot(w2, FZ)

    answer = Z4 - (mu2*FG_Zw_2*GZ1 - mu1*FG_Zw_1*GZ2) / (mu1*FG_Zw_1*FZ2 - mu2*FG_Zw_2*FZ1)

    return answer


def solve_for_Z(low_ratios, high_ratios, spec_energies, low_wts, high_wts):

    # Optimize the Z_equation for the given ratios and solve for the electron densities as well
    Z_experimental = np.empty(len(low_ratios))

    entries_to_delete = np.array([])

    for r in np.arange(len(low_ratios)):
        mu1 = low_ratios[r]
        mu2 = high_ratios[r]

        try:
            Z_exp = optimize.brentq(Z_equation, 4, 30, args=([mu1, mu2], low_wts, high_wts,
                                                             spec_energies))
        except ValueError as e:
            print(ValueError, e)
            print(r)
            entries_to_delete = np.append(entries_to_delete, int(r))
            continue

        Z_experimental[r] = Z_exp

        #print('Effective Z = ', Z_exp)

    #if len(entries_to_delete) > 0:
    #    Z_experimental = np.delete(Z_experimental, entries_to_delete)

    return Z_experimental


# Function to get the electron densities
def electron_density(low_ratios, ex_z):

    electron_densities = np.empty(len(ex_z))

    for ind, z in enumerate(ex_z):

        mu = low_ratios[ind]

        FZ = F(spectra_energies, z)
        FZ = np.multiply(FZ, z**4)
        GZ = G(spectra_energies, z)

        FG_Z = np.add(FZ, GZ)
        FG_Z = np.dot(spectra_weights_low, FG_Z)

        elec_dens = mu * (FG_Zw_1 / FG_Z)

        electron_densities[ind] = elec_dens

    return electron_densities

lr, hr = ratios_ROI(folder=slice_folder, filename='volume0135.npy', num_ROIs=6)
zpts = np.linspace(4, 30, 100)

for z in zpts:
    ans = Z_equation(z, [lr[5], hr[5]], spectra_weights_low, spectra_weights_high, spectra_energies)

    #print('Z:', z, 'Ans:', ans)

ans = solve_for_Z([lr[5]], [hr[5]], spectra_energies, spectra_weights_low, spectra_weights_high)

print('Z:', ans)

#HU = 1000
#mu1 = HU/1000 + 1

elec = electron_density([lr[5]], [ans])
print('Electron density:', elec)



#%%

#plt.imshow(np.load(slice_folder + energies[1] + '/volume0205.npy'))
plt.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    right=False,
    left=False,
    top=False,         # ticks along the top edge are off
    labelbottom=False,
    labelleft=False) # labels along the bottom edge are off
plt.show()

#start, stop = 125, 126

# For ROI-wise
#num_ROIs = 6
#Zs = np.empty([stop-start, num_ROIs])
#ps = np.empty([stop-start, num_ROIs])

# For pixel-wise
#Zs = np.array([])
#ps = np.array([])

#for i in np.arange(start, stop):
#    lr, hr = ratios_ROI(beam_energies=energies, folder=slice_folder, filename='volume0' + str(i) + '.npy',
#                        num_ROIs=num_ROIs, radius=6)  # ROI
    #lr, hr = ratios_pixel(beam_energies=energies, folder=slice_folder, filename='volume0' + str(i) + '.npy',
    #                       crop=True)  # Pixel

#    z = solve_for_Z(lr, hr, spectra_energies, spectra_weights_low, spectra_weights_high)

    #p = electron_density(lr, z)

    # ROI
#    Zs[i-start] = z
    #ps[i-start] = p

    # Pixel
    #Zs = np.append(Zs, z)
    #ps = np.append(ps, p)

#np.save(slice_folder + 'Z-ROI.npy', Zs)
#np.save(slice_folder + 'dens-ROI.npy', ps)

