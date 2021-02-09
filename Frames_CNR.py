import numpy as np
import matplotlib.pyplot as plt
from obsolete import sct_analysis as sct
import seaborn as sns
from scipy.optimize import curve_fit
import mask_functions as grm

directory = 'D:/Research/Python Data/Spectral CT/'
folders = ['Cu_0.5_Time_0.1_11-4-19', 'Frame2_Time_0.1', 'Frame3_Time_0.1', 'Frame4_Time_0.1', 'Frame5_Time_0.1',
           'Frame6_Time_0.1', 'Frame7_Time_0.1', 'Frame8_Time_0.1', 'Frame9_Time_0.1', 'Frame10_Time_0.1']


def sqroot(x, a, b):
    return a*np.sqrt(b*x)

def logg(x, a, b):
    return a*np.log(b*x)

#%% Creates an air mask for all folders
# Calculate the background based on the entire phantom, exluding the vial masks, or the air mask

# Go through each of the folders
for i, folder in enumerate(folders):
    image = np.load(directory + folder + '/Slices/Bin6_Slice13.npy')

    continue_flag = True
    while continue_flag:
        air = grm.air_mask(image)
        val = input('Were the ROIs acceptable? (y/n)')
        if val is 'y':
            continue_flag = False

    # Save if desired
    np.save(directory + folder + '/Air_Mask.npy', air)

#%% Gets and plot K-edge CNR for all different frames

# Edges to easily call each of the file types
edges = ['4-3', '2-1', '3-2', '1-0', '11-10', '9-8', '10-9', '8-7']
#edges = ['11-10', '9-8', '10-9', '8-7']

# Create a figure to plot the CNR of the contrast agent of your choice
# 0 = Au, 1 = Lu, 2 = Dy, 3 = Gd
element = 7
fig = plt.figure(figsize=(8, 8))
xpts = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])  # Time points
CNR = np.zeros(10)  # To hold the CNR values for each of the time points
CNR_err = np.zeros(10)

sns.set_style('whitegrid')

# Go through the 3 folders
for i, folder in enumerate(folders):
    low_z, high_z = 10, 18  # Get the good slices for this folder
    vials = np.load(directory + folder + '/Vial_Masks.npy')  # Vial masks
    background = np.load(directory + folder + '/Phantom_Mask.npy')  # Phantom mask

    # Initialize matrices to hold the K-edge CNR and noise data
    # Orientation: [0-3: CNR of water in K-edge image in descending order (4-3), (3-2), etc. 4-7: CNR of the specfio
    # element (follows the same order), slice]
    mean_kedge = np.empty([8, high_z - low_z + 1])
    std_kedge = np.empty([8, high_z - low_z + 1])

    # Go through the slices
    # Order: Au, Lu, Dy, Gd
    for z in np.arange(low_z, high_z + 1):

        # Calculate the K-edge CNR in the water vial of each K-edge image
        image43 = np.load(directory + folder + '/K-Edge/Bin' + edges[0] + '_Slice' + str(z) + '.npy')
        image32 = np.load(directory + folder + '/K-Edge/Bin' + edges[2] + '_Slice' + str(z) + '.npy')
        image21 = np.load(directory + folder + '/K-Edge/Bin' + edges[1] + '_Slice' + str(z) + '.npy')
        image10 = np.load(directory + folder + '/K-Edge/Bin' + edges[3] + '_Slice' + str(z) + '.npy')

        image1110 = np.load(directory + folder + '/K-Edge/Bin' + edges[4] + '_Slice' + str(z) + '.npy')
        image109 = np.load(directory + folder + '/K-Edge/Bin' + edges[6] + '_Slice' + str(z) + '.npy')
        image98 = np.load(directory + folder + '/K-Edge/Bin' + edges[5] + '_Slice' + str(z) + '.npy')
        image87 = np.load(directory + folder + '/K-Edge/Bin' + edges[7] + '_Slice' + str(z) + '.npy')

        # Calculate the K-edge CNR in the element vial in the corresponding K-edge image
        mean_kedge[0, z - low_z], std_kedge[0, z - low_z] = sct.cnr(image43, vials[1], background)
        mean_kedge[1, z - low_z], std_kedge[1, z - low_z] = sct.cnr(image32, vials[3], background)
        mean_kedge[2, z - low_z], std_kedge[2, z - low_z] = sct.cnr(image21, vials[2], background)
        mean_kedge[3, z - low_z], std_kedge[3, z - low_z] = sct.cnr(image10, vials[4], background)

        mean_kedge[4, z - low_z], std_kedge[0, z - low_z] = sct.cnr(image1110, vials[1], background)
        mean_kedge[5, z - low_z], std_kedge[1, z - low_z] = sct.cnr(image109, vials[3], background)
        mean_kedge[6, z - low_z], std_kedge[2, z - low_z] = sct.cnr(image98, vials[2], background)
        mean_kedge[7, z - low_z], std_kedge[3, z - low_z] = sct.cnr(image87, vials[4], background)


    # Average the data over the slices
    mean_kedge = np.mean(mean_kedge, axis=1)
    std_kedge = np.mean(std_kedge, axis=1)

    # Save the matrices if desired
    #np.save(directory + folder + '/Mean_Kedge_CNR_Time.npy', mean_kedge)
    #np.save(directory + folder + '/Std_Kedge_CNR_Time.npy', std_kedge)

    CNR[i] = mean_kedge[element]
    CNR_err[i] = std_kedge[element]

# Do a square root fit
coeffs, covar = curve_fit(sqroot, xpts, CNR)
x_sq = np.linspace(0, 2, 500)
y_sq = sqroot(x_sq, coeffs[0], coeffs[1])

# Do a log fit
coeffs2, covar2 = curve_fit(logg, xpts, CNR)
y2 = logg(x_sq, coeffs2[0], coeffs2[1])

titles = ['Gold SEC', 'Lutetium SEC', 'Dysprosium SEC', 'Gadolinium SEC',
          'Gold CC', 'Lutetium CC', 'Dysprosium CC', 'Gadolinium CC']

#plt.errorbar(xpts, CNR, yerr=CNR_err, capsize=6, lw=3, capthick=2, marker='o', ls='', markersize=10)  # With error
plt.scatter(xpts, CNR, marker='o', color='red')  # Without errorbars
plt.plot(x_sq, y_sq, color='mediumblue')
plt.plot(x_sq, y2, color='green')
plt.title(titles[element] + ' CNR vs. Projection Time', fontsize=20)
plt.xlabel('Projection time (s)', fontsize=20)
plt.ylabel('K-edge CNR', fontsize=20)
plt.tick_params(labelsize=18)
plt.legend(['Sq. root fit', 'Ln fit', 'Data'], fontsize=17)
plt.annotate('Sq root fit: a=%5.3f, b=%5.3f' % tuple(coeffs), (0.55, 4.2), fontsize=17)
plt.annotate('Poly fit: a=%5.3f, b=%5.3f' % tuple(coeffs2), (0.55, 2.8), fontsize=17)
plt.ylim([0, np.max(CNR)*2])
plt.xlim([0, 1.2])
plt.show()
plt.savefig(directory + 'Time 0.1 Multiple Frames/K-EdgeCNR_' + titles[element] + '.png', dpi=500)

#%% Gets and plot K-edge SNR for all different frames

# Edges to easily call each of the file types
edges = ['4-3', '2-1', '3-2', '1-0', '11-10', '9-8', '10-9', '8-7']

# Create a figure to plot the CNR of the contrast agent of your choice
# 0 = Au, 1 = Lu, 2 = Dy, 3 = Gd
element = 7
fig = plt.figure(figsize=(8, 8))
xpts = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])  # Time points
SNR = np.zeros(10)  # To hold the CNR values for each of the time points

sns.set_style('whitegrid')

# Go through the 3 folders
for i, folder in enumerate(folders):
    low_z, high_z = 10, 18  # Get the good slices for this folder
    vials = np.load(directory + folder + '/Vial_Masks.npy')  # Vial masks
    background = np.load(directory + folder + '/Phantom_Mask.npy')  # Phantom mask

    # Initialize matrices to hold the K-edge CNR and noise data
    # Orientation: [0-3: CNR of water in K-edge image in descending order (4-3), (3-2), etc. 4-7: CNR of the specfio
    # element (follows the same order), slice]
    mean_kedge = np.empty([8, high_z - low_z + 1])
    std_kedge = np.empty([8, high_z - low_z + 1])

    # Go through the slices
    # Order: Au, Lu, Dy, Gd
    for z in np.arange(low_z, high_z + 1):

        # Grab the images for each of the agents
        image43 = np.load(directory + folder + '/K-Edge/Bin' + edges[0] + '_Slice' + str(z) + '.npy')
        image32 = np.load(directory + folder + '/K-Edge/Bin' + edges[2] + '_Slice' + str(z) + '.npy')
        image21 = np.load(directory + folder + '/K-Edge/Bin' + edges[1] + '_Slice' + str(z) + '.npy')
        image10 = np.load(directory + folder + '/K-Edge/Bin' + edges[3] + '_Slice' + str(z) + '.npy')

        image1110 = np.load(directory + folder + '/K-Edge/Bin' + edges[4] + '_Slice' + str(z) + '.npy')
        image109 = np.load(directory + folder + '/K-Edge/Bin' + edges[6] + '_Slice' + str(z) + '.npy')
        image98 = np.load(directory + folder + '/K-Edge/Bin' + edges[5] + '_Slice' + str(z) + '.npy')
        image87 = np.load(directory + folder + '/K-Edge/Bin' + edges[7] + '_Slice' + str(z) + '.npy')

        # Calculate the noise in each of the vials
        noise43 = np.nanstd(image43 * background)
        noise32 = np.nanstd(image32 * background)
        noise21 = np.nanstd(image21 * background)
        noise10 = np.nanstd(image10 * background)
        noise1110 = np.nanstd(image1110 * background)
        noise109 = np.nanstd(image109 * background)
        noise98 = np.nanstd(image98 * background)
        noise87 = np.nanstd(image87 * background)

        # Calculate the signal from each the vial in each image
        signal43 = np.nanmean(image43 * vials[1])
        signal32 = np.nanmean(image32 * vials[3])
        signal21 = np.nanmean(image21 * vials[2])
        signal10 = np.nanmean(image10 * vials[4])

        signal1110 = np.nanmean(image1110 * vials[1])
        signal109 = np.nanmean(image109 * vials[3])
        signal98 = np.nanmean(image98 * vials[2])
        signal87 = np.nanmean(image87 * vials[4])

        # Calculate the K-edge SNR in the element vial in the corresponding K-edge image
        mean_kedge[0, z - low_z] = signal43/noise43
        mean_kedge[1, z - low_z] = signal32/noise32
        mean_kedge[2, z - low_z] = signal21/noise21
        mean_kedge[3, z - low_z] = signal10/noise10
        mean_kedge[4, z - low_z] = signal1110 / noise1110
        mean_kedge[5, z - low_z] = signal109 / noise109
        mean_kedge[6, z - low_z] = signal98 / noise98
        mean_kedge[7, z - low_z] = signal87 / noise87

    # Average the data over the slices
    mean_kedge = np.mean(mean_kedge, axis=1)

    # Save the matrices if desired
    #np.save(directory + folder + '/Mean_Kedge_SNR_Time.npy', mean_kedge)

    SNR[i] = mean_kedge[element]

# Do a square root fit
coeffs, covar = curve_fit(sqroot, xpts, SNR)
x_sq = np.linspace(0, 2, 500)
y_sq = sqroot(x_sq, coeffs[0], coeffs[1])

# Do a log fit
coeffs2, covar2 = curve_fit(logg, xpts, SNR)
y2 = logg(x_sq, coeffs2[0], coeffs2[1])

titles = ['Gold SEC', 'Lutetium SEC', 'Dysprosium SEC', 'Gadolinium SEC',
          'Gold CC', 'Lutetium CC', 'Dysprosium CC', 'Gadolinium CC']


#plt.errorbar(xpts, CNR, yerr=CNR_err, capsize=6, lw=3, capthick=2, marker='o', ls='', markersize=10)  # With error
plt.scatter(xpts, SNR, marker='o', color='red')  # Without errorbars
plt.plot(x_sq, y_sq, color='mediumblue')
plt.plot(x_sq, y2, color='green')
plt.title(titles[element] + ' SNR vs. Projection Time', fontsize=20)
plt.xlabel('Projection time (s)', fontsize=20)
plt.ylabel('K-edge SNR', fontsize=20)
plt.tick_params(labelsize=18)
plt.legend(['Sq. root fit', 'Ln fit', 'Data'], fontsize=17)
#plt.annotate('Sq root fit: a=%5.3f, b=%5.3f' % tuple(coeffs), (0.55, 4.2), fontsize=17)
#plt.annotate('Poly fit: a=%5.3f, b=%5.3f' % tuple(coeffs2), (0.55, 2.8), fontsize=17)
plt.ylim([0, np.max(SNR)*2])
plt.xlim([0, 1.2])
plt.show()
plt.savefig(directory + 'Time 0.1 Multiple Frames/K-EdgeSNR_' + titles[element] + '.png', dpi=500)

#%% Gets and plots CT SNR for all different frames

#time_folders = ['Cu_0.5_Time_0.1_11-4-19', 'Cu_0.5_Time_0.1_0.5s_11-4-19', 'Cu_0.5_Time_0.1_1s_11-4-19']
#real_folders = ['Cu_0.5_Time_0.1_11-4-19', 'Cu_0.5_Time_0.5_11-11-19', 'Cu_0.5_Time_1.0_02-20-20']

# Create a figure to plot the SNR of the contrast agent of your choice
# 1 = Au, 3 = Lu, 2 = Dy, 4 = Gd
element = 1
b = 12
tt = 'CC'
fig = plt.figure(figsize=(5, 5))
xpts = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])  # Time points for reg. folders
#xpts = np.array([0.1, 0.5, 1.0])  # Time points for time folders
SNR = np.zeros(len(xpts))  # To hold the CNR values for each of the time points
time_noise = np.zeros(len(xpts))
sns.set_style('whitegrid')


# Go through the folders
for i, folder in enumerate(folders):
    low_z, high_z = 10, 18  # Get the good slices for this folder
    vials = np.load(directory + folder + '/Vial_Masks.npy')  # Vial masks
    background = np.load(directory + folder + '/Phantom_Mask.npy')  # Phantom mask
    #air = np.load(directory + folder + '/Air_Mask.npy')

    # Initialize matrices to hold the SNR and noise
    mean_reg = np.empty(high_z - low_z + 1)
    noise = np.empty(high_z - low_z + 1)
    # Go through the slices
    # Order: Au, Lu, Dy, Gd
    for z in np.arange(low_z, high_z + 1):

        # Grab the images for each of the agents
        image43 = np.load(directory + folder + '/CT Sum Slices/' + tt + '_' + str(z) + '.npy')

        # Calculate the noise in each of the vials
        noise43 = np.nanstd(image43 * vials[0])
        noise[z - low_z] = noise43

        # Calculate the signal from each the vial in each image
        signal43 = np.nanmean(image43 * vials[element])

        # Calculate the K-edge SNR in the element vial in the corresponding K-edge image
        mean_reg[z - low_z] = signal43/noise43

    # Average the data over the slices
    print(mean_reg)
    print(noise)
    print()
    mean_reg = np.mean(mean_reg)

    # Save the matrices if desired
    #np.save(directory + folder + '/Mean_REG_SNR_Time.npy', mean_kedge)

    SNR[i] = mean_reg
    time_noise[i] = np.mean(noise)

# Do a square root fit
coeffs, covar = curve_fit(sqroot, xpts, SNR)
x_sq = np.linspace(0, 1.2, 100)
y_sq = sqroot(x_sq, coeffs[0], coeffs[1])

# Do a log fit
coeffs2, covar2 = curve_fit(logg, xpts, SNR)
y2 = logg(x_sq, coeffs2[0], coeffs2[1])

titles = ['water', 'Gold', 'Dysprosium', 'Lutetium', 'Gadolinium']

#plt.errorbar(xpts, CNR, yerr=CNR_err, capsize=6, lw=3, capthick=2, marker='o', ls='', markersize=10)  # With error
#plt.scatter(xpts, SNR, marker='o', color='red')  # Without errorbars
plt.plot(xpts, noise)
#plt.plot(x_sq, y_sq, color='mediumblue')
#plt.plot(x_sq, y2, color='green')
plt.title(tt + ' summed bin', fontsize=20)
plt.xlabel('Projection time (s)', fontsize=20)
plt.ylabel('SNR', fontsize=20)
plt.tick_params(labelsize=18)
plt.legend(['Sq. root fit', 'Ln fit', 'Data'], fontsize=17)
#plt.annotate('Sq root fit: a=%5.3f, b=%5.3f' % tuple(coeffs), (0.55, 4.2), fontsize=17)
#plt.annotate('Poly fit: a=%5.3f, b=%5.3f' % tuple(coeffs2), (0.55, 2.8), fontsize=17)
#plt.ylim([0, np.max(SNR)*2])
#plt.xlim([0, 1.2])
plt.show()
plt.subplots_adjust(left=0.20, bottom=0.15, right=0.95, top=0.9)
#plt.savefig(directory + 'Time 0.1 Multiple Frames/Summed' + tt + '.png', dpi=500)