import numpy as np
import matplotlib.pyplot as plt
import sCT_Analysis as sct
import generateROImask as grm

directory = 'D:/Research/Python Data/Spectral CT/'

folders = ['Al_2.0_8-14-19', 'Al_2.0_10-17-19_3P', 'Al_2.0_10-17-19_1P',
           'Cu_0.5_8-14-19', 'Cu_0.5_9-13-19', 'Cu_0.5_10-17-19',
           'Cu_1.0_8-14-19', 'Cu_1.0_9-13-19', 'Cu_1.0_10-17-19',
           'Cu_0.5_Time_0.5_11-11-19', 'Cu_0.5_Time_0.1_11-4-19',
           'AuGd_width_5_12-2-19', 'AuGd_width_10_12-2-19', 'AuGd_width_14_12-2-19', 'AuGd_width_20_12-9-19']
folder2 = 'Cu_0.5_Time_1.0_02-20-20'
folder3 = 'Cu_0.5_Time1.0_Uniformity_02-25-20'

time_folders = ['Cu_0.5_Time_0.1_11-4-19', 'Cu_0.5_Time_0.1_0.5s_11-4-19', 'Cu_0.5_Time_0.1_1s_11-4-19']
gs = [10, 18]
gs2 = [12, 18]

# Slices that actually have data (not just blank, no bubbles in the vials)
good_slices = [[5, 19], [10, 18], [11, 18],
               [4, 15], [7, 15], [12, 19],
               [4, 14], [5, 16], [10, 19],
               [10, 19], [10, 18],
               [11, 19], [11, 19], [11, 19], [11, 19]]

#%% Phantom masks

# Calculate the background based on the entire phantom, exluding the vial masks, or the air mask

# Go through each of the folders
for i, folder in enumerate(time_folders[1:]):
    image = np.load(directory + folder + '/Slices/Bin6_Slice13.npy')

    continue_flag = True
    while continue_flag:
        phantom_mask = grm.entire_phantom(image)
        #air = grm.air_mask(image)
        val = input('Were the ROIs acceptable? (y/n)')
        if val is 'y':
            continue_flag = False

    # Save if desired
    np.save(directory + folder + '/Phantom_Mask.npy', phantom_mask)
    #np.save(directory + folder + '/Air_Mask.npy', air)

#%% Calcuulate the noise in each slice for all the "good" slices to find the lowest noise slice
# Goes through each folder

#low_noise = np.empty([len(folders), 2])  # Slice number of the lowest noise for each folders and the noise value
#phantom_noise = np.empty([len(folders), 2])  # Noise of the phantom
#water_noise = np.empty([len(folders), 2])  # Noise in the water vial
#CNR = np.empty([len(folders), 6])  # CNR of all vials including the water vial

#for i, folder in enumerate(folders):
    #low_z, high_z = good_slices[i][0], good_slices[i][1]

    # Current phantom mask
    #p_mask = np.load(directory + folder + '/Phantom_Mask.npy')

    # Empty matrices to hold the noise and CNR values for each slice
    #temp_noise = np.zeros(high_z-low_z+1)
    #temp_CNR = np.zeros([high_z-low_z+1, 6])

    # Find the slice with the least noise
    #low_noise[i] = sct.find_least_noise(folder, low_z, high_z)

    # Go through each slice and calc noise from std and then calc CNR
    #for j in np.arange(low_z, high_z+1):
    #    image = np.load(directory + folder + '/Slices/Bin6_Slice' + str(j) + '.npy')
    #    temp_noise[j-low_z] = np.nanstd(image*p_mask)
    #    temp_CNR[j-low_z, :] = sct.get_ct_cnr(folder, j, type='phantom')

    #print(temp_noise)

    # Calculate the average noise over the slices and the std of the noise as well
    #phantom_noise[i, :] = np.array([np.nanmean(temp_noise), np.nanstd(temp_noise)])

    # Calculate the average CNR over the slices
    #CNR[i] = np.mean(temp_CNR, axis=0)

#%% Calculate CNR of K-edge Images

# Empty matrices for the data
#lan_CNR = np.empty([11, 4])  # CNR for the Lanthanide phantoms (all except the bin width optimization)
#AuGd_CNR = np.empty([4, 6])  # CNR for the AuGd phantom

# All the filter optimization folders and the time optimization
#for i, folder in enumerate(folders[0:11]):
#    low_z, high_z = good_slices[i][0], good_slices[i][1]  # Grab good slices for that particular folder
#    low_noise[i] = sct.find_least_noise(folder, low_z, high_z)  # Find lowest noise slice for that folder
#    vials = np.load(directory + folder + '/Vial_Masks.npy')  # Load vial masks

    # Load the each image num_num signifies num subtract num
#    img1_0 = np.load(directory + folder + '/K-Edge/Bin1-0_Slice' + str(int(low_noise[i][0])) + '.npy')
#    img2_1 = np.load(directory + folder + '/K-Edge/Bin2-1_Slice' + str(int(low_noise[i][0])) + '.npy')
#    img3_2 = np.load(directory + folder + '/K-Edge/Bin3-2_Slice' + str(int(low_noise[i][0])) + '.npy')
#    img4_3 = np.load(directory + folder + '/K-Edge/Bin4-3_Slice' + str(int(low_noise[i][0])) + '.npy')

    # Calculate the CNR for each K-edge image
#    au = sct.cnr(img4_3, vials[1], vials[0])
#    dy = sct.cnr(img2_1, vials[2], vials[0])
#    lu = sct.cnr(img3_2, vials[3], vials[0])
#    gd = sct.cnr(img1_0, vials[4], vials[0])

    # The CNR for each K-edge image for that particular folder
#    lan_CNR[i, :] = np.array([au, dy, lu, gd])

# These folders are the bin width folders
#for i, folder in enumerate(folders[11:]):
#    low_z, high_z = good_slices[i][0], good_slices[i][1]  # Grab good slices
#    low_noise[i] = sct.find_least_noise(folder, low_z, high_z)  # Find lowest noise slice
#    vials = np.load(directory + folder + '/Vial_Masks.npy')  # Load the vial masks

    # Only need the 1-0 and 4-3 K-edge images
#    img1_0 = np.load(directory + folder + '/K-Edge/Bin1-0_Slice' + str(15) + '.npy')
#    img4_3 = np.load(directory + folder + '/K-Edge/Bin4-3_Slice' + str(15) + '.npy')

    # Calculate the CNR for 3%, 0.5% and 0.5% mixed for Au (4-3) and Gd (1-0)
#    au3 = sct.cnr(img4_3, vials[1], vials[0])
#    au05 = sct.cnr(img4_3, vials[5], vials[0])
#    aum = sct.cnr(img4_3, vials[3], vials[0])
#    gd3 = sct.cnr(img1_0, vials[4], vials[0])
#    gd05 = sct.cnr(img1_0, vials[2], vials[0])
#    gdm = sct.cnr(img1_0, vials[3], vials[0])

    # Set them in array for that particular folder
#    AuGd_CNR[i, :] = np.array([au3, au05, aum, gd3, gd05, gdm])

#%% Get airscan flux
flux = np.empty(9)

# Get the total flux of each of the first 9 folders
for i, folder in enumerate(folders[0:9]):
    flux[i] = sct.airscan_flux(folder)[6]

#%% Figure 1 Information Collection

# Collect data from the first 9 folders
for i, folder in enumerate(folders[0:9]):
    vials = np.load(directory + folder + '/Vial_Masks.npy')  # Load the vial masks
    low_z, high_z = good_slices[i][0], good_slices[i][1]  # Get the good slices for this specifc

    # Initialize matrices for the mean and std signal from each of vials
    # Orientation [Bin, Slice, Vial]
    mean_signal = np.zeros([5, high_z-low_z+1, 6])
    std_signal = np.zeros([5, high_z-low_z+1, 6])

    # Go through each bin (0-4)
    for b in np.arange(5):
        # Go through each of the good slices
        for j in np.arange(low_z, high_z+1):
            image = np.load(directory + folder + '/Slices/Bin' + str(b) + '_Slice' + str(j) + '.npy')  # Load the image
            # Go through each of the 6 vials (0-5)
            for k, vial in enumerate(vials):
                mult_img = np.multiply(vial, image)  # Multiply the image by the specific vial mask
                mean_signal[b, j-low_z, k] = np.nanmean(mult_img)  # Calc mean signal in the vial
                std_signal[b, j-low_z, k] = np.nanstd(mult_img)  # Calc std of signal in the vial

    # Final arrays are of the form [Bin, Vial]; Taking the mean over the slices
    mean_signal = np.mean(mean_signal, axis=1)
    std_signal = np.mean(std_signal, axis=1)

    # Save the matrices if desired
    #np.save(directory + folder + '/Mean_Signal.npy', mean_signal)
    #np.save(directory + folder + '/Std_Signal.npy', std_signal)

#%% Figure 1 K-Edge CNR
import sCT_Analysis as sct

# Get the K-Edge CNR in the first 9 folders

# Edges to easily call each of the file types
edges = ['4-3', '2-1', '3-2', '1-0']

# Go through each of the folders
for i, folder in enumerate(folders[0:9]):
    low_z, high_z = good_slices[i][0], good_slices[i][1]  # Get the specific good slices for this folder
    vials = np.load(directory + folder + '/Vial_Masks.npy')  # Load the vial masks
    background = np.load(directory + folder + '/Phantom_Mask.npy')  # Load the background mask for the entire phantom

    # Initialize the matrices to hold the K-edge CNR data
    # Orientation: [0-3: CNR of water in K-edge image in descending order (4-3), (3-2), etc. 4-7: CNR of the specfio
    # element (follows the same order), slice]
    mean_kedge = np.empty([8, high_z - low_z + 1])
    std_kedge = np.empty([8, high_z - low_z + 1])

    # Go through the slices
    # Order: Au, Lu, Dy, Gd
    for z in np.arange(low_z, high_z + 1):

        # Load the specific K-edge images (43 is the 4-3 Image)
        image43 = np.load(directory + folder + '/K-Edge/Bin' + edges[0] + '_Slice' + str(z) + '.npy')
        image32 = np.load(directory + folder + '/K-Edge/Bin' + edges[2] + '_Slice' + str(z) + '.npy')
        image21 = np.load(directory + folder + '/K-Edge/Bin' + edges[1] + '_Slice' + str(z) + '.npy')
        image10 = np.load(directory + folder + '/K-Edge/Bin' + edges[3] + '_Slice' + str(z) + '.npy')

        # Calculate the mean and std of the CNR for the water vial in each of the images
        mean_kedge[0, z - low_z], std_kedge[0, z - low_z] = sct.cnr(image43, vials[0], background)
        mean_kedge[1, z - low_z], std_kedge[1, z - low_z] = sct.cnr(image32, vials[0], background)
        mean_kedge[2, z - low_z], std_kedge[2, z - low_z] = sct.cnr(image21, vials[0], background)
        mean_kedge[3, z - low_z], std_kedge[3, z - low_z] = sct.cnr(image10, vials[0], background)

        # Calculate the mean and std of the CNR in the element's specific vial
        mean_kedge[4, z - low_z], std_kedge[4, z - low_z] = sct.cnr(image43, vials[1], background)
        mean_kedge[5, z - low_z], std_kedge[5, z - low_z] = sct.cnr(image32, vials[3], background)
        mean_kedge[6, z - low_z], std_kedge[6, z - low_z] = sct.cnr(image21, vials[2], background)
        mean_kedge[7, z - low_z], std_kedge[7, z - low_z] = sct.cnr(image10, vials[4], background)

    # Take the mean over all the slices
    mean_kedge = np.mean(mean_kedge, axis=1)
    std_kedge = np.mean(std_kedge, axis=1)

    # Save the matrices if desired
    #np.save(directory + folder + '/Mean_Kedge_CNR_Filter.npy', mean_kedge)
    #np.save(directory + folder + '/Std_Kedge_CNR_Filter.npy', std_kedge)

#%% Figure 2 Information Collection Signal

# Find the signal for the Bin Width Data

# Go through the bin width folders
for i, folder in enumerate(folders[11:]):
    vials = np.load(directory + folder + '/Vial_Masks.npy')  # Load the vial masks
    low_z, high_z = 11, 19  # Get the good slices

    # Initialize the matrices to hold the signal data
    # Orientation: [bin, slice, vial]
    mean_signal = np.zeros([5, high_z - low_z + 1, 5])
    std_signal = np.zeros([5, high_z - low_z + 1, 5])

    # Go through each bin
    for b in np.arange(5):
        # Go through the slices
        for j in np.arange(low_z, high_z+1):
            # Load the image
            image = np.load(directory + folder + '/Slices/Bin' + str(b) + '_Slice' + str(j) + '.npy')

            # This array goes in the order, 0%, 0.5% Au, 3% Au, 0.5% Gd, 3% Gd
            for idx, k in enumerate(np.array([0, 5, 1, 2, 4])):
                vial = vials[k]  # Choose the vial in the order listed in enumerate above
                mult_img = np.multiply(vial, image)  # Convolve to get data just in the specific vial
                mean_signal[b, j-low_z, idx] = np.nanmean(mult_img)  # Calculate the mean signal in the vial
                std_signal[b, j-low_z, idx] = np.nanstd(mult_img)  # Calculate the std of the signal

    # Average over the slices
    mean_signal = np.mean(mean_signal, axis=1)
    std_signal = np.mean(std_signal, axis=1)

    # Save the matrices if desired
    #np.save(directory + folder + '/Mean_Signal_BinWidth.npy', mean_signal)
    #np.save(directory + folder + '/Std_Signal_BinWidth.npy', std_signal)

#%% Figure 2 K-Edge CNR Collection
import sCT_Analysis as sct

# Get the K-Edge CNR in the first bin width folders

# Edges to easily call each of the file types
edges = ['4-3', '1-0']

# Go through each of the bin width folder
for i, folder in enumerate(folders[11:]):
    vials = np.load(directory + folder + '/Vial_Masks.npy')  # Load the vial masks of the folder
    background = np.load(directory + folder + '/Phantom_Mask.npy')  # Load the phantom background mask
    low_z, high_z = 11, 19  # The good slices

    # Initialize matrices to hold the K-edge CNR data
    # Orientation: [slice, vial (in order 0, 5, 1, 0, 2, 4)]
    mean_CNR = np.zeros([high_z - low_z + 1, 6])
    std_CNR = np.zeros([high_z - low_z + 1, 6])

    # Go through good slices
    for j in np.arange(low_z, high_z+1):
        image43 = np.load(directory + folder + '/K-Edge/Bin' + edges[0] + '_Slice' + str(j) + '.npy')  # Load 4-3 image
        image10 = np.load(directory + folder + '/K-Edge/Bin' + edges[1] + '_Slice' + str(j) + '.npy')  # Load 1-0 image

        # Go through the vials in the order below
        # This array goes in the order, 0% Au, 0.5% Au, 3% Au, 0% Gd, 0.5% Gd, 3% Gd
        for idx, k in enumerate(np.array([0, 5, 1, 0, 2, 4])):
            vial = vials[k]  # Grab the vial
            # if idx is less than 3, calculate the gold (4-3) vials
            if idx < 3:
                cnr1, cnr1_err = sct.cnr(image43, vial, background)  # Calculate the CNR and CNRerr of the vial
                mean_CNR[j-low_z, idx] = cnr1
                std_CNR[j-low_z, idx] = cnr1_err
            # otherwise calculate the gadolinium (1-0) vials
            else:
                cnr1, cnr1_err = sct.cnr(image10, vial, background)  # Calculate the CNR and CNRerr of the vial
                mean_CNR[j - low_z, idx] = cnr1
                std_CNR[j - low_z, idx] = cnr1_err

    # Average over all slices
    mean_CNR = np.mean(mean_CNR, axis=0)
    std_CNR = np.mean(std_CNR, axis=0)

    # Save the matrices if desired
    #np.save(directory + folder + '/Mean_Signal_BinWidth_CNR.npy', mean_CNR)
    #np.save(directory + folder + '/Std_Signal_BinWidth_CNR.npy', std_CNR)

#%% Figure 3
import sCT_Analysis as sct

# Get the CNR of the time optimization data

# Go through the time optimization folders
for i, folder in enumerate(np.concatenate(([folders[4]], folders[9:11]))):
    fig3_slices = [[7, 15], [10, 19], [10, 18]]  # The good slices for the three folders
    vials = np.load(directory + folder + '/Vial_Masks.npy')  # The vial masks
    background = np.load(directory + folder + '/Phantom_Mask.npy')  # Load the phantom background mask
    low_z, high_z = fig3_slices[i][0], fig3_slices[i][1]  # Grab the good slices for this folder

    # Initialize matrices to hold the CNR and noise data
    # Orientation for CNR data: [bin, slice, vial]
    # Orientation for the noise data: [bin, slice]
    mean_CNR = np.zeros([5, high_z - low_z + 1, 6])
    std_CNR = np.zeros([5, high_z - low_z + 1, 6])

    std_noise = np.zeros([5, high_z - low_z + 1])

    # Go through the bins
    for b in np.arange(5):
        # Go through the slices
        for j in np.arange(low_z, high_z+1):
            image = np.load(directory + folder + '/Slices/Bin' + str(b) + '_Slice' + str(j) + '.npy')  # Load the image
            # Go through the vials
            for k, vial in enumerate(vials):
                # Calculate the mean CNR and std of the CNR within the slice for that vial
                mean_CNR[b, j-low_z, k], std_CNR[b, j-low_z, k] = sct.cnr(image, vial, background)

            std_noise[b, j-low_z] = np.nanstd(image*background)

    # Average the CNR data over the slices
    mean_CNR = np.mean(mean_CNR, axis=1)
    std_CNR = np.mean(std_CNR, axis=1)

    # Calculate the mean of the noise and std of the noise over the slices
    mean_noise = np.mean(std_noise, axis=1)
    std_noise = np.std(std_noise, axis=1)

    # Save the matrices if desired
    #np.save(directory + folder + '/Mean_CNR_Time.npy', mean_CNR)
    #np.save(directory + folder + '/Std_CNR_Time.npy', std_CNR)

    #np.save(directory + folder + '/Mean_Noise_Time.npy', mean_noise)
    #np.save(directory + folder + '/Std_Noise_Time.npy', std_noise)


#%% Figure 3 K-Edge CNR
import sCT_Analysis as sct

# Get the K-edge CNR of the time optimization data

# Edges to easily call each of the file types
edges = ['4-3', '2-1', '3-2', '1-0']

# The good slices of the three folders
fig3_slices = [[12, 18], [10, 19], [10, 18]]

# Go through the 3 folders
for i, folder in enumerate(time_folders):
    low_z, high_z = 10, 18  # Get the good slices for this folder
    vials = np.load(directory + folder + '/Vial_Masks.npy')  # Vial masks
    background = np.load(directory + folder + '/Phantom_Mask.npy')  # Phantom mask

    # Initialize matrices to hold the K-edge CNR and noise data
    # Orientation: [0-3: CNR of water in K-edge image in descending order (4-3), (3-2), etc. 4-7: CNR of the specfio
    # element (follows the same order), slice]
    mean_kedge = np.empty([4, high_z - low_z + 1])
    std_kedge = np.empty([4, high_z - low_z + 1])

    # Go through the slices
    # Order: Au, Lu, Dy, Gd
    for z in np.arange(low_z, high_z + 1):

        # Calculate the K-edge CNR in the water vial of each K-edge image
        image43 = np.load(directory + folder + '/K-Edge/Bin' + edges[0] + '_Slice' + str(z) + '.npy')
        image32 = np.load(directory + folder + '/K-Edge/Bin' + edges[2] + '_Slice' + str(z) + '.npy')
        image21 = np.load(directory + folder + '/K-Edge/Bin' + edges[1] + '_Slice' + str(z) + '.npy')
        image10 = np.load(directory + folder + '/K-Edge/Bin' + edges[3] + '_Slice' + str(z) + '.npy')

        # Calculate the K-edge CNR in the element vial in the corresponding K-edge image
        mean_kedge[0, z - low_z], std_kedge[0, z - low_z] = sct.cnr(image43, vials[1], background)
        mean_kedge[1, z - low_z], std_kedge[1, z - low_z] = sct.cnr(image32, vials[3], background)
        mean_kedge[2, z - low_z], std_kedge[2, z - low_z] = sct.cnr(image21, vials[2], background)
        mean_kedge[3, z - low_z], std_kedge[3, z - low_z] = sct.cnr(image10, vials[4], background)


    # Average the data over the slices
    mean_kedge = np.mean(mean_kedge, axis=1)
    std_kedge = np.mean(std_kedge, axis=1)
    print(mean_kedge)
    print()
    print(std_kedge)
    print()
    print()
    # Save the matrices if desired
    np.save(directory + folder + '/Mean_Kedge_CNR_Time.npy', mean_kedge)
    np.save(directory + folder + '/Std_Kedge_CNR_Time.npy', std_kedge)

#%% Figure 4 K-Edge
import numpy as np
edges = ['4-3', '2-1', '3-2', '1-0']
for i, folder in enumerate(folders[0:9]):
    low_z, high_z = good_slices[i][0], good_slices[i][1]

    vials = np.load(directory + folder + '/Vial_Masks.npy')
    background = np.load(directory + folder + '/Phantom_Mask.npy')

    mean_kedge = np.empty([8, high_z-low_z+1])
    std_kedge = np.empty([8, high_z-low_z+1])


    for z in np.arange(low_z, high_z+1):
        image43 = np.load(directory + folder + '/K-Edge/Bin' + edges[0] + '_Slice' + str(z) + '.npy')
        image32 = np.load(directory + folder + '/K-Edge/Bin' + edges[2] + '_Slice' + str(z) + '.npy')
        image21 = np.load(directory + folder + '/K-Edge/Bin' + edges[1] + '_Slice' + str(z) + '.npy')
        image10 = np.load(directory + folder + '/K-Edge/Bin' + edges[3] + '_Slice' + str(z) + '.npy')


        mean_kedge[0, z - low_z] = np.nanmean(vials[0] * image43)
        mean_kedge[1, z - low_z] = np.nanmean(vials[0] * image32)
        mean_kedge[2, z - low_z] = np.nanmean(vials[0] * image21)
        mean_kedge[3, z - low_z] = np.nanmean(vials[0] * image10)

        std_kedge[0, z - low_z] = np.nanstd(vials[0] * image43)
        std_kedge[1, z - low_z] = np.nanstd(vials[0] * image32)
        std_kedge[2, z - low_z] = np.nanstd(vials[0] * image21)
        std_kedge[3, z - low_z] = np.nanstd(vials[0] * image10)

        mean_kedge[4, z - low_z] = np.nanmean(vials[1] * image43)
        mean_kedge[5, z - low_z] = np.nanmean(vials[3] * image32)
        mean_kedge[6, z - low_z] = np.nanmean(vials[2] * image21)
        mean_kedge[7, z - low_z] = np.nanmean(vials[4] * image10)

        std_kedge[4, z - low_z] = np.nanstd(vials[1] * image43)
        std_kedge[5, z - low_z] = np.nanstd(vials[3] * image32)
        std_kedge[6, z - low_z] = np.nanstd(vials[2] * image21)
        std_kedge[7, z - low_z] = np.nanstd(vials[4] * image10)

    mean_kedge = np.mean(mean_kedge, axis=1)
    std_kedge = np.mean(std_kedge, axis=1)

    # Save the matrices if desired
    #np.save(directory + folder + '/Mean_Kedge.npy', mean_kedge)
    #np.save(directory + folder + '/Std_Kedge.npy', std_kedge)