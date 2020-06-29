import numpy as np
import sct_analysis as sct
import mask_functions as grm

directory = 'D:/Research/Python Data/Spectral CT/'

folders = ['Al_2.0_8-14-19', 'Al_2.0_10-17-19_3P', 'Al_2.0_10-17-19_1P',
           'Cu_0.5_8-14-19', 'Cu_0.5_9-13-19', 'Cu_0.5_10-17-19',
           'Cu_1.0_8-14-19', 'Cu_1.0_9-13-19', 'Cu_1.0_10-17-19',
           'Cu_0.5_Time_0.1_11-4-19', 'Frame2_Time_0.1', 'Frame3_Time_0.1', 'Frame4_Time_0.1', 'Frame5_Time_0.1',
           'Frame6_Time_0.1', 'Frame7_Time_0.1', 'Frame8_Time_0.1', 'Frame9_Time_0.1', 'Frame10_Time_0.1',
           'AuGd_width_5_12-2-19', 'AuGd_width_10_12-2-19', 'AuGd_width_14_12-2-19', 'AuGd_width_20_12-9-19']

old_folders = ['AuGd_width_5_12-2-19-with-stripe-removal', 'AuGd_width_10_12-2-19-with-stripe-removal',
               'AuGd_width_14_12-2-19-with-stripe-removal', 'AuGd_width_20_12-9-19-with-stripe-removal']

good_slices = [[5, 19], [10, 18], [11, 18],                         # 0
               [4, 15], [7, 15], [12, 19],                          # 3
               [4, 14], [5, 16], [10, 19],                          # 6
               [10, 18], [10, 18], [10, 18], [10, 18], [10, 18],    # 9
               [10, 18], [10, 18], [10, 18], [10, 18], [10, 18],    # 14
               [11, 19], [11, 19], [11, 19], [11, 19]]              # 19


def run_sct_main(reanalyze=False):
    for folder in folders:
        sct.main(folder, re_analyze=reanalyze)
    normed_kedge()
    figure_four()
    table_one()
    figure_six()
    figure_seven()
    figure_eight()


def get_background_mask():
    """
    Get the background mask based on the entire phantom, exluding the vial masks
    """

    # Go through each of the folders
    for i, folder in enumerate(folders[1:]):
        image = np.load(directory + folder + '/Slices/Bin6_Slice13.npy')

        continue_flag = True
        while continue_flag:
            phantom_mask = grm.entire_phantom(image)
            val = input('Were the ROIs acceptable? (y/n)')
            if val is 'y':
                continue_flag = False

        # Save if desired
        np.save(directory + folder + '/Phantom_Mask.npy', phantom_mask)


def lowest_noise():
    # Calculate the noise in each slice for all the "good" slices to find the lowest noise slice
    # Goes through each folder

    low_noise = np.empty([len(folders), 2])  # Slice number of the lowest noise for each folders and the noise value
    phantom_noise = np.empty([len(folders), 2])  # Noise of the phantom
    water_noise = np.empty([len(folders), 2])  # Noise in the water vial
    CNR = np.empty([len(folders), 6])  # CNR of all vials including the water vial

    for i, folder in enumerate(folders):
        low_z, high_z = good_slices[i][0], good_slices[i][1]

        # Current phantom mask
        p_mask = np.load(directory + folder + '/Phantom_Mask.npy')

        # Empty matrices to hold the noise and CNR values for each slice
        temp_noise = np.zeros(high_z-low_z+1)
        temp_CNR = np.zeros([high_z-low_z+1, 6])

        # Find the slice with the least noise
        low_noise[i] = sct.find_least_noise(folder, low_z, high_z)

        # Go through each slice and calc noise from std and then calc CNR
        for j in np.arange(low_z, high_z+1):
            image = np.load(directory + folder + '/Slices/Bin6_Slice' + str(j) + '.npy')
            temp_noise[j-low_z] = np.nanstd(image*p_mask)
            temp_CNR[j-low_z, :] = sct.get_ct_cnr(folder, j, type='phantom')

        print(temp_noise)

        # Calculate the average noise over the slices and the std of the noise as well
        phantom_noise[i, :] = np.array([np.nanmean(temp_noise), np.nanstd(temp_noise)])

        # Calculate the average CNR over the slices
        CNR[i] = np.mean(temp_CNR, axis=0)


def figure_four():
    # Figure 4 Information Collection

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
        np.save(directory + folder + '/Mean_Signal.npy', mean_signal)
        np.save(directory + folder + '/Std_Signal.npy', std_signal)


def table_one(z):
    # Table 1 K-Edge CNR

    # Get the K-Edge CNR in the first 9 folders

    # Edges to easily call each of the file types
    edges = ['4-3', '2-1', '3-2', '1-0']

    # Go through each of the folders
    for i, folder in enumerate(folders[0:9]):
        low_z, high_z = good_slices[i][0], good_slices[i][1]  # Get the specific good slices for this folder
        #low_z, high_z = z, z
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
        np.save(directory + folder + '/Mean_Kedge_CNR_Filter.npy', mean_kedge)
        np.save(directory + folder + '/Std_Kedge_CNR_Filter.npy', std_kedge)


def figure_six(folds=folders[19:]):
    # Figure 6 K-Edge CNR Collection

    # Get the K-Edge CNR in the first bin width folders

    # Edges to easily call each of the file types
    edges = ['4-3', '1-0']

    # Go through each of the bin width folder
    for i, folder in enumerate(folds):
        vials = np.load(directory + folder + '/Vial_Masks.npy')  # Load the vial masks of the folder
        background = np.load(directory + folder + '/Phantom_Mask.npy')  # Load the phantom background mask
        low_z, high_z = 11, 19
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
        np.save(directory + folder + '/Mean_Signal_BinWidth_CNR.npy', mean_CNR)
        np.save(directory + folder + '/Std_Signal_BinWidth_CNR.npy', std_CNR)


def figure_seven():
    # Figure 7 K-Edge CNR

    # Get the K-edge CNR of the time optimization data

    # Edges to easily call each of the file types
    edges = ['4-3', '2-1', '3-2', '1-0']

    # The good slices of the three folders
    fig3_slices = [[12, 18], [10, 19], [10, 18]]

    # Go through the 3 folders
    for i, folder in enumerate(folders[9:19]):
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


def figure_eight():
    # Figure 8 K-Edge

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
        np.save(directory + folder + '/Mean_Kedge.npy', mean_kedge)
        np.save(directory + folder + '/Std_Kedge.npy', std_kedge)


def normed_kedge():
    # Normalization values
    Al_norm = np.zeros([4, 2])
    Cu05_norm = np.zeros([4, 2])
    Cu1_norm = np.zeros([4, 2])

    for i in np.arange(4):
        # Order of the vials according to the bin (Gd (1-0), Dy (2-1), Lu (3-2), Au (4-3))
        vials = np.array([4, 2, 3, 1])

        # Find the norm values for each 5% value and 0% for each bin in each of the filters
        Al_norm[i, :] = sct.find_norm_value(folders[0], good_slices[0], vials[i], i, directory=directory)
        Cu05_norm[i, :] = sct.find_norm_value(folders[3], good_slices[3], vials[i], i, directory=directory)
        Cu1_norm[i, :] = sct.find_norm_value(folders[6], good_slices[6], vials[i], i, directory=directory)

        # Get the current linear fit coefficients for each filter
        coeffs_Al = sct.linear_fit(Al_norm[i, 0], Al_norm[i, 1])
        coeffs_Cu05 = sct.linear_fit(Cu05_norm[i, 0], Cu05_norm[i, 1])
        coeffs_Cu1 = sct.linear_fit(Cu1_norm[i, 0], Cu1_norm[i, 1])

        # Normalize the K-Edge images
        # Aluminum 2.0 mm
        sct.norm_kedge(folders[0], coeffs_Al, i, directory=directory)  # 5%
        sct.norm_kedge(folders[1], coeffs_Al, i, directory=directory)  # 3%
        sct.norm_kedge(folders[2], coeffs_Al, i, directory=directory)  # 1%

        # Copper 0.5 mm
        sct.norm_kedge(folders[3], coeffs_Cu05, i, directory=directory)  # 5%
        sct.norm_kedge(folders[4], coeffs_Cu05, i, directory=directory)  # 3%
        sct.norm_kedge(folders[5], coeffs_Cu05, i, directory=directory)  # 1%

        # Copper 1.0 mm
        sct.norm_kedge(folders[6], coeffs_Cu1, i, directory=directory)  # 5%
        sct.norm_kedge(folders[7], coeffs_Cu1, i, directory=directory)  # 3%
        sct.norm_kedge(folders[8], coeffs_Cu1, i, directory=directory)  # 1%

        # Time Acquisitions
        sct.norm_kedge(folders[9], coeffs_Cu05, i, directory=directory)  # 0.1s
        sct.norm_kedge(folders[10], coeffs_Cu05, i, directory=directory)  # 0.2s
        sct.norm_kedge(folders[11], coeffs_Cu05, i, directory=directory)  # 0.3s
        sct.norm_kedge(folders[12], coeffs_Cu05, i, directory=directory)  # 0.4s
        sct.norm_kedge(folders[13], coeffs_Cu05, i, directory=directory)  # 0.5s
        sct.norm_kedge(folders[14], coeffs_Cu05, i, directory=directory)  # 0.6s
        sct.norm_kedge(folders[15], coeffs_Cu05, i, directory=directory)  # 0.7s
        sct.norm_kedge(folders[16], coeffs_Cu05, i, directory=directory)  # 0.8s
        sct.norm_kedge(folders[17], coeffs_Cu05, i, directory=directory)  # 0.9s
        sct.norm_kedge(folders[18], coeffs_Cu05, i, directory=directory)  # 1.0s

        # Bin Width
        sct.norm_kedge(folders[19], coeffs_Cu05, i, directory=directory)  # 5, 5
        sct.norm_kedge(folders[20], coeffs_Cu05, i, directory=directory)  # 10, 10
        sct.norm_kedge(folders[21], coeffs_Cu05, i, directory=directory)  # 14, 14
        sct.norm_kedge(folders[22], coeffs_Cu05, i, directory=directory)  # 8, 20
