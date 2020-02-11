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

good_slices = [[5, 19], [10, 18], [11, 18],
               [4, 15], [7, 15], [12, 19],
               [4, 14], [5, 16], [10, 19],
               [10, 19], [10, 18],
               [11, 19], [11, 19], [11, 19], [11, 19]]

#%% Phantom masks

#for i, folder in enumerate([folders[4]]):
#    image = np.load(directory + folder + '/Slices/Bin6_Slice11.npy')

#    continue_flag = True
#    while continue_flag:
#        phantom_mask = grm.entire_phantom(image)
#        val = input('Were the ROIs acceptable? (y/n)')
#        if val is 'y':
#            continue_flag = False

#    np.save(directory + folder + '/Phantom_Mask.npy', phantom_mask)

#%%
#low_noise = np.empty([len(folders), 2])  # Slice number of the lowest noise for each folders and the noise value
#phantom_noise = np.empty([len(folders), 2])
#water_noise = np.empty([len(folders), 2])
#CNR = np.empty([len(folders), 6])  # CNR of all vials with water vial

#for i, folder in enumerate(folders):
    #low_z, high_z = good_slices[i][0], good_slices[i][1]

    #p_mask = np.load(directory + folder + '/Phantom_Mask.npy')
    #temp_noise = np.zeros(high_z-low_z+1)
    #temp_CNR = np.zeros([high_z-low_z+1, 6])
    #low_noise[i] = sct.find_least_noise(folder, low_z, high_z)
    #for j in np.arange(low_z, high_z+1):
    #    image = np.load(directory + folder + '/Slices/Bin6_Slice' + str(j) + '.npy')
    #    temp_noise[j-low_z] = np.nanstd(image*p_mask)
    #    temp_CNR[j-low_z, :] = sct.get_ct_cnr(folder, j, type='phantom')

    #print(temp_noise)
    #phantom_noise[i, :] = np.array([np.nanmean(temp_noise), np.nanstd(temp_noise)])

    #water_noise[i] = sct.total_image_noise_stats(image, folder, load=True)


    #CNR[i] = np.mean(temp_CNR, axis=0)



#%% Create background masks
#for i, folder in enumerate(folders):
#    low_z, high_z = good_slices[i][0], good_slices[i][1]

    #img = np.load(directory + folder + '/Slices/Bin6_Slice' + str(low_z) + '.npy')

    #continue_flag = True
    #while continue_flag:
    #    back = grm.background_ROI(img)
    #    val = input('Were the ROIs acceptable? (y/n)')
    #    if val is 'y':
    #        continue_flag = False
    #np.save(directory + folder + '/Background_Mask.npy', back)

#%% Calculate CNR of K-edge Images

#lan_CNR = np.empty([11, 4])
#AuGd_CNR = np.empty([4, 6])

#for i, folder in enumerate(folders[0:11]):
#    low_z, high_z = good_slices[i][0], good_slices[i][1]
#    low_noise[i] = sct.find_least_noise(folder, low_z, high_z)
#    vials = np.load(directory + folder + '/Vial_Masks.npy')

#    img1_0 = np.load(directory + folder + '/K-Edge/Bin1-0_Slice' + str(int(low_noise[i][0])) + '.npy')
#    img2_1 = np.load(directory + folder + '/K-Edge/Bin2-1_Slice' + str(int(low_noise[i][0])) + '.npy')
#    img3_2 = np.load(directory + folder + '/K-Edge/Bin3-2_Slice' + str(int(low_noise[i][0])) + '.npy')
#    img4_3 = np.load(directory + folder + '/K-Edge/Bin4-3_Slice' + str(int(low_noise[i][0])) + '.npy')

#    au = sct.cnr(img4_3, vials[1], vials[0])
#    dy = sct.cnr(img2_1, vials[2], vials[0])
#    lu = sct.cnr(img3_2, vials[3], vials[0])
#    gd = sct.cnr(img1_0, vials[4], vials[0])

#    lan_CNR[i, :] = np.array([au, dy, lu, gd])

#for i, folder in enumerate(folders[11:]):
#    low_z, high_z = good_slices[i][0], good_slices[i][1]
#    low_noise[i] = sct.find_least_noise(folder, low_z, high_z)
#    vials = np.load(directory + folder + '/Vial_Masks.npy')

#    img1_0 = np.load(directory + folder + '/K-Edge/Bin1-0_Slice' + str(15) + '.npy')
#    img4_3 = np.load(directory + folder + '/K-Edge/Bin4-3_Slice' + str(15) + '.npy')

#    au3 = sct.cnr(img4_3, vials[1], vials[0])
#    au05 = sct.cnr(img4_3, vials[5], vials[0])
#    aum = sct.cnr(img4_3, vials[3], vials[0])
#    gd3 = sct.cnr(img1_0, vials[4], vials[0])
#    gd05 = sct.cnr(img1_0, vials[2], vials[0])
#    gdm = sct.cnr(img1_0, vials[3], vials[0])

#    AuGd_CNR[i, :] = np.array([au3, au05, aum, gd3, gd05, gdm])

#%% Get airscan flux
flux = np.empty(9)

for i, folder in enumerate(folders[0:9]):
    flux[i] = sct.airscan_flux(folder)[6]

#%% Figure 1 Information Collection

for i, folder in enumerate(folders[0:9]):
    vials = np.load(directory + folder + '/Vial_Masks.npy')
    low_z, high_z = good_slices[i][0], good_slices[i][1]

    mean_signal = np.zeros([5, high_z-low_z+1, 6])
    std_signal = np.zeros([5, high_z-low_z+1, 6])
    for b in np.arange(5):
        for j in np.arange(low_z, high_z+1):
            image = np.load(directory + folder + '/Slices/Bin' + str(b) + '_Slice' + str(j) + '.npy')
            for k, vial in enumerate(vials):
                mult_img = np.multiply(vial, image)
                mean_signal[b, j-low_z, k] = np.nanmean(mult_img)
                std_signal[b, j-low_z, k] = np.nanstd(mult_img)

    # Final arrays are of the form bin by vial
    #mean_signal = np.mean(mean_signal, axis=1)
    #std_signal = np.mean(std_signal, axis=1)
    #np.save(directory + folder + '/Mean_Signal.npy', mean_signal)
    #np.save(directory + folder + '/Std_Signal.npy', std_signal)

#%% Figure 1 K-Edge CNR

import sCT_Analysis as sct

edges = ['4-3', '2-1', '3-2', '1-0']

for i, folder in enumerate(folders[0:9]):
    low_z, high_z = good_slices[i][0], good_slices[i][1]

    vials = np.load(directory + folder + '/Vial_Masks.npy')
    background = np.load(directory + folder + '/Phantom_Mask.npy')

    mean_kedge = np.empty([8, high_z - low_z + 1])
    std_kedge = np.empty([8, high_z - low_z + 1])

    # Au, Lu, Dy, Gd
    for z in np.arange(low_z, high_z + 1):
        image43 = np.load(directory + folder + '/K-Edge/Bin' + edges[0] + '_Slice' + str(z) + '.npy')
        image32 = np.load(directory + folder + '/K-Edge/Bin' + edges[2] + '_Slice' + str(z) + '.npy')
        image21 = np.load(directory + folder + '/K-Edge/Bin' + edges[1] + '_Slice' + str(z) + '.npy')
        image10 = np.load(directory + folder + '/K-Edge/Bin' + edges[3] + '_Slice' + str(z) + '.npy')

        mean_kedge[0, z - low_z], std_kedge[0, z - low_z] = sct.cnr(image43, vials[0], background)
        mean_kedge[1, z - low_z], std_kedge[1, z - low_z] = sct.cnr(image32, vials[0], background)
        mean_kedge[2, z - low_z], std_kedge[2, z - low_z] = sct.cnr(image21, vials[0], background)
        mean_kedge[3, z - low_z], std_kedge[3, z - low_z] = sct.cnr(image10, vials[0], background)

        mean_kedge[4, z - low_z], std_kedge[4, z - low_z] = sct.cnr(image43, vials[1], background)
        mean_kedge[5, z - low_z], std_kedge[5, z - low_z] = sct.cnr(image32, vials[3], background)
        mean_kedge[6, z - low_z], std_kedge[6, z - low_z] = sct.cnr(image21, vials[2], background)
        mean_kedge[7, z - low_z], std_kedge[7, z - low_z] = sct.cnr(image10, vials[4], background)

    mean_kedge = np.mean(mean_kedge, axis=1)
    std_kedge = np.mean(std_kedge, axis=1)

    #np.save(directory + folder + '/Mean_Kedge_CNR_Filter.npy', mean_kedge)
    #np.save(directory + folder + '/Std_Kedge_CNR_Filter.npy', std_kedge)

#%% Figure 2 Information Collection Signal

for i, folder in enumerate(folders[11:]):
    vials = np.load(directory + folder + '/Vial_Masks.npy')
    low_z, high_z = 11, 19

    mean_signal = np.zeros([5, high_z - low_z + 1, 5])
    std_signal = np.zeros([5, high_z - low_z + 1, 5])
    for b in np.arange(5):
        for j in np.arange(low_z, high_z+1):
            image = np.load(directory + folder + '/Slices/Bin' + str(b) + '_Slice' + str(j) + '.npy')

            # This array goes in the order, 0%, 0.5% Au, 3% Au, 0.5% Gd, 3% Gd
            for idx, k in enumerate(np.array([0, 5, 1, 2, 4])):
                vial = vials[k]
                mult_img = np.multiply(vial, image)
                mean_signal[b, j-low_z, idx] = np.nanmean(mult_img)
                std_signal[b, j-low_z, idx] = np.nanstd(mult_img)

    # Average over all slices
    mean_signal = np.mean(mean_signal, axis=1)
    std_signal = np.mean(std_signal, axis=1)

    #np.save(directory + folder + '/Mean_Signal_BinWidth.npy', mean_signal)
    #np.save(directory + folder + '/Std_Signal_BinWidth.npy', std_signal)

#%% Figure 2 K-Edge CNR Collection
import sCT_Analysis as sct

edges = ['4-3', '1-0']
for i, folder in enumerate(folders[11:]):
    vials = np.load(directory + folder + '/Vial_Masks.npy')
    background = np.load(directory + folder + '/Phantom_Mask.npy')
    low_z, high_z = 11, 19

    mean_CNR = np.zeros([high_z - low_z + 1, 6])
    std_CNR = np.zeros([high_z - low_z + 1, 6])

    for j in np.arange(low_z, high_z+1):
        image43 = np.load(directory + folder + '/K-Edge/Bin' + edges[0] + '_Slice' + str(j) + '.npy')
        image10 = np.load(directory + folder + '/K-Edge/Bin' + edges[1] + '_Slice' + str(j) + '.npy')

        # This array goes in the order, 0% Au, 0.5% Au, 3% Au, 0% Gd, 0.5% Gd, 3% Gd
        for idx, k in enumerate(np.array([0, 5, 1, 0, 2, 4])):
            vial = vials[k]
            if idx < 3:
                cnr1, cnr1_err = sct.cnr(image43, vial, background)
                mean_CNR[j-low_z, idx] = cnr1
                std_CNR[j-low_z, idx] = cnr1_err
            else:
                cnr1, cnr1_err = sct.cnr(image10, vial, background)
                mean_CNR[j - low_z, idx] = cnr1
                std_CNR[j - low_z, idx] = cnr1_err

    # Average over all slices
    mean_CNR = np.mean(mean_CNR, axis=0)
    std_CNR = np.mean(std_CNR, axis=0)

    #np.save(directory + folder + '/Mean_Signal_BinWidth_CNR.npy', mean_CNR)
    #np.save(directory + folder + '/Std_Signal_BinWidth_CNR.npy', std_CNR)

#%% Figure 3

import sCT_Analysis as sct

for i, folder in enumerate(np.concatenate(([folders[4]], folders[9:11]))):
    fig3_slices = [[7, 15], [10, 19], [10, 18]]
    vials = np.load(directory + folder + '/Vial_Masks.npy')
    background = np.load(directory + folder + '/Phantom_Mask.npy')
    low_z, high_z = fig3_slices[i][0], fig3_slices[i][1]

    mean_CNR = np.zeros([5, high_z - low_z + 1, 6])
    std_CNR = np.zeros([5, high_z - low_z + 1, 6])

    std_noise = np.zeros([5, high_z - low_z + 1])

    for b in np.arange(5):
        for j in np.arange(low_z, high_z+1):
            image = np.load(directory + folder + '/Slices/Bin' + str(b) + '_Slice' + str(j) + '.npy')
            for k, vial in enumerate(vials):
                mean_CNR[b, j-low_z, k], std_CNR[b, j-low_z, k] = sct.cnr(image, vial, background)

            std_noise[b, j-low_z] = np.nanstd(image*background)

    mean_CNR = np.mean(mean_CNR, axis=1)
    std_CNR = np.mean(std_CNR, axis=1)

    mean_noise = np.mean(std_noise, axis=1)
    std_noise = np.std(std_noise, axis=1)

    #np.save(directory + folder + '/Mean_CNR_Time.npy', mean_CNR)
    #np.save(directory + folder + '/Std_CNR_Time.npy', std_CNR)

    #np.save(directory + folder + '/Mean_Noise_Time.npy', mean_noise)
    #np.save(directory + folder + '/Std_Noise_Time.npy', std_noise)

#%% Figure 3 K-Edge CNR
import sCT_Analysis as sct

edges = ['4-3', '2-1', '3-2', '1-0']
fig3_slices = [[7, 15], [10, 19], [10, 18]]
for i, folder in enumerate(np.concatenate(([folders[4]], folders[9:11]))):
    low_z, high_z = fig3_slices[i][0], fig3_slices[i][1]

    vials = np.load(directory + folder + '/Vial_Masks.npy')
    background = np.load(directory + folder + '/Phantom_Mask.npy')

    mean_kedge = np.empty([4, high_z - low_z + 1])
    std_kedge = np.empty([4, high_z - low_z + 1])

    # Order: Au, Lu, Dy, Gd
    for z in np.arange(low_z, high_z + 1):
        image43 = np.load(directory + folder + '/K-Edge/Bin' + edges[0] + '_Slice' + str(z) + '.npy')
        image32 = np.load(directory + folder + '/K-Edge/Bin' + edges[2] + '_Slice' + str(z) + '.npy')
        image21 = np.load(directory + folder + '/K-Edge/Bin' + edges[1] + '_Slice' + str(z) + '.npy')
        image10 = np.load(directory + folder + '/K-Edge/Bin' + edges[3] + '_Slice' + str(z) + '.npy')

        mean_kedge[0, z - low_z], std_kedge[0, z - low_z] = sct.cnr(image43, vials[1], background)
        mean_kedge[1, z - low_z], std_kedge[1, z - low_z] = sct.cnr(image32, vials[3], background)
        mean_kedge[2, z - low_z], std_kedge[2, z - low_z] = sct.cnr(image21, vials[2], background)
        mean_kedge[3, z - low_z], std_kedge[3, z - low_z] = sct.cnr(image10, vials[4], background)

    mean_kedge = np.mean(mean_kedge, axis=1)
    std_kedge = np.mean(std_kedge, axis=1)

    #np.save(directory + folder + '/Mean_Kedge_CNR_Time.npy', mean_kedge)
    #np.save(directory + folder + '/Std_Kedge_CNR_Time.npy', std_kedge)

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

    #np.save(directory + folder + '/Mean_Kedge.npy', mean_kedge)
    #np.save(directory + folder + '/Std_Kedge.npy', std_kedge)