import os
import numpy as np
import mask_functions as msk

directory = r'D:\OneDrive - University of Victoria\Research\LDA Data'
folder = '21-04-14_CT_bin_width_10'

# sub = 'water_phantom'
sub = 'metal_phantom'

data = np.load(os.path.join(directory, folder, sub, 'Norm CT', 'CT_norm.npy'))[:, 11:14]


air_mask = np.load(os.path.join(directory, folder, sub, 'air_mask.npy'))
phantom_mask = np.load(os.path.join(directory, folder, 'phantom_mask_mtf_metal.npy'))
roi_masks = np.load(os.path.join(directory, folder, 'masks_mtf_metal.npy'))

bar_size = np.array([1, 0.75, 0.66, 0.5, 0.33, 0.25])
freq = 1 / (2 * bar_size)

mtf_vals = np.zeros((6, 2, 6))  # Shape <bins, mean and std, ROI (from largest to smallest)

# Get the std vals
for i in range(6):

    mtf_final_z = []
    for z in range(3):
        temp_data = data[i, z]
        air_val = np.nanmean(temp_data*air_mask)
        air_std = np.nanstd(temp_data*air_mask)

        phantom_val = np.nanmean(temp_data*phantom_mask)
        phantom_std = np.nanstd(temp_data*phantom_mask)

        contrast = np.abs(phantom_val - air_val)
        std_vals = np.zeros(6)
        for roi in range(6):
            std_vals[roi] = np.nanstd(temp_data*roi_masks[roi])

        # M' is the measured noise (STD) in each of the bar patterns, uncorrected
        M_prime = std_vals

        # squared noise using ROIs of air and plastic
        # N^2 = (Np^2+Nw^2)/2
        noise_2 = (air_std ** 2 + phantom_std ** 2) / 2
        # correct std of bar patterns for noise using the water ROI, M = sqrt(M'^2-N^2)
        M = []
        for std in M_prime:
            if std ** 2 > noise_2:
                M.append(np.sqrt(std ** 2 - noise_2))
            else:
                M.append(0)


        # M0 = (CT1-CT2)/2: |Air - Plastic|/2
        M0 = contrast / 2

        # MTF = (pi*sqrt(2)/4)*(M/M0)
        MTF = (np.pi * np.sqrt(2) / 4) * (M / M0)
        mtf_final_z.append(MTF)

    mtf_final_z = np.array(mtf_final_z)

    mtf_vals[i, 0] = np.mean(mtf_final_z, axis=0)
    mtf_vals[i, 1] = np.std(mtf_final_z, axis=0)

np.save(os.path.join(directory, folder, sub, 'mtf.npy'), mtf_vals)