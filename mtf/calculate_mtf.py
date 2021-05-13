import os
import numpy as np
import mask_functions as msk

directory = r'D:\OneDrive - University of Victoria\Research\LDA Data'
# folder = '21-05-05_CT_metal_artifact'
folder = '21-05-05_CT_metal_30keV_bins'
sub = '3_metal'

bar_size = np.array([1, 0.75, 0.66, 0.5, 0.33, 0.25])
freq = 1 / (2 * bar_size)


def calc_mtf(folder=folder, sub=sub, use_contrast_vals=True):

    data = np.load(os.path.join(directory, folder, sub, 'Norm CT', 'CT_norm.npy'))[:, 11:14]
    air_mask = np.load(os.path.join(directory, folder, sub, 'air_mask.npy'))
    roi_masks = np.load(os.path.join(directory, folder, sub, 'masks_mtf.npy'))

    if not use_contrast_vals:
        phantom_mask = np.load(os.path.join(directory, folder, sub, 'phantom_mask_mtf.npy'))
    else:
        contrast_vals = np.load(os.path.join(directory, folder, 'water_only', 'contrast_mtf.npy'))

    num_bins = np.shape(data)[0]

    mtf_vals = np.zeros((num_bins, 2, 6))  # Shape <bins, mean and std, ROI (from largest to smallest)

    # Get the std vals
    for i in range(num_bins):

        mtf_final_z = []
        for z in range(3):
            temp_data = data[i, z]

            if not use_contrast_vals:
                air_val = np.nanmean(temp_data*air_mask)
                air_std = np.nanstd(temp_data*air_mask)

                phantom_val = np.nanmean(temp_data*phantom_mask)
                phantom_std = np.nanstd(temp_data*phantom_mask)

                contrast = np.abs(phantom_val - air_val)
            else:
                contrast = contrast_vals[i, 0]

            std_vals = np.zeros(6)
            for roi in range(6):
                std_vals[roi] = np.nanstd(temp_data*roi_masks[roi])

            # M' is the measured noise (STD) in each of the bar patterns, uncorrected
            M_prime = std_vals

            # squared noise using ROIs of air and plastic
            # N^2 = (Np^2+Nw^2)/2
            if not use_contrast_vals:
                noise_2 = (air_std ** 2 + phantom_std ** 2) / 2
            else:
                noise_2 = contrast_vals[i, 1]

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

    if not use_contrast_vals:
        np.save(os.path.join(directory, folder, sub, 'mtf.npy'), mtf_vals)
    else:
        np.save(os.path.join(directory, folder, sub, 'mtf_water_contrast.npy'), mtf_vals)


def calc_contrast_vals(folder=folder, sub=sub):

    data = np.load(os.path.join(directory, folder, sub, 'Norm CT', 'CT_norm.npy'))[:, 11:14]
    air_mask = np.load(os.path.join(directory, folder, sub, 'air_mask.npy'))
    phantom_mask = np.load(os.path.join(directory, folder, sub, 'phantom_mask_mtf.npy'))

    num_bins = np.shape(data)[0]

    contrast_vals = np.zeros((num_bins, 2, 3))  # Shape <bins, mean and std, slice

    # Get the std vals
    for i in range(num_bins):

        for z in range(3):
            temp_data = data[i, z]

            air_val = np.nanmean(temp_data * air_mask)
            air_std = np.nanstd(temp_data * air_mask)

            phantom_val = np.nanmean(temp_data * phantom_mask)
            phantom_std = np.nanstd(temp_data * phantom_mask)

            contrast_vals[i, 0, z] = np.abs(phantom_val - air_val)
            contrast_vals[i, 1, z] = (air_std ** 2 + phantom_std ** 2) / 2

    # Calculate the mean of the contrast and std along the slices axis
    contrast_vals = np.mean(contrast_vals, axis=2)

    np.save(os.path.join(directory, folder, sub, 'contrast_mtf.npy'), contrast_vals)


# calc_contrast_vals()
# calc_mtf(use_contrast_vals=False)
calc_mtf(use_contrast_vals=True)