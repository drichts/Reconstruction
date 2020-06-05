import numpy as np
import matplotlib.pyplot as plt
import sCT_Analysis as sct
import generateROImask as grm
import general_OS_functions as gof

def analyze(filename, folder, directory='D:/Research/Python Data/Redlen/', re_analyze=False):

    data = np.load(directory + folder + filename)
    data = np.transpose(data, axes=(0, 3, 1, 2))

    test_image = data[12, 15]
    continue_flag = True

    if re_analyze:
        continue_flag = False
        masks = np.load(directory + folder + 'Vial_Masks.npy')
        phantom_mask = np.load(directory + folder + '/Phantom_Mask.npy')

    while continue_flag:
        masks = grm.phantom_ROIs(test_image, radius=7)
        val = input('Were the ROIs acceptable? (y/n)')
        if val is 'y':
            continue_flag = False

    np.save(directory + folder + '/Vial_Masks.npy', masks)

    continue_flag = True
    if re_analyze:
        continue_flag = False
    while continue_flag:
        phantom_mask = grm.entire_phantom(test_image)
        val = input('Were the ROIs acceptable? (y/n)')
        if val is 'y':
            continue_flag = False
    np.save(directory + folder + '/Phantom_Mask.npy', phantom_mask)

    water = masks[0]

    normal_data = np.zeros(np.shape(data))
    for i, energybin in enumerate(data):
        for j, image in enumerate(energybin):
            water_value = np.nanmean(image*water)
            #air_value = np.nanmean(image*air)
            normal_img = 1000*np.divide((np.subtract(image, water_value)), (water_value))
            normal_data[i, j] = normal_img

    np.save(directory + folder + '/Normalized_no_stripe_data_4.npy', normal_data)

analyze('no_stripe_images_4.npy', 'More tests/', re_analyze=True)

#%%
import numpy as np
b = 12
s = 14
directory = 'D:/Research/Python Data/Redlen/'
folder = 'More tests/'
#py = np.load(directory + folder + 'Normalized_data.npy')
py_full = np.load(directory + folder + 'Normalized_no_stripe_data_1.npy')

slices = np.arange(12, 18)

py_mean = np.zeros([7, len(slices)])
py_std = np.zeros([7, len(slices)])

mat_mean = np.zeros([7, len(slices)])
mat_std = np.zeros([7, len(slices)])

for j, s in enumerate(slices):
    py = py_full[b, int(s)]
    py_masks = np.load(directory + folder + 'Vial_Masks.npy')
    py_phantom_mask = np.load(directory + folder + 'Phantom_Mask.npy')

    mat = np.load(r'D:\Research\Python Data\Spectral CT\AuGd_width_14_12-2-19\Slices\Bin' + str(6) + '_Slice' + str(s) + '.npy')
    mat_masks = np.load(r'D:\Research\Python Data\Spectral CT\AuGd_width_14_12-2-19\Vial_Masks.npy')
    mat_phantom_mask = np.load(r'D:\Research\Python Data\Spectral CT\AuGd_width_14_12-2-19\Phantom_Mask.npy')

    for i in np.arange(6):
        py_mean[i, j] = np.nanmean(py * py_masks[i])
        py_std[i, j] = np.nanstd(py * py_masks[i])
        mat_mean[i, j] = np.nanmean(mat * mat_masks[i])
        mat_std[i, j] = np.nanstd(mat * mat_masks[i])

    py_mean[6, j] = np.nanmean(py*py_phantom_mask)
    py_std[6, j] = np.nanstd(py*py_phantom_mask)
    mat_mean[6, j] = np.nanmean(mat*mat_phantom_mask)
    mat_std[6, j] = np.nanstd(mat*mat_phantom_mask)

py_mean = np.mean(py_mean, axis=1)
py_std = np.mean(py_std, axis=1)
mat_mean = np.mean(mat_mean, axis=1)
mat_std = np.mean(mat_std, axis=1)

print(py_mean)
print(py_std)
print(mat_mean)
print(mat_std)


#%%
import matplotlib.pyplot as plt
import numpy as np
b = 5
s = 17
#py = np.load('D:/Research/Python Data/Redlen/More tests/Normalized_data.npy')
py = np.load('D:/Research/Python Data/Redlen/More tests/Normalized_no_stripe_data_1.npy')
py = py[b, s-1]
mat = np.load(r'D:\Research\Python Data\Spectral CT\AuGd_width_14_12-2-19\Slices\Bin' + str(b) + '_Slice' + str(s) + '.npy')

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
mn, mx = -500, 1200
ax[0].imshow(py, cmap='gray', vmin=mn, vmax=mx)
ax[0].set_title('Python', fontsize=18)
ax[0].axis('off')
ax[1].imshow(mat, cmap='gray', vmin=mn, vmax=mx)
ax[1].set_title('Matlab', fontsize=18)
ax[1].axis('off')

plt.show()
#%%
import glob

directory = 'D:/Research/Python Data/Redlen/Images/'
files = glob.glob(directory + 'spline*.npy')
masks = np.load(directory + 'Vial_Masks.npy')

for filename in files:
    dat = np.load(filename)
    dat = np.squeeze(dat)
    save = filename[-11:]
    dat = np.transpose(dat, axes=(2, 0, 1))

    water = masks[0]

    normal_dat = np.zeros(np.shape(dat))
    for i, energybin in enumerate(dat):
        for j, image in enumerate(energybin):
            water_value = np.nanmean(image * water)
            normal_img = sct.norm_individual(image, water_value)
            normal_dat[i, j] = normal_img

    np.save(directory + '/Normalized' + save, normal_dat)

#%%

s0 = np.load(directory + '/Normalizedspline0.npy')
s1 = np.load(directory + '/Normalizedspline1.npy')
s2 = np.load(directory + '/Normalizedspline2.npy')
s3 = np.load(r'D:\Research\Python Data\Redlen\Images\Python Filtered\Normalized_data.npy')
s3 = s3[12]
s4 = np.load(directory + '/Normalizedspline4.npy')
s5 = np.load(directory + '/Normalizedspline5.npy')

fig, ax = plt.subplots(2, 3, figsize=(10, 8))
mn, mx = -1000, 1000
ax[0, 0].imshow(s0[15], cmap='gray', vmin=mn, vmax=mx)
ax[0, 0].set_title('Sp0')
ax[0, 1].imshow(s1[15], cmap='gray', vmin=mn, vmax=mx)
ax[0, 1].set_title('Sp1')
ax[0, 2].imshow(s2[15], cmap='gray', vmin=mn, vmax=mx)
ax[0, 2].set_title('Sp2')
ax[1, 0].imshow(s3[15], cmap='gray', vmin=mn, vmax=mx)
ax[1, 0].set_title('Sp3')
ax[1, 1].imshow(s4[15], cmap='gray', vmin=mn, vmax=mx)
ax[1, 1].set_title('Sp4')
ax[1, 2].imshow(s5[15], cmap='gray', vmin=mn, vmax=mx)
ax[1, 2].set_title('Sp5')

for i in np.arange(6):
    print('Sp3: ' + str(np.nanmean(s3[15]*masks[i])) + '+-' + str(np.nanstd(s3[15]*masks[i])))
    print('Sp4: ' + str(np.nanmean(s4[15]*masks[i])) + '+-' + str(np.nanstd(s4[15]*masks[i])))
    print('Sp4: ' + str(np.nanmean(s5[15]*masks[i])) + '+-' + str(np.nanstd(s5[15]*masks[i])))
    print()
