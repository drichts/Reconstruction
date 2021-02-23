import os
import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import general_functions as gen

directory = '/home/knoll/LDAData/21-02-18_CT_water_only'

d1 = np.load(os.path.join('/home/knoll/LDAData', 'dead_pixel_mask.npy'))
# d2 = np.load(os.path.join(directory, 'dead_pixel_mask_2.npy'))
# d3 = np.load(os.path.join(directory, 'dead_pixel_mask_greater1per.npy'))

# fig, ax = plt.subplots(3, 1, figsize=(10, 8))
# ax[0].imshow(d1)
# ax[1].imshow(d2)
# ax[2].imshow(d3)
# plt.show()


folder1 = 'airscan_60s_gold'
folder2 = 'airscan_60s_gold_1'
dark = np.load(os.path.join(directory, 'darkscan_60s_gold/Data/data.npy'))

air1 = np.load(os.path.join(directory, folder1, 'Data', 'data.npy'))
air2 = np.load(os.path.join(directory, folder2, 'Data', 'data.npy'))

dpm = np.load('/home/knoll/LDAData/dead_pixel_mask_2.npy')

air1 = gen.correct_dead_pixels(air1, dpm)
air2 = gen.correct_dead_pixels(air2, dpm)
dark = gen.correct_dead_pixels(dark, dpm)

air1 = air1 - dark
air2 = air2 - dark
num = 2
corr = np.abs(np.log(air1) - np.log(air2)) * 100
# fig = plt.figure(figsize=(12, 4))
# plt.imshow(corr[:, :, num], vmin=1, vmax=2)
# plt.show()

# base = np.array(np.argwhere(corr > 2), dtype='int')
# base_nan = np.array(np.argwhere(np.isnan(corr)), dtype='int')
#
# dpm = np.ones((24, 576, 7))
#
# for pixel in base:
#     pixel = tuple(pixel)
#     dpm[pixel] = np.nan
# for pixel in base_nan:
#     pixel = tuple(pixel)
#     dpm[pixel] = np.nan
#
# np.save('/home/knoll/LDAData/dead_pixel_mask_2.npy', dpm)
for i in range(7):
    print(len(np.argwhere(corr[:, :, i] > 2)))
    print(len(np.argwhere(np.isnan(corr[:, :, i]))))
    print((len(np.argwhere(corr[:, :, i] > 2)) + len(np.argwhere(np.isnan(corr[:, :, i])))) / (24*576) * 100)
#
# coords = np.argwhere(corr[:, :, 6] > 2)
# nancoord = np.argwhere(np.isnan(corr[:, :, 6]))
#
# # folder = r'D:\OneDrive - University of Victoria\Research\LDA Data'
# # s1 = 'airscan_120kVp_2mA_1mmAl_60s'
# # s2 = 'large_phant_120kVp_2mA_1mmAl_top'
# #
# # f1 = os.path.join(folder, s1, 'Data/data.npy')
# # f2 = os.path.join(folder, s2, 'Data/data.npy')
# #
# # air = np.load(f1)
# # data = np.load(f2)
# #
# # proj = np.log(air) - np.log(data)
# # proj = np.sum(proj, axis=0)
# # # proj = np.load(os.path.join(folder, s2, 'Data', 'data_corr.npy'))[:, :, :, 6]
# #
# # deads = np.argwhere(np.isnan(proj[:, :, 6]))
#
# mask = np.ones((24, 576), dtype='float')
#
# for c in coords:
#     mask[c[0], c[1]] = np.nan
#
# for n in nancoord:
#     mask[n[0], n[1]] = np.nan
#
# for d in np.load(r'D:\OneDrive - University of Victoria\Research\LDA Data\extra_pixels.npy'):
#     mask[d[0], d[1]] = np.nan
#
# mask[13, 493] = np.nan
# mask[9, 173] = np.nan
# mask[2, 216] = np.nan
# mask[15, 300] = np.nan
# mask[15, 301] = np.nan
# mask[22, 326] = np.nan
# mask[0, 358] = np.nan
# mask[2, 360] = np.nan
# mask[23, 538] = np.nan
# mask[21, 542] = np.nan
#
# mask[23, 177] = np.nan
# mask[23, 178] = np.nan
# mask[23, 179] = np.nan
# mask[23, 180] = np.nan
# mask[23, 181] = np.nan
# mask[23, 182] = np.nan


# np.save(r'D:\OneDrive - University of Victoria\Research\LDA Data\dead_pixel_mask_2.npy', mask)
