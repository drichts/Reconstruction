import numpy as np
import matplotlib.pyplot as plt

directory = 'D:/Research/Python Data/Spectral CT/'

folder5_Al = 'Al_2.0_8-14-19'
folder3_Al = 'Al_2.0_10-17-19_3P'
folder1_Al = 'Al_2.0_10-17-19_1P'

title = 'Al 2.0mm'

# Open the files of all K-Edge images for Slice 15
img_1_0_5P = np.load(directory + folder5_Al + '/K-Edge/Bin1-0_Slice15.npy')
img_2_1_5P = np.load(directory + folder5_Al + '/K-Edge/Bin2-1_Slice15.npy')
img_3_2_5P = np.load(directory + folder5_Al + '/K-Edge/Bin3-2_Slice15.npy')
img_4_3_5P = np.load(directory + folder5_Al + '/K-Edge/Bin4-3_Slice15.npy')

img_1_0_3P = np.load(directory + folder3_Al + '/K-Edge/Bin1-0_Slice15.npy')
img_2_1_3P = np.load(directory + folder3_Al + '/K-Edge/Bin2-1_Slice15.npy')
img_3_2_3P = np.load(directory + folder3_Al + '/K-Edge/Bin3-2_Slice15.npy')
img_4_3_3P = np.load(directory + folder3_Al + '/K-Edge/Bin4-3_Slice15.npy')

img_1_0_1P = np.load(directory + folder1_Al + '/K-Edge/Bin1-0_Slice15.npy')
img_2_1_1P = np.load(directory + folder1_Al + '/K-Edge/Bin2-1_Slice15.npy')
img_3_2_1P = np.load(directory + folder1_Al + '/K-Edge/Bin3-2_Slice15.npy')
img_4_3_1P = np.load(directory + folder1_Al + '/K-Edge/Bin4-3_Slice15.npy')

# Grab the 4 ROI masks from each of the folders
# 5 Percent
img_size_5 = np.shape(img_1_0_5P)
ROI_5P = np.empty([5, img_size_5[0], img_size_5[1]])
for i in np.arange(5):
    x = np.load(directory + folder5_Al + '/Vial' + str(i) + '_MaskMatrix.npy')
    ROI_5P[i, :, :] = np.load(directory + folder5_Al + '/Vial' + str(i) + '_MaskMatrix.npy')

# 3 Percent
img_size_3 = np.shape(img_1_0_3P)
ROI_3P = np.empty([5, img_size_3[0], img_size_3[1]])
for j in np.arange(5):
    ROI_3P[j, :, :] = np.load(directory + folder3_Al + '/Vial' + str(j) + '_MaskMatrix.npy')

# 1 Percent
img_size_1 = np.shape(img_1_0_1P)
ROI_1P = np.empty([5, img_size_1[0], img_size_1[1]])
for k in np.arange(5):
    ROI_1P[k, :, :] = np.load(directory + folder1_Al + '/Vial' + str(k) + '_MaskMatrix.npy')

# Grab the mean value in the ROI that is visible
Au5 = img_4_3_5P * ROI_5P[1]
Au3 = img_4_3_3P * ROI_3P[1]
Au1 = img_4_3_1P * ROI_1P[1]
Au0 = img_4_3_5P * ROI_5P[0]
Au_5 = np.nanmean(Au5)
Au_3 = np.nanmean(Au3)
Au_1 = np.nanmean(Au1)
Au_0 = np.nanmean(Au0)

Dy5 = img_2_1_5P * ROI_5P[2]
Dy3 = img_2_1_3P * ROI_3P[2]
Dy1 = img_2_1_1P * ROI_1P[2]
Dy0 = img_2_1_5P * ROI_5P[0]
Dy_5 = np.nanmean(Dy5)
Dy_3 = np.nanmean(Dy3)
Dy_1 = np.nanmean(Dy1)
Dy_0 = np.nanmean(Dy0)

Lu5 = img_3_2_5P * ROI_5P[3]
Lu3 = img_3_2_3P * ROI_3P[3]
Lu1 = img_3_2_1P * ROI_1P[3]
Lu0 = img_3_2_5P * ROI_5P[0]
Lu_5 = np.nanmean(Lu5)
Lu_3 = np.nanmean(Lu3)
Lu_1 = np.nanmean(Lu1)
Lu_0 = np.nanmean(Lu0)

Gd5 = img_1_0_5P * ROI_5P[4]
Gd3 = img_1_0_3P * ROI_3P[4]
Gd1 = img_1_0_1P * ROI_1P[4]
Gd0 = img_1_0_5P * ROI_5P[0]
Gd_5 = np.nanmean(Gd5)
Gd_3 = np.nanmean(Gd3)
Gd_1 = np.nanmean(Gd1)
Gd_0 = np.nanmean(Gd0)
print(Gd_0, Gd_1, Gd_3, Gd_5)

# Normalize to between 0.2 and 1
x = np.array([Gd_0, Gd_1, Gd_3, Gd_5])
x = 0.8 * (np.subtract(x, np.min(x))/(np.max(x)-np.min(x))) + 0.2
Gd_0_new = x[0]
Gd_1_new = x[1]
Gd_3_new = x[2]
Gd_5_new = x[3]
print(Gd_0_new, Gd_1_new, Gd_3_new, Gd_5_new)

Au5_error = np.nanstd(Au5)
Au3_error = np.nanstd(Au3)
Au1_error = np.nanstd(Au1)
Au0_error = np.nanstd(Au0)
#print(Au1_error, Au3_error, Au5_error)

Dy5_error = np.nanstd(Dy5)
Dy3_error = np.nanstd(Dy3)
Dy1_error = np.nanstd(Dy1)
Dy0_error = np.nanstd(Dy0)

Lu5_error = np.nanstd(Lu5)
Lu3_error = np.nanstd(Lu3)
Lu1_error = np.nanstd(Lu1)
Lu0_error = np.nanstd(Lu0)

Gd5_error = np.nanstd(Gd5)
Gd3_error = np.nanstd(Gd3)
Gd1_error = np.nanstd(Gd1)
Gd0_error = np.nanstd(Gd0)

# Normalize to the 5 Percent value
Au0_norm = Au_0/Au_5
Au1_norm = Au_1/Au_5
Au3_norm = Au_3/Au_5
Au5_norm = Au_5/Au_5
#print(Au1_norm, Au3_norm, Au5_norm)

Dy0_norm = Dy_0/Dy_5
Dy1_norm = Dy_1/Dy_5
Dy3_norm = Dy_3/Dy_5
Dy5_norm = Dy_5/Dy_5

Lu0_norm = Lu_0/Lu_5
Lu1_norm = Lu_1/Lu_5
Lu3_norm = Lu_3/Lu_5
Lu5_norm = Lu_5/Lu_5

Gd0_norm = Gd_0/Gd_5
Gd1_norm = Gd_1/Gd_5
Gd3_norm = Gd_3/Gd_5
Gd5_norm = Gd_5/Gd_5
#Gd0_norm = Gd_0_new/Gd_5_new
#Gd1_norm = Gd_1_new/Gd_5_new
#Gd3_norm = Gd_3_new/Gd_5_new
#Gd5_norm = Gd_5_new/Gd_5_new
print(Gd0_norm, Gd1_norm, Gd3_norm, Gd5_norm)

Au0_error = np.abs(Au0_norm) * np.sqrt((Au0_error/Au_0)**2 + (Au5_error/Au_5)**2)
Au1_error = np.abs(Au1_norm) * np.sqrt((Au1_error/Au_1)**2 + (Au5_error/Au_5)**2)
Au3_error = np.abs(Au3_norm) * np.sqrt((Au3_error/Au_3)**2 + (Au5_error/Au_5)**2)
Au5_error = np.abs(Au5_norm) * np.sqrt((Au5_error/Au_5)**2 + (Au5_error/Au_5)**2)

Dy0_error = np.abs(Dy0_norm) * np.sqrt((Dy0_error/Dy_0)**2 + (Dy5_error/Dy_5)**2)
Dy1_error = np.abs(Dy1_norm) * np.sqrt((Dy1_error/Dy_1)**2 + (Dy5_error/Dy_5)**2)
Dy3_error = np.abs(Dy3_norm) * np.sqrt((Dy3_error/Dy_3)**2 + (Dy5_error/Dy_5)**2)
Dy5_error = np.abs(Dy5_norm) * np.sqrt((Dy5_error/Dy_5)**2 + (Dy5_error/Dy_5)**2)

Lu0_error = np.abs(Lu0_norm) * np.sqrt((Lu0_error/Lu_0)**2 + (Lu5_error/Lu_5)**2)
Lu1_error = np.abs(Lu1_norm) * np.sqrt((Lu1_error/Lu_1)**2 + (Lu5_error/Lu_5)**2)
Lu3_error = np.abs(Lu3_norm) * np.sqrt((Lu3_error/Lu_3)**2 + (Lu5_error/Lu_5)**2)
Lu5_error = np.abs(Lu5_norm) * np.sqrt((Lu5_error/Lu_5)**2 + (Lu5_error/Lu_5)**2)

Gd0_error = np.abs(Gd0_norm) * np.sqrt((Gd0_error/Gd_0)**2 + (Gd5_error/Gd_5)**2)
Gd1_error = np.abs(Gd1_norm) * np.sqrt((Gd1_error/Gd_1)**2 + (Gd5_error/Gd_5)**2)
Gd3_error = np.abs(Gd3_norm) * np.sqrt((Gd3_error/Gd_3)**2 + (Gd5_error/Gd_5)**2)
Gd5_error = np.abs(Gd5_norm) * np.sqrt((Gd5_error/Gd_5)**2 + (Gd5_error/Gd_5)**2)

#Gd0_error = np.abs(Gd0_norm) * np.sqrt((Gd0_error/Gd_0)**2 + (Gd5_error/Gd_5)**2)
#Gd1_error = np.abs(Gd1_norm) * np.sqrt((Gd1_error/Gd_1)**2 + (Gd5_error/Gd_5)**2)
#Gd3_error = np.abs(Gd3_norm) * np.sqrt((Gd3_error/Gd_3)**2 + (Gd5_error/Gd_5)**2)
#Gd5_error = np.abs(Gd5_norm) * np.sqrt((Gd5_error/Gd_5)**2 + (Gd5_error/Gd_5)**2)


# What the 1 and 3 percent values should be once normalized
predict_0P = 0
predict_1P = 0.2
predict_3P = 0.6
predict_5P = 1

# Calculate the RMSE
sq_dif_Au = (predict_0P - Au0_norm)**2 + (predict_1P - Au1_norm)**2 + (predict_3P - Au3_norm)**2 + (predict_5P - Au5_norm)**2
rmse_Au = np.sqrt(sq_dif_Au/2)
rmse_st_Au = '%0.3f' % rmse_Au

sq_dif_Dy = (predict_0P - Dy0_norm)**2 + (predict_1P - Dy1_norm)**2 + (predict_3P - Dy3_norm)**2 + (predict_5P - Dy5_norm)**2
rmse_Dy = np.sqrt(sq_dif_Dy/2)
rmse_st_Dy = '%0.3f' % rmse_Dy

sq_dif_Lu = (predict_0P - Lu0_norm)**2 + (predict_1P - Lu1_norm)**2 + (predict_3P - Lu3_norm)**2 + (predict_5P - Lu5_norm)**2
rmse_Lu = np.sqrt(sq_dif_Au/2)
rmse_st_Lu = '%0.3f' % rmse_Lu

sq_dif_Gd = (predict_0P - Gd0_norm)**2 + (predict_1P - Gd1_norm)**2 + (predict_3P - Gd3_norm)**2 + (predict_5P - Gd5_norm)**2
rmse_Gd = np.sqrt(sq_dif_Gd/2)
rmse_st_Gd = '%0.3f' % rmse_Gd


# Plot
xpts = np.linspace(-1, 6, 100)
ypts = np.multiply(xpts, 0.2)
plt.figure(figsize=(12, 8))
plt.errorbar([-0.2, 0.8, 2.8, 4.8], [Au0_norm, Au1_norm, Au3_norm, Au5_norm], yerr=[Au0_error, Au1_error, Au3_error, Au5_error], fmt='.', color='orange')
plt.errorbar([0, 1, 3, 5], [Dy0_norm, Dy1_norm, Dy3_norm, Dy5_norm], yerr=[Dy0_error, Dy1_error, Dy3_error, Dy5_error], fmt='.', color='blue')
plt.errorbar([0.2, 1.2, 3.2, 5.2], [Lu0_norm, Lu1_norm, Lu3_norm, Lu5_norm], yerr=[Lu0_error, Lu1_error, Lu3_error, Lu5_error], fmt='.', color='red')
plt.errorbar([0.4, 1.4, 3.4, 5.4], [Gd0_norm, Gd1_norm, Gd3_norm, Gd5_norm], yerr=[Gd0_error, Gd1_error, Gd3_error, Gd5_error], fmt='.', color='green')
plt.plot(xpts-0.2, ypts, color='orange')
plt.plot(xpts, ypts, color='blue')
plt.plot(xpts+0.2, ypts, color='red')
plt.plot(xpts+0.4, ypts, color='green')
plt.title(title, fontsize=40)
plt.xlabel('Contrast Concentration (%)', fontsize=30)
plt.ylabel('Normalized signal to 5%', fontsize=30)
plt.legend(['Au', 'Dy', 'Lu', 'Gd'], fontsize=25, loc='upper left')
plt.annotate('Au RMSE = ' + rmse_st_Au + '\nDy RMSE = ' + rmse_st_Dy + '\nLu RMSE = ' + rmse_st_Lu + '\nGd RMSE = ' +
             rmse_st_Gd, xy=(1, 0), xycoords='axes fraction', textcoords='offset pixels', xytext=(-10, 10),
             horizontalalignment='right', verticalalignment='bottom', fontsize=25)
plt.xlim([-0.5, 6])
#plt.ylim([-0.25, 1.2])
plt.tick_params(labelsize=25)

plt.show()

np.save(directory + 'Graphs/RMSE/' + title + 'Au_points.npy', [Au0_norm, Au1_norm, Au3_norm, Au5_norm])
np.save(directory + 'Graphs/RMSE/' + title + 'Au_error.npy', [Au0_error, Au1_error, Au3_error, Au5_error])

np.save(directory + 'Graphs/RMSE/' + title + 'Dy_points.npy', [Dy0_norm, Dy1_norm, Dy3_norm, Dy5_norm])
np.save(directory + 'Graphs/RMSE/' + title + 'Dy_error.npy', [Dy0_error, Dy1_error, Dy3_error, Dy5_error])

np.save(directory + 'Graphs/RMSE/' + title + 'Lu_points.npy', [Lu0_norm, Lu1_norm, Lu3_norm, Lu5_norm])
np.save(directory + 'Graphs/RMSE/' + title + 'Lu_error.npy', [Lu0_error, Lu1_error, Lu3_error, Lu5_error])

np.save(directory + 'Graphs/RMSE/' + title + 'Gd_points.npy', [Gd0_norm, Gd1_norm, Gd3_norm, Gd5_norm])
np.save(directory + 'Graphs/RMSE/' + title + 'Gd_error.npy', [Gd0_error, Gd1_error, Gd3_error, Gd5_error])



#%%
import numpy as np
import matplotlib.pyplot as plt


directory = 'D:/Research/Python Data/Spectral CT/Graphs/RMSE/npy points/'

title = 'Gd'

Al2 = np.load(directory + 'Al 2.0mm' + title + '_points.npy')
Cu5 = np.load(directory + 'Cu 0.5mm' + title + '_points.npy')
Cu1 = np.load(directory + 'Cu 1.0mm' + title + '_points.npy')

Al2_error = np.load(directory + 'Al 2.0mm' + title + '_error.npy')
Cu5_error = np.load(directory + 'Cu 0.5mm' + title + '_error.npy')
Cu1_error = np.load(directory + 'Cu 1.0mm' + title + '_error.npy')

# What the 1 and 3 percent values should be once normalized
predict_0P = 0
predict_1P = 0.2
predict_3P = 0.6
predict_5P = 1

# Calculate the RMSE
sq_dif_Al = (predict_0P - Al2[0])**2 + (predict_1P - Al2[1])**2 + (predict_3P - Al2[2])**2 + (predict_5P - Al2[3])**2
rmse_Al = np.sqrt(sq_dif_Al/2)
rmse_st_Al = '%0.3f' % rmse_Al

sq_dif_Cu5 = (predict_0P - Cu5[0])**2 + (predict_1P - Cu5[1])**2 + (predict_3P - Cu5[2])**2 + (predict_5P - Cu5[3])**2
rmse_Cu5 = np.sqrt(sq_dif_Cu5/2)
rmse_st_Cu5 = '%0.3f' % rmse_Cu5

sq_dif_Cu1 = (predict_0P - Cu1[0])**2 + (predict_1P - Cu1[1])**2 + (predict_3P - Cu1[2])**2 + (predict_5P - Cu1[3])**2
rmse_Cu1 = np.sqrt(sq_dif_Cu1/2)
rmse_st_Cu1 = '%0.3f' % rmse_Cu1


# Plot
xpts = np.linspace(-1, 6, 100)
ypts = np.multiply(xpts, 0.2)
plt.figure(figsize=(12, 8))
plt.errorbar([-0.2, 0.8, 2.8, 4.8], Al2, yerr=Al2_error, fmt='.', color='orange')
plt.errorbar([0, 1, 3, 5], Cu5, yerr=Cu5_error, fmt='.', color='blue')
plt.errorbar([0.2, 1.2, 3.2, 5.2], Cu1, yerr=Cu1_error, fmt='.', color='red')
plt.plot(xpts-0.2, ypts, color='orange')
plt.plot(xpts, ypts, color='blue')
plt.plot(xpts+0.2, ypts, color='red')
plt.title(title, fontsize=40)
plt.xlabel('Contrast Concentration (%)', fontsize=30)
plt.ylabel('Normalized signal to 5%', fontsize=30)
plt.legend(['Al 2.0mm', 'Cu 0.5mm', 'Cu 1.0mm'], fontsize=25, loc='upper left')
plt.annotate('Al 2.0 RMSE = ' + rmse_st_Al + '\nCu 0.5 RMSE = ' + rmse_st_Cu5 + '\nCu 1.0 RMSE = ' + rmse_st_Cu1,
             xy=(1, 0), xycoords='axes fraction', textcoords='offset pixels', xytext=(-10, 10),
             horizontalalignment='right', verticalalignment='bottom', fontsize=25)
#plt.xlim([-0.5, 6])
#plt.ylim([-0.25, 1.2])
plt.tick_params(labelsize=25)

plt.show()