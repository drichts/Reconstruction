import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns

# Things to enter
save_nm = 'I_Au_5-17'  # Save name to append
extra_append = '_Gold'  # Can be left blank
folder = 'I_Au_5_17_19'  # Folder name
lower_bin, upper_bin = 3, 4  # Upper and lower bins (upper-lower)
vials = 6  # Number of ROIs
x = [0, 1, 1, 5]  # Concentrations

# Load the two bin matrices in question
s_lower = np.load(folder + '/Bin' + str(lower_bin) + '_Matrix_' + save_nm + '.npy')
s_upper = np.load(folder + '/Bin' + str(upper_bin) + '_Matrix_' + save_nm + '.npy')

# Get the K-edge subtraction
k_image = np.subtract(s_upper, s_lower)

# Normalize to get rid of negative values
rng = np.ptp(k_image)
min_val = np.min(k_image)
k_image = np.divide(np.subtract(k_image, min_val), rng)

# Show the image
plt.imshow(k_image, cmap='gray', vmin=0.45)
plt.savefig(folder + '/K-Edge_Image_' + save_nm + extra_append + '.png', dpi=500)
plt.close()

# Load all of the ROIs
v = np.empty([vials, 120, 120])
for k in np.arange(vials):
    v[k, :, :] = np.load(folder + '/Vial' + str(k+1) + '_MaskMatrix_' + save_nm + '.npy')
back = np.load(folder + '/BackgroundMaskMatrix_' + save_nm + '.npy')  # Background ROI matrix

# Calculate the CNR of each ROI

# Std. Dev. of background
sig_bg = np.nanstd(back*k_image)

# Average background reading
bg = np.nanmean(back*k_image)

# Mean signal and std dev from each vial
mean = np.empty(vials)
stddev = np.empty(vials)
for j in np.arange(vials):
    temp = k_image*v[j]
    mean[j] = np.nanmean(temp)
    stddev[j] = np.nanstd(temp)

# Calculate the CNR of each vial
CNR = np.divide(np.subtract(mean, bg), sig_bg)

# CHANGE THIS IF YOU ONLY WANT TO SELECT A CERTAIN NUMBER OF VIALS
#mean = np.concatenate((mean[0:2], mean[4:]))
#stddev = np.concatenate((stddev[0:2], stddev[4:]))
#CNR = np.concatenate((CNR[0:2], CNR[4:]))
mean = mean[0:4]
stddev = stddev[0:4]
CNR = CNR[0:4]

# Construct data for a linear fit of the data
# y1 = CNR
# # coeffs = np.polyfit(x, CNR, 1)
x1 = [0, 1, 5]
y1 = np.concatenate((CNR[0:1], CNR[2:]))
coeffs = np.polyfit(x1, y1, 1)

# Calculate y points from the fit above
xpts = np.linspace(-1, 6, 50)
y_all = coeffs[0] * xpts + coeffs[1]

# r-squared
p = np.poly1d(coeffs)
# fit values, and mean
yhat = p(x1)  #
ybar = np.mean(y1)  # average value of y
ssreg = np.sum((yhat - ybar) ** 2)  # or sum([ (yihat - ybar)**2 for yihat in yhat])
sstot = np.sum((y1 - ybar) ** 2)  # or sum([ (yi - ybar)**2 for yi in y])
r_sq = ssreg / sstot
r_sq = '%.3f' % r_sq
r_squared = str(r_sq)

# Plot the CNR
# sns.set()
plt.scatter(x[2:], CNR[2:], color='midnightblue')
plt.scatter(x[1], CNR[1], color='red')
plt.legend(['Single', 'Mixed'])
plt.plot(xpts[1:], y_all[1:], color='lightblue')
plt.plot(xpts, 5*np.ones(len(xpts)), color='red')
plt.title('K-Edge CNR' + extra_append)
plt.annotate("$R^2$ = " + r_squared, xy=(1, 0), xycoords='axes fraction', xytext=(-20, 20), textcoords='offset pixels',
             horizontalalignment='right', verticalalignment='bottom', fontsize=20)
plt.savefig(folder + '/K-Edge_CNR_' + save_nm + extra_append + '.png', dpi=500)
plt.close()

# Linearity

# Normalize to HU
water = mean[0]
mean = 1000*np.divide((np.subtract(mean, water)), water)
stddev = 1000*np.divide(stddev, water)

# Construct data for a linear fit of the data
# y1 = mean
# coeffs = np.polyfit(x, mean, 1)
y1 = np.concatenate((mean[0:1], mean[2:]))
coeffs = np.polyfit(x1, y1, 1)

# Calculate y points from the fit above
y_all = coeffs[0]*xpts + coeffs[1]

# r-squared
p = np.poly1d(coeffs)
# fit values, and mean
yhat = p(x1)  #
ybar = np.mean(y1)  # average value of y
ssreg = np.sum((yhat - ybar) ** 2)  # or sum([ (yihat - ybar)**2 for yihat in yhat])
sstot = np.sum((y1 - ybar) ** 2)  # or sum([ (yi - ybar)**2 for yi in y])
r_sq = ssreg / sstot
r_sq = '%.3f' % r_sq
r_squared = str(r_sq)

# Plot
# sns.set()
plt.scatter(x[2:], mean[2:], color='midnightblue')
plt.scatter(x[1], mean[1], color='red')
plt.legend(['Single', 'Mixed'])
plt.plot(xpts[1:], y_all[1:], color='lightblue')
plt.errorbar(x[1:], mean[1:], yerr=stddev[1:], fmt='none', color='orange')
plt.title('K-Edge Linearity' + extra_append, fontsize=30)
plt.annotate("$R^2$ = " + r_squared, xy=(1, 0), xycoords='axes fraction', xytext=(-20, 20), textcoords='offset pixels',
             horizontalalignment='right', verticalalignment='bottom', fontsize=20)
plt.savefig(folder + '/K-Edge_Linearity_' + save_nm + extra_append + '.png', dpi=500)
plt.show()
