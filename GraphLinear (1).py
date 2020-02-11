import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Things to enter
save_nm = 'I_Au_5-17'  # Save name to append
folder = 'I_Au_5_17_19'  # Folder name
vials = 6  # Number of ROIs
x = [0, 0.5, 1, 2, 3, 4]  # Concentrations
graph_title = ['16', '33', '50', '63', '81', '120']  # Bin values for graph titles

# Load each of the vial ROI matrices
v = np.empty([vials, 120, 120])
for k in np.arange(vials):
    v[k, :, :] = np.load(folder + '/Vial' + str(k+1) + '_MaskMatrix_' + save_nm + '.npy')

# Set up graph
#sns.set(style='darkgrid', context='poster')
fig, ax = plt.subplots(2, 3, figsize=(18, 11), sharex='col', sharey='row')
# add a big axes, hide frame
fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axes
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.grid(False)
plt.xlabel("Concentration (%)", labelpad=15, fontsize=35)
plt.ylabel("Signal (HU)", labelpad=30, fontsize=35)
plt.title("Signal vs. concentration with respect to photon energy bins", pad=50, fontsize=40)

# Set up data arrays
mean = np.empty(vials)
stddev = np.empty(vials)

for i in np.arange(6):

    # Load the appropriate matrix
    if i == 5:
        s = np.load(folder + '\Bin' + str(i+1) + '_Matrix_' + save_nm + '.npy')
    else:
        s = np.load(folder + '\Bin' + str(i) + '_Matrix_' + save_nm + '.npy')

    for j in np.arange(vials):
        temp = s*v[j]
        mean[j] = np.nanmean(temp)
        stddev[j] = np.nanstd(temp)

    # Normalize to HU
    water = mean[0]
    mean = 1000*np.divide((np.subtract(mean, water)), water)
    stddev = 1000*np.divide(stddev, water)

    # Construct data for a linear fit of the data
    coeffs = np.polyfit(x, mean, 1)

    # Calculate y points from the fit above
    xpts = np.linspace(-1, 6, 50)
    y_all = coeffs[0]*xpts + coeffs[1]
    contrast = str(int(coeffs[0]))

    # r-squared
    p = np.poly1d(coeffs)
    # fit values, and mean
    yhat = p(x)  #
    ybar = np.sum(mean) / len(mean)  # average value of y
    ssreg = np.sum((yhat - ybar) ** 2)  # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((mean - ybar) ** 2)  # or sum([ (yi - ybar)**2 for yi in y])
    r_sq = ssreg / sstot
    r_sq = '%.3f' % r_sq
    r_squared = str(r_sq)

    # Subplot title
    if i == 5:
        title = '16-120 keV'
    else:
        title = graph_title[i] + '-' + graph_title[i+1] + ' keV'

    if i < 3:
        ax[0, i].scatter(x[1:], mean[1:], color='midnightblue')
        ax[0, i].plot(xpts[1:], y_all[1:], color='lightblue')
        ax[0, i].errorbar(x[1:], mean[1:], yerr=stddev[1:], fmt='none', color='orange')
        ax[0, i].set_title(title, fontsize=30)
        ax[0, i].set_xlim(0, 6)
        ax[0, i].annotate("Contrast = " + contrast + " HU / %Au", xy=(1, 0), xycoords='axes fraction',
                          xytext=(-20, 20), textcoords='offset pixels',
                          horizontalalignment='right',
                          verticalalignment='bottom', fontsize=20)
        ax[0, i].annotate("$R^2$ = " + r_squared, xy=(0, 1), xycoords='axes fraction',
                          xytext=(200, -300), textcoords='offset pixels',
                          horizontalalignment='left',
                          verticalalignment='top', fontsize=20)

    else:
        ax[1, i - 3].scatter(x[1:], mean[1:], color='midnightblue')
        ax[1, i - 3].plot(xpts[1:], y_all[1:], color='lightblue')
        ax[1, i - 3].errorbar(x[1:], mean[1:], yerr=stddev[1:], fmt='none', color='orange')
        ax[1, i - 3].set_title(title, fontsize=30)
        ax[1, i - 3].set_xlim(0, 6)
        ax[1, i - 3].annotate("Contrast = " + contrast + " HU / %Au", xy=(1, 0), xycoords='axes fraction',
                              xytext=(-20, 20), textcoords='offset pixels',
                              horizontalalignment='right',
                              verticalalignment='bottom', fontsize=20)
        ax[1, i - 3].annotate("$R^2$ = " + r_squared, xy=(0, 1),
                              xycoords='axes fraction', xytext=(200, -300), textcoords='offset pixels',
                              horizontalalignment='left',
                              verticalalignment='top', fontsize=20)

plt.savefig(folder + '\AllBinsGraphs_' + save_nm + '.png', dpi=1000)

#plt.show()




