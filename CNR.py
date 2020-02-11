import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
# Things to enter
directory = 'D:/Research/Python Data/Spectral CT/'
folder1 = 'Au_9-17-19/'  # Folder name
#folder2 = 'I_9-17-19/'
path1 = directory + folder1
#path2 = directory + folder2
vials = 6  # Number of ROIs
conc = [0, 4, 3, 2, 1, 0.5]  # Concentrations
graph_title = ['16', '33', '50', '64', '81', '120']  # Bin values for graph titles

# Set up graph
fig, ax = plt.subplots(2, 3, figsize=(18, 11))
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.xlabel("Concentration (%)", labelpad=20, fontsize=28)
plt.ylabel("CNR", labelpad=200, fontsize=28)
sns.set()
# Load each of the vial ROI matrices
v1 = np.load(path1 + '/Vial_Masks.npy')
#v2 = np.load(path2 + '/Vial_Masks.npy')

# Calculate the CNR for each of the vials in the bins and graph
for i in np.arange(6):

    # Load the appropriate matrix
    if i == 5:
        s1 = np.load(path1 + '\Slices\Bin' + str(i + 1) + '_Slice14.npy')
        #s2 = np.load(path2 + '\Slices\Bin' + str(i + 1) + '_Slice14.npy')
    else:
        s1 = np.load(path1 + '\Slices\Bin' + str(i) + '_Slice14.npy')
        #s2 = np.load(path2 + '\Slices\Bin' + str(i + 1) + '_Slice14.npy')

    # Std. Dev. of background
    sig_bg1 = np.nanstd(v1[0]*s1)
    #sig_bg2 = np.nanstd(v2[0] * s2)

    # Average background reading
    bg1 = np.nanmean(v1[0]*s1)
    #bg2 = np.nanmean(v2[0] * s2)

    # Mean signal and std dev from each vial
    mean1 = np.empty(vials)
    stddev1 = np.empty(vials)
    #mean2 = np.empty(vials)
    #stddev2 = np.empty(vials)

    for j in np.arange(vials):
        temp1 = s1*v1[j]
     #   temp2 = s2 * v2[j]

        mean1[j] = np.nanmean(temp1)
        stddev1[j] = np.nanstd(temp1)

    #    mean2[j] = np.nanmean(temp2)
    #   stddev2[j] = np.nanstd(temp2)

    # Calculate the CNR of each vial
    CNR1 = np.divide(np.subtract(mean1, bg1), sig_bg1)
    #CNR2 = np.divide(np.subtract(mean2, bg2), sig_bg2)

    stddev1 = np.divide(stddev1, sig_bg1)
    #stddev2 = np.divide(stddev2, sig_bg2)

    # Construct data for a linear fit of the data
    coeffs1 = np.polyfit(conc, CNR1, 1)
    #coeffs2 = np.polyfit(conc, CNR2, 1)

    # Calculate y points from the fit above
    xpts = np.linspace(-1, 5, 100)
    ypts1 = coeffs1[0] * xpts + coeffs1[1]
    #ypts2 = coeffs2[0] * xpts + coeffs2[1]

    # r-squared
    p = np.poly1d(coeffs1)
    # fit values, and mean
    yhat = p(conc)  #
    ybar = np.sum(CNR1) / len(CNR1)  # average value of y
    ssreg = np.sum((yhat - ybar) ** 2)  # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((CNR1 - ybar) ** 2)  # or sum([ (yi - ybar)**2 for yi in y])
    r_sq = ssreg / sstot
    r_sq = '%.3f' % r_sq
    r_squared = str(r_sq)

    # Subplot title
    if i == 6:
        title = '16-120 keV'
    elif i == 5:
        title = '16-120 keV'
    else:
        title = graph_title[i] + '-' + graph_title[i+1] + ' keV'

    if i < 3:
        ax[0, i].plot(xpts, 4*np.ones(len(xpts)), color='red', linestyle='dashed', linewidth=2)
        ax[0, i].plot(xpts[1:], ypts1[1:], color='midnightblue', linewidth=2)
        ax[0, i].errorbar(conc[1:], CNR1[1:], yerr=stddev1[1:], fmt='none', color='midnightblue', capsize=3)
        #ax[0, i].scatter(conc[1:], CNR1[1:], color='orange', linewidths=2)
        #ax[0, i].plot(xpts[1:], ypts2[1:], color='blue', linewidth=2)
        #ax[0, i].errorbar(conc[1:], CNR2[1:], yerr=stddev2[1:], fmt='none', color='blue')
        #ax[0, i].scatter(conc[1:], CNR2[1:], color='blue', linewidths=2)
        ax[0, i].tick_params(labelsize=25)
        ax[0, i].set_title(title, fontsize=25)
        ax[0, i].set_xlim(0, 5)
        ax[0, i].set_ylim(-5, 150)
        ax[0, i].annotate("$R^2$ = " + r_squared, xy=(0, 1), xycoords='axes fraction',
                          xytext=(100, -150), textcoords='offset pixels',
                          horizontalalignment='left',
                          verticalalignment='top', fontsize=25)

    elif 3 <= i < 6:
        ax[1, i - 3].plot(xpts, 4*np.ones(len(xpts)), color='red', linestyle='dashed', linewidth=2)
        ax[1, i - 3].plot(xpts[1:], ypts1[1:], color='midnightblue', linewidth=2)
        #ax[1, i - 3].plot(xpts[1:], ypts2[1:], color='blue', linewidth=2)
        #ax[1, i - 3].scatter(conc[1:], CNR1[1:], color='orange', linewidths=2)
        ax[1, i - 3].errorbar(conc[1:], CNR1[1:], yerr=stddev1[1:], fmt='none', color='midnightblue', capsize=3)

        #ax[1, i - 3].scatter(conc[1:], CNR2[1:], color='blue', linewidths=2)
        #ax[1, i - 3].errorbar(conc[1:], CNR2[1:], yerr=stddev2[1:], fmt='none', color='blue')
        ax[1, i - 3].tick_params(labelsize=25)
        ax[1, i - 3].set_title(title, fontsize=25)
        ax[1, i - 3].set_xlim(0, 5)
        ax[1, i - 3].set_ylim(-5, 150)
        ax[1, i - 3].annotate("$R^2$ = " + r_squared, xy=(0, 1), xycoords='axes fraction',
                              xytext=(100, -150), textcoords='offset pixels',
                              horizontalalignment='left',
                              verticalalignment='top', fontsize=25)


# Bins 0-5

ax[1, 1].legend(['Rose Criterion', 'Au', 'I'], fontsize=25, loc='upper center', bbox_to_anchor=(0.5, -0.3),
                fancybox=True, shadow=True, ncol=3)
plt.show()

#%%
#plt.plot(xpts, 4*np.ones(len(xpts)), color='red', linestyle='dashed', linewidth=3)
#plt.plot(xpts[1:], y_all[1:], color='lightblue', linewidth=3)
#plt.scatter(conc[1:], CNR[1:], color='midnightblue', s=100)
#plt.errorbar(conc[1:], mean[1:], yerr=stddev[1:], fmt='none', color='orange')
#plt.tick_params(labelsize=30)
#plt.title('Gold (16-120 keV)', fontsize=40)
#plt.xlabel('Concentration (%)', fontsize=35)
#plt.ylabel('CNR', fontsize=35)
#plt.xlim(0, 6)
#plt.ylim(0, 60)
#plt.annotate("$R^2$ = " + r_squared, xy=(0, 1), xycoords='axes fraction',
#             xytext=(50, -50), textcoords='offset pixels', horizontalalignment='left', verticalalignment='top',
#             fontsize=30)
#plt.show()