import numpy as np
import matplotlib.pyplot as plt

filepath = 'D:/Research/Python Data/CBCT/Lan_7-18-19/'

percentages = ['3_Percent/']#, '3_Percent/', '5_Percent/']

# Set whether or not the data is ROI based (True) or pixel by pixel based (False)
ROI_flag = True

for percent in percentages:

    folder = filepath + percent

    if ROI_flag:
        Z_values = np.load(folder + 'Z-ROI.npy')
        densities = np.load(folder + 'dens-ROI.npy')

        #Z_values = np.reshape(Z_values, (14, 3))
        #Z_tot = np.array([])
        Z_tot = np.append(Z_values[:, 0], Z_values[:, 1])
        Z_tot = np.append(Z_tot, Z_values[:, 2])
        Z_tot = np.append(Z_tot, Z_values[:, 3])
        Z_tot = np.append(Z_tot, Z_values[:, 4])

       # densities = np.reshape(densities, (14, 3))
        #elec = np.array([])
        elec = np.append(densities[:, 0], densities[:, 1])
        elec = np.append(elec, densities[:, 2])
        elec = np.append(elec, densities[:, 3])
        elec = np.append(elec, densities[:, 4])

        # Construct data for a linear fit of the data
        coeffs = np.polyfit(Z_tot, elec, 1)

        # Calculate y points from the fit above
        xpts = np.linspace(5, 20, 100)

        # r-squared
        p = np.poly1d(coeffs)
        # fit values, and mean
        yhat = p(Z_tot)  #
        ybar = np.sum(elec) / len(elec)  # average value of y
        ssreg = np.sum((yhat - ybar) ** 2)  # or sum([ (yihat - ybar)**2 for yihat in yhat])
        sstot = np.sum((elec - ybar) ** 2)  # or sum([ (yi - ybar)**2 for yi in y])
        r_sq = ssreg / sstot
        r_sq = '%.3f' % r_sq
        r_squared = str(r_sq)

        plt.scatter(Z_values[:, 0], densities[:, 0], color='blue')
        plt.scatter(Z_values[:, 1], densities[:, 1], color='orange')
        plt.scatter(Z_values[:, 2], densities[:, 2], color='green')
        plt.scatter(Z_values[:, 4], densities[:, 4], color='purple')
        plt.scatter(Z_values[:, 3], densities[:, 3], color='red')
        plt.scatter(Z_values[:, 5], densities[:, 5], color='black')
        #plt.plot(xpts, p(xpts), color='black')
        #plt.annotate("$R^2$ = " + r_squared, xy=(0, 1), xycoords='axes fraction', xytext=(200, -300),
        #             textcoords='offset pixels', horizontalalignment='left', verticalalignment='top')

    else:
        Z_values = np.load(folder + 'Z-pixel.npy')
        densities = np.load(folder + 'dens-pixel.npy')

        plt.scatter(Z_values, densities, s=2)


if ROI_flag:
    plt.title('Lanthanide Phantom Material Decomposition', fontsize=40)
    plt.ylabel('Electron density', fontsize=30)
    plt.xlabel('Z', fontsize=30)
    plt.xlim([5, 16])
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.legend(['water', 'I (Z=53)', 'Gd (Z=64)', 'Dy (Z=66)', 'Lu (Z=71)', 'Au (Z=79)'], fontsize=20)

else:
    plt.title('Pixel by Pixel Z vs. electron density; Pixel count: ' + str(len(Z_values)), fontsize=40)
    plt.ylabel(r'$\rho_e$', fontsize=30)
    plt.xlabel('Z', fontsize=30)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.xlim([5, 20])
    plt.ylim([0, 2.2])

plt.show()
