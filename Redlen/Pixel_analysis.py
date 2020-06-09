import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

def get_CNR_vs_num_pixel_data(test_num, rng,
                           directory='C:/Users/10376/Documents/Phantom Data/Uniformity/Multiple Energy Thresholds/3w/'):
    """
    This function grabs the CNR at each pixel size for
    :param test_num:
    :param rng:
    :param directory:
    :return:
    """
    data = np.zeros([7, 2])
    for i, folder in enumerate(['Data', '2x2 Data', '3x3 Data', '4x4 Data', '6x6 Data', '8x8 Data', '12x12 Data']):
        path = directory + folder + '/Single Frame Avg/' + str(test_num) + '_' + rng + ' keV.npy'
        data[i] = np.load(path)

    return data


def line(x, m):
    return m*x

def quadratic(x, a, b):
    return a*x*x + b*x

def sqroot(x, a, b):
    return a*np.sqrt(np.abs(b)*x)


def plot_all_bins(test_num, order, save=False, nine=False,
                  directory='C:/Users/10376/Documents/Phantom Data/Uniformity/Multiple Energy Thresholds/3w/'):
    binrngs = [['20-30', '30-50', '50-70', '70-90', '90-120', '20-120'],
               ['20-30', '30-40', '40-50', '50-60', '60-70', '20-70'],
               ['20-35', '35-50', '50-65', '65-80', '80-90', '20-90'],
               ['25-35', '35-45', '45-55', '55-65', '65-75', '25-75'],
               ['25-40', '40-55', '55-70', '70-80', '80-95', '25-95'],
               ['30-45', '45-60', '60-75', '75-85', '85-95', '30-95'],
               ['20-30', '30-70', '70-85', '85-100', '100-120', '20-120']]

    rngs = binrngs[test_num-1]  # Get the bin widths for the test number

    if nine:
        pixels = np.array([0, 1, 4, 9, 16, 36, 64, 144])
        xpts = np.linspace(0, 145, 100)
    else:
        pixels = np.array([0, 1, 2, 3, 4, 6, 8, 12])  # Number of pixels in the row and column direction (2x2, 1x1, 3x3)
        xpts = np.linspace(0, 12.5, 50)

    fig, axes = plt.subplots(2, 3, figsize=(8, 6), sharey=True)
    ax1 = fig.add_subplot(111, frameon=False)
    ax1.grid(False)
    # Hide axes ticks
    ax1.set_xticks([])
    ax1.set_yticks([])

    for i, ax in enumerate(axes.flat):
        data = np.zeros([8, 2])
        data[1:] = get_CNR_vs_num_pixel_data(test_num, rngs[i], directory=directory)
        if order == 1:
            coeffs, covar = curve_fit(line, pixels, data[:, 0])
            ypts = line(xpts, coeffs[0])
            y_pred = line(pixels, coeffs[0])
            r2 = r2_score(data[:, 0], y_pred)
            ax.annotate('Slope: m=%5.3f' % tuple(coeffs), (0.2, 30.5))
            ax.annotate(r'R$^2$=%5.3f' % r2, (0.2, 27))
            ax.set_xlabel('CNR(1)=%5.2f' % data[1, 0])
        elif order == 2:
            coeffs, covar = curve_fit(quadratic, pixels, data[:, 0])
            ypts = quadratic(xpts, coeffs[0], coeffs[1])
            y_pred = quadratic(pixels, coeffs[0], coeffs[1])
            r2 = r2_score(data[:, 0], y_pred)
            ax.annotate('Coeffs: a=%5.3f, \n'
                        '            b=%5.3f' % tuple(coeffs), (0.2, 29))
            ax.annotate(r'R$^2$=%5.3f' % r2, (0.2, 25))
        else:
            coeffs, covar = curve_fit(sqroot, pixels, data[:, 0])
            ypts = sqroot(xpts, coeffs[0], coeffs[1])
            y_pred = sqroot(pixels, coeffs[0], coeffs[1])
            r2 = r2_score(data[:, 0], y_pred)
            ax.annotate('Coeffs: a=%5.3f, \n'
                        '            b=%5.3f' % tuple(coeffs), (0.2, 29))
            ax.annotate(r'R$^2$=%5.3f' % r2, (0.2, 25))

        ax.plot(xpts, ypts, color='midnightblue', linewidth=2)
        ax.errorbar(pixels, data[:, 0], yerr=data[:, 1], fmt='none', color='midnightblue', capsize=3)
        ax.set_title(rngs[i] + ' keV')
        if nine:
            ax.set_xlim([0, 145])
        else:
            ax.set_xlim([0, 12.5])
        ax.set_ylim([0, 35])

    if order == 1:
        ax1.set_title('Linear fit', fontsize=15, pad=30)
    elif order == 2:
        ax1.set_title('Quadratic fit', fontsize=15, pad=30)
    else:
        ax1.set_title('Sq. root fit', fontsize=15, pad=30)

    if nine:
        ax1.set_xlabel('Total # of Pixels', fontsize=15, labelpad=25)
    else:
        ax1.set_xlabel('Number of Pixels Sq.', fontsize=15, labelpad=35)
    ax1.set_ylabel('CNR', fontsize=15, labelpad=30)
    plt.subplots_adjust(hspace=0.4, bottom=0.15)
    plt.show()

    if save:
        if order == 1:
            plt.savefig(r'C:\Users\10376\Documents\Phantom Data\Uniformity\Multiple Energy Thresholds\Plots/'
                        r'CNR vs Pixels/Test Num ' + str(test_num) + ' Linear.png', dpi=fig.dpi)
        elif order == 2:
            plt.savefig(r'C:\Users\10376\Documents\Phantom Data\Uniformity\Multiple Energy Thresholds\Plots/'
                        r'CNR vs Pixels/Test Num ' + str(test_num) + ' Quadratic.png', dpi=fig.dpi)
        else:
            plt.savefig(r'C:\Users\10376\Documents\Phantom Data\Uniformity\Multiple Energy Thresholds\Plots/'
                        r'CNR vs Pixels/Test Num ' + str(test_num) + ' SqRoot.png', dpi=fig.dpi)
        plt.close()


def plot_ranges(order, save=False, nine=False,
                  directory='C:/Users/10376/Documents/Phantom Data/Uniformity/Multiple Energy Thresholds/3w/'):

    nbt = [[20, 30, 50, 70, 90, 120],
           [20, 30, 40, 50, 60, 70],
           [20, 35, 50, 65, 80, 90],
           [25, 35, 45, 55, 65, 75],
           [25, 40, 55, 70, 80, 95],
           [30, 45, 60, 75, 85, 95],
           [20, 30, 70, 85, 100, 120]]

    rngs = [[0, 0, 3], [2, 0, 3], [6, 0, 2], [0, 0, 2], [1, 0, 3], [2, 0, 2]]  # Get the bin widths for the test number

    if nine:
        pixels = np.array([0, 1, 4, 9, 16, 36, 64, 144])
        xpts = np.linspace(0, 145, 100)
    else:
        pixels = np.array([0, 1, 2, 3, 4, 6, 8, 12])  # Number of pixels in the row and column direction (2x2, 1x1, 3x3)
        xpts = np.linspace(0, 12.5, 50)

    fig, axes = plt.subplots(2, 3, figsize=(8, 6), sharey=True)
    ax1 = fig.add_subplot(111, frameon=False)
    ax1.grid(False)
    # Hide axes ticks
    ax1.set_xticks([])
    ax1.set_yticks([])

    for i, ax in enumerate(axes.flat):
        data = np.zeros([8, 2])
        test_num = rngs[i][0] + 1
        rng = str(nbt[rngs[i][0]][rngs[i][1]]) + '-' + str(nbt[rngs[i][0]][rngs[i][2]])
        data[1:] = get_CNR_vs_num_pixel_data(test_num, rng, directory=directory)
        if order == 1:
            coeffs, covar = curve_fit(line, pixels, data[:, 0])
            ypts = line(xpts, coeffs[0])
            y_pred = line(pixels, coeffs[0])
            r2 = r2_score(data[:, 0], y_pred)
            ax.annotate('Slope: m=%5.3f' % tuple(coeffs), (0.2, 30.5))
            ax.annotate(r'R$^2$=%5.3f' % r2, (0.2, 27))
            ax.set_xlabel('CNR(1)=%5.2f' % data[1, 0])
        elif order == 2:
            coeffs, covar = curve_fit(quadratic, pixels, data[:, 0])
            ypts = quadratic(xpts, coeffs[0], coeffs[1])
            y_pred = quadratic(pixels, coeffs[0], coeffs[1])
            r2 = r2_score(data[:, 0], y_pred)
            ax.annotate('Coeffs: a=%5.3f, \n'
                        '            b=%5.3f' % tuple(coeffs), (0.2, 29))
            ax.annotate(r'R$^2$=%5.3f' % r2, (0.2, 25))
        else:
            coeffs, covar = curve_fit(sqroot, pixels, data[:, 0])
            ypts = sqroot(xpts, coeffs[0], coeffs[1])
            y_pred = sqroot(pixels, coeffs[0], coeffs[1])
            r2 = r2_score(data[:, 0], y_pred)
            ax.annotate('Coeffs: a=%5.3f, \n'
                        '            b=%5.3f' % tuple(coeffs), (0.2, 29))
            ax.annotate(r'R$^2$=%5.3f' % r2, (0.2, 25))

        ax.plot(xpts, ypts, color='midnightblue', linewidth=2)
        ax.errorbar(pixels, data[:, 0], yerr=data[:, 1], fmt='none', color='midnightblue', capsize=3)
        ax.set_title(rng + ' keV')
        if nine:
            ax.set_xlim([0, 145])
        else:
            ax.set_xlim([0, 12.5])
        ax.set_ylim([0, 35])

    if order == 1:
        ax1.set_title('Linear fit', fontsize=15, pad=30)
    elif order == 2:
        ax1.set_title('Quadratic fit', fontsize=15, pad=30)
    else:
        ax1.set_title('Sq. root fit', fontsize=15, pad=30)

    if nine:
        ax1.set_xlabel('Total # of Pixels', fontsize=15, labelpad=25)
    else:
        ax1.set_xlabel('Number of Pixels Sq.', fontsize=15, labelpad=35)
    ax1.set_ylabel('CNR', fontsize=15, labelpad=30)
    plt.subplots_adjust(hspace=0.4, bottom=0.15)
    plt.show()

    if save:
        if order == 1:
            plt.savefig(r'C:\Users\10376\Documents\Phantom Data\Uniformity\Multiple Energy Thresholds\Plots/'
                        r'CNR vs Pixels/High Ranges Linear.png', dpi=fig.dpi)
        elif order == 2:
            plt.savefig(r'C:\Users\10376\Documents\Phantom Data\Uniformity\Multiple Energy Thresholds\Plots/'
                        r'CNR vs Pixels/High Ranges Quadratic.png', dpi=fig.dpi)
        else:
            plt.savefig(r'C:\Users\10376\Documents\Phantom Data\Uniformity\Multiple Energy Thresholds\Plots/'
                        r'CNR vs Pixels/High Ranges SqRoot.png', dpi=fig.dpi)
        plt.close()
