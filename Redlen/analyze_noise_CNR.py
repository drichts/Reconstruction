import numpy as np
from scipy.interpolate import make_interp_spline as spline
import matplotlib.pyplot as plt


#directory = r'X:\TEST LOG\MINI MODULE\Canon\M20358_Q20/multiple_energy_thresholds_flatfield_3w/'
#path = directory + '/Raw Test Data/M20358_Q20/UNIFORMITY/*A0*'
#files = glob.glob(path)
#scandata = loadmat(files[0])

def gogo(b, scan_data):

    datafull = np.array(scan_data['cc_struct']['data'][0][0][0][0][0])

    #mask = np.load(directory + 'a0_Mask.npy')
    bg = np.load('C:/Users/10376/Documents/Phantom Data/Uniformity/Multiple Energy Thresholds/3w/a0_Background.npy')
    #datafull = np.squeeze(np.load(path))[b]
    datafull = np.squeeze(datafull)[b]
    data = datafull[0:500]
    data2 = datafull[0:10]
    print(np.shape(data))
    dif001 = np.zeros(500)
    std001 = np.zeros(500)

    for i in np.arange(500):
        dif001[i] = np.nanmean(data[i]*bg)  # - np.nanmean(data[i]*bg)
        std001[i] = np.nanstd(data[i]*bg)

    dif001 = np.mean(dif001)
    std001 = np.mean(std001)

    data = np.sum(data, axis=0)
    data2 = np.sum(data2, axis=0)
    datafull = np.sum(datafull, axis=0)

    dif2 = np.nanmean(data2*bg)# * mask) - np.nanmean(data2 * bg)
    std2 = np.nanstd(data2*bg)# * bg)
    dif5 = np.nanmean(data*bg)#*mask) - np.nanmean(data*bg)
    std5 = np.nanstd(data*bg)#*bg)
    dif1 = np.nanmean(datafull*bg)# * mask) - np.nanmean(datafull * bg)
    std1 = np.nanstd(datafull*bg)# * bg)

    print('1 ms noise/mean: ' + str(std001/dif001))
    print('10 ms noise/mean: ' + str(std2 / dif2))
    print('500 ms noise/mean: ' + str(std5/dif5))
    print('1 s noise/mean: ' + str(std1/dif1))


def gogo2(b):

    directory = 'C:/Users/10376/Documents/Phantom Data/Uniformity/Multiple Energy Thresholds/3w/'
    mask = np.load(directory + 'a0_Mask.npy')
    bg = np.load(directory + 'a0_Background.npy')
    path = directory + 'Data/Thresholds_1.npy'
    data = np.squeeze(np.load(path))[b]

    data5 = data[0:500]
    data01 = data[0:10]

    dif001 = np.zeros(500)
    mean001 = np.zeros(500)

    for i in np.arange(500):
        dif001[i] = np.nanmean(data[i] * mask) - np.nanmean(data[i]*bg)
        mean001[i] = np.nanmean(data[i] * bg)

    #dif001 = np.mean(dif001)
    #mean001 = np.mean(mean001)

    dif001 = np.nanmean(data[0] * mask) - np.nanmean(data[0] * bg)
    mean001 = np.nanmean(data[0] * bg)
    std001 = np.nanstd(data[0]*bg)

    data = np.sum(data, axis=0)
    data5 = np.sum(data5, axis=0)
    data01 = np.sum(data01, axis=0)

    dif5 = np.nanmean(data5 * mask) - np.nanmean(data5 * bg)
    mean5 = np.nanmean(data5 * bg)
    std5 = np.nanstd(data5 * bg)

    dif01 = np.nanmean(data01 * mask) - np.nanmean(data01*bg)
    mean01 = np.nanmean(data01 * bg)
    std01 = np.nanstd(data01 * bg)

    dif1 = np.nanmean(data * mask) - np.nanmean(data * bg)
    mean1 = np.nanmean(data * bg)
    std1 = np.nanstd(data * bg)

    print('1 ms contrast/mean: ' + str(dif001/mean001))
    print('10 ms contrast/mean: ' + str(dif01/mean01))
    print('500 ms contrast/mean: ' + str(dif5/mean5))
    print('1 s contrast/mean: ' + str(dif1/mean1))

    print('1 ms noise/mean: ' + str(std001 / mean001))
    print('10 ms noise/mean: ' + str(std01 / mean01))
    print('500 ms noise/mean: ' + str(std5 / mean5))
    print('1 s noise/mean: ' + str(std1 / mean1))


def gogo3(b):

    directory = 'C:/Users/10376/Documents/Phantom Data/Uniformity/Multiple Energy Thresholds/3w/'

    xpts = [0, 1, 2, 3, 4, 6]
    xsm = np.linspace(0, xpts[-1], 100)

    ypts001 = np.zeros(len(xpts))
    ypts01 = np.zeros(len(xpts))
    ypts5 = np.zeros(len(xpts))
    titles = ['20-30 keV', '30-50 keV', '50-70 keV', '70-90 keV', '90-120 keV']
    for idx, i in enumerate(xpts[1:]):
        if i == 1:
            mask = np.load(directory + 'a0_Mask.npy')
            bg = np.load(directory + 'a0_Background.npy')
            path = directory + 'Data/Thresholds_1.npy'
        else:
            pxp = str(i) + 'x' + str(i)
            mask = np.load(directory + pxp + '_a0_Mask.npy')
            bg = np.load(directory + pxp + '_a0_Background.npy')
            path = directory + pxp + ' Data/Thresholds_1.npy'

        data = np.squeeze(np.load(path))[b]

        data5 = data[0:500]
        data01 = data[0:10]

        dif001 = np.zeros(500)
        mean001 = np.zeros(500)
        std001 = np.zeros(500)

        for j in np.arange(500):
            dif001[j] = np.nanmean(data[j] * mask) - np.nanmean(data[j] * bg)
            mean001[j] = np.nanmean(data[j] * bg)
            std001[j] = np.nanstd(data[j] * bg)

        dif001 = np.mean(dif001)
        mean001 = np.mean(mean001)
        std001 = np.mean(std001)

        #dif001 = np.nanmean(data[0] * mask) - np.nanmean(data[0] * bg)
        #mean001 = np.nanmean(data[0] * bg)
        #std001 = np.nanstd(data[0] * bg)

        data = np.sum(data, axis=0)
        data5 = np.sum(data5, axis=0)
        data01 = np.sum(data01, axis=0)

        dif5 = np.nanmean(data5 * mask) - np.nanmean(data5 * bg)
        mean5 = np.nanmean(data5 * bg)
        std5 = np.nanstd(data5 * bg)

        dif01 = np.nanmean(data01 * mask) - np.nanmean(data01 * bg)
        mean01 = np.nanmean(data01 * bg)
        std01 = np.nanstd(data01 * bg)

        dif1 = np.nanmean(data * mask) - np.nanmean(data * bg)
        mean1 = np.nanmean(data * bg)
        std1 = np.nanstd(data * bg)
        #print(i)
        #print('1 ms contrast/mean: ' + str(dif001 / mean001))
        #print('10 ms contrast/mean: ' + str(dif01 / mean01))
        #print('500 ms contrast/mean: ' + str(dif5 / mean5))
        #print('1 s contrast/mean: ' + str(dif1 / mean1))
        #print()
        #print('1 ms noise/mean: ' + str(std001 / mean001))
        #rint('10 ms noise/mean: ' + str(std01 / mean01))
        #print('500 ms noise/mean: ' + str(std5 / mean5))
        #rint('1 s noise/mean: ' + str(std1 / mean1))
        #print()

        ypts001[idx+1] = std001 / mean001
        ypts01[idx + 1] = std01 / mean01
        ypts5[idx + 1] = std5 / mean5

    ysm001 = spline(xpts, ypts001)
    ysm01 = spline(xpts, ypts01)
    ysm5 = spline(xpts, ypts5)

    fig = plt.figure(figsize=(5, 5))
    plt.plot(xsm, ysm001(xsm), color='black')
    plt.plot(xsm, ysm01(xsm), color='red')
    plt.plot(xsm, ysm5(xsm), color='blue')
    plt.scatter(xpts, ypts001, color='black', s=20)
    plt.scatter(xpts, ypts01, color='red', s=20)
    plt.scatter(xpts, ypts5, color='blue', s=20)
    plt.legend(['1 ms', '10 ms', '500 ms'])
    plt.title(titles[b-6])
    plt.xlabel('Num Pixels')
    plt.ylabel('Noise in background (% of mean background)')
    plt.subplots_adjust(left=0.15, right=0.95)
    plt.show()


