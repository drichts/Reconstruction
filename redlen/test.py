import numpy as np
import matplotlib.pyplot as plt


def cnr(corr1, b, mask1, bg1):
    return np.abs(np.nanmean(corr1[b] * mask1) - np.nanmean(corr1[b] * bg1)) / np.nanstd(corr1[b] * bg1)


def noise(data1, b, bg1):
    return np.nanstd(data1[b] * bg1)

def bgsignal(data1, b, bg1):
    return np.nanmean(data1[b] * bg1)

def signal(data1, b, mask1):
    return np.nanmean(data1[b] * mask1)

def sumpxp(data, num_pixels):
    """
    This function takes a data array and sums nxn pixels along the row and column data
    :param data: 5D ndarray
               The full data array <captures, counters, views, rows, columns>
    :return: The new data array with nxn pixels from the inital data summed together
    """
    dat_shape = np.array(np.shape(data))
    dat_shape[-2] = int(dat_shape[-2] / num_pixels)  # Reduce size by num_pixels in the row and column directions
    dat_shape[-1] = int(dat_shape[-1] / num_pixels)
    ndata = np.zeros(dat_shape)
    n = num_pixels
    for row in np.arange(dat_shape[-2]):
       for col in np.arange(dat_shape[-1]):
           # Get each 2x2 subarray over all of the first 2 axes
           if len(dat_shape) == 5:
               temp = data[:, :, :, n * row:n * row + n, n * col:n * col + n]
               ndata[:, :, :, row, col] = np.sum(temp, axis=(-2, -1))  # Sum over only the rows and columns
           elif len(dat_shape) == 4:
               temp = data[:, :, n * row:n * row + n, n * col:n * col + n]
               ndata[:, :, row, col] = np.sum(temp, axis=(-2, -1))  # Sum over only the rows and columns
           else:
               ndata = 0
    return ndata


dir = r'C:\Users\10376\Documents\Phantom Data\UNIFORMITY'

r_folder = '/many_thresholds_BB4mm/'
r_airfolder = '/many_thresholds_airscan/'

for f, folder in enumerate([r_folder]):

    airfolder = r_airfolder

    data = np.load(dir + folder + 'TestNum1_DataA0.npy')
    airdata = np.load(dir + airfolder + 'TestNum1_DataA0.npy')

    mask = np.load(dir + folder + 'Masks.npz', allow_pickle=True)
    bg = mask['bg'][3]
    mask = mask['mask'][3]

    # plt.imshow(bg)
    # plt.show()
    # plt.pause(1)
    # plt.close()
    #
    # plt.imshow(mask)
    # plt.show()
    # plt.pause(1)
    # plt.close()

    data = sumpxp(data, 4)
    airdata = sumpxp(airdata, 4)

    time = 25
    num = int(1000/time)

    airdata = np.sum(airdata, axis=1) / num

    cnr_vals = np.zeros([6, num])
    noise_vals = np.zeros([6, num])
    bgs = np.zeros([6, num])
    sig = np.zeros([6, num])

    for ii, i in enumerate(np.arange(0, 1000, time)):
        tdata = np.sum(data[:, i:i+time], axis=1)


    #tdata_extra = np.sum(tdata[], axis=)

        corr = -1*np.log(tdata/airdata)
        for ji, j in enumerate([6, 7, 8, 9, 10, 12]):
            cnr_vals[ji, ii] = cnr(corr, j, mask, bg)
            noise_vals[ji, ii] = noise(corr, j, bg)
            bgs[ji, ii] = bgsignal(corr, j, bg)
            sig[ji, ii] = signal(corr, j, mask)

    cont = abs(sig - bgs)
    for i in np.arange(6):
        # print(cnr_vals[i])
        # print(np.mean(cnr_vals[i]))
        # print(np.max(cnr_vals[i]))
        # print(np.argwhere(cnr_vals[i] > np.mean(cnr_vals[i])+200))
        # print(np.min(cnr_vals[i]))
        # print(np.argwhere(cnr_vals[i] < np.mean(cnr_vals[i]-200)))
        # #
        # # print(np.std(noise_vals[i]))
        # print()
        print(np.std(cnr_vals[i])/np.mean(cnr_vals[i])*100)
        print(np.std(noise_vals[i]) / np.mean(noise_vals[i])*100)
        print(np.std(cont[i])/np.mean(cont[i])*100)
        print()


    # print(np.load(dir + folder + 'TestNum1_noise_time.npy')[2, 6:13, 0, 12])
    print(np.load(dir + folder + 'TestNum1_cnr_time.npy')[4, 6:13, 0, 12])


    # print(np.mean(cnr_vals, axis=1))
    # print(np.std(cnr_vals, axis=1))
    # print()
    # print(np.mean(noise_vals, axis=1))
    # print(np.std(noise_vals, axis=1))
    # print()
    # print(np.mean(bgs, axis=1))
    # print(np.std(bgs, axis=1))
    # print()
    # print(np.mean(sig, axis=1))
    # print(np.std(sig, axis=1))

    # time_num = 12
    # print(np.load(dir + folder + 'TestNum1_cnr_time.npy')[0, 6, 0, time_num])
    # print(np.load(dir + folder + 'TestNum1_cnr_time.npy')[0, 7, 0, time_num])
    # print(np.load(dir + folder + 'TestNum1_cnr_time.npy')[0, 8, 0, time_num])
    # print(np.load(dir + folder + 'TestNum1_cnr_time.npy')[0, 9, 0, time_num])
    # print(np.load(dir + folder + 'TestNum1_cnr_time.npy')[0, 10, 0, time_num])
    # print(np.load(dir + folder + 'TestNum1_cnr_time.npy')[0, 12, 0, time_num])
    # print()
    # print()
    # print(np.load(dir + folder + 'TestNum1_cnr_time.npy')[0, 12, 0])

# plt.imshow(corr[12])
# plt.show()