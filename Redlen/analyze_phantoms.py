import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import general_OS_functions as gof
import sCT_Analysis as sct


def get_CNR_over_time_data_raw(folder, air_folder, directory='C:/Users/10376/Documents/Phantom Data/Uniformity/'):
    """
    This function takes a folder and collects the CNR is all CC bins at 0.001, 0.01, 0.1, 0.5, 1, 2, 3.... seconds
    :param folder:
    :param air_folder:
    :param directory:
    :return:
    """
    contrast_mask = np.load(directory + folder + '/a0_Mask.npy')
    bg_mask = np.load(directory + folder + '/a0_Background.npy')

    air = np.load(directory + air_folder + '/Raw Data/a0.npy')
    air = np.squeeze(air)

    data = np.load(directory + folder + '/Raw Data/Run001_a0.npy')
    data = np.squeeze(data)

    time_pts = np.array([0.001, 0.01, 0.1, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    CNR_pts = np.zeros([6, len(time_pts)])
    CNR_err = np.zeros([6, len(time_pts)])

    # The first fractions of a second results
    for i in np.arange(5):
        slices = int(time_pts[i]*1000)
        temp_data = np.sum(data[:, 0:slices, :, :], axis=1)
        temp_air = np.sum(air[:, 0:slices, :, :], axis=1)

        corrected_data = np.log(np.divide(temp_air, temp_data))

        for idx, b in enumerate(np.array([6, 7, 8, 9, 10, 12])):
            CNR_pts[idx, i], CNR_err[idx, i] = sct.cnr(corrected_data[b], contrast_mask, bg_mask)

    data = np.sum(data, axis=1)
    air = np.sum(air, axis=1)

    for j in np.arange(5, len(time_pts)):
        air = np.add(air, air)
        add_data = np.load(directory + folder + '/Raw Data/Run' + '{:03d}'.format(j-3) + '_a0.npy')
        add_data = np.squeeze(add_data)
        add_data = np.sum(add_data, axis=1)
        data = np.add(data, add_data)

        corrected_data = np.log(np.divide(air, data))

        for idx, b in enumerate(np.array([6, 7, 8, 9, 10, 12])):
            CNR_pts[idx, j], CNR_err[idx, j] = sct.cnr(corrected_data[b], contrast_mask, bg_mask)

    return time_pts, CNR_pts, CNR_err


def get_CNR_over_time_data_corrected_10s(folder, directory='C:/Users/10376/Documents/Phantom Data/Uniformity/'):

    contrast_mask = np.load(directory + folder + '/a0_Mask.npy')
    bg_mask = np.load(directory + folder + '/a0_Background.npy')

    time_pts = np.arange(0.001, 10.001, 0.001)  # Time points from 0.001 s to 10 s by 0.001 s increments
    CNR_pts = np.zeros([6, len(time_pts)])
    CNR_err = np.zeros([6, len(time_pts)])

    total_data = np.zeros([6, 24, 36])
    #random_order = np.array([10, 5, 6, 3, 9, 1, 4, 2, 8, 7])
    random_order = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
    for i in np.arange(1, 11):
        nn = random_order[i-1]
        add_data = np.load(directory + folder + '/Corrected Data/Run' + '{:03d}'.format(nn) + '_a0.npy')
        add_data = np.squeeze(add_data)
        add_data = add_data[6:11]

        temp_data = np.zeros([6, 24, 36])
        for j in np.arange(1000):
            single_frame = add_data[:, j]
            temp_data[0:5] = np.add(temp_data[0:5], single_frame)
            sumcc_single_frame = np.sum(single_frame, axis=0)
            temp_data[5] = np.add(temp_data[5], sumcc_single_frame)

            total_data = np.add(total_data, temp_data)

            for k, img in enumerate(total_data):
                CNR_pts[k, (i-1)*1000+j], blah = sct.cnr(img, contrast_mask, bg_mask)

    return time_pts, CNR_pts


def plot_CNR_over_time_10s(time_pts, CNR_pts, CNR_err=[], title='n/a', save=False,
                       directory='C:/Users/10376/Documents/Phantom Data/Uniformity/'):

    sns.set_style('whitegrid')
    fig, axes = plt.subplots(2, 3, figsize=(8, 6))
    ax1 = fig.add_subplot(111, frameon=False)
    ax1.grid(False)
    ax1.set_xticks([])
    ax1.set_yticks([])
    titles = ['20-30 keV', '30-50 keV', '50-70 keV', '70-90 keV', '90-120 keV', 'Sum CC']

    max_CNR = np.max(CNR_pts) + 0.25
    for i, ax in enumerate(axes.flat):
        ax.plot(time_pts, CNR_pts[i], lw=1)
        ax.set_title(titles[i])
        ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        ax.set_ylim([0, max_CNR])

    plt.subplots_adjust(left=0.12, bottom=0.11, right=0.96, top=0.88, wspace=0.31, hspace=0.44)
    ax1.set_xlabel('Acquisition Time (s)', fontsize=14, labelpad=25)
    ax1.set_ylabel('CNR', fontsize=14, labelpad=30)
    ax1.set_title(title, fontsize=15, pad=25)
    plt.show()

    if save:
        plt.savefig(directory + '/Plots/CNR_' + title + '.png', dpi=fig.dpi)
    plt.close()


def add_adj_bins(data, bins):
    """
    This function takes the adjacent bins given in bins and sums along the bin axis, can sum multiple bins
    :param data: 4D numpy array
                Data array with shape <counters, views, rows, columns
    :param bins: 1D array
                Bin numbers (as python indices, i.e the 1st bin would be 0) to sum
                Form: [Starting bin, Ending bin]
                Ex. for the 2nd through 5th bins, bins = [1, 4]
    :return: The summed data with the summed bins added together and the rest of the data intact
                shape <counters, views, rows, columns>
    """
    data_shape = np.array(np.shape(data))
    data_shape[0] = data_shape[0] - (bins[1] - bins[0])  # The new data will have the number of added bins - 1 new counters
    new_data = np.zeros(data_shape)

    new_data[0:bins[0]] = data[0:bins[0]]
    new_data[bins[0]] = np.sum(data[bins[0]:bins[-1]+1], axis=0)
    new_data[bins[0]+1:] = data[bins[1]+1:]

    return new_data


def get_CNR_over_1s_sum_adj_bin(folder, bins, corr3x3=False, CC=False,
                                directory='C:/Users/10376/Documents/Phantom Data/Uniformity/'):

    num_bins = 6 - (bins[1] - bins[0])
    if corr3x3:
        contrast_mask = np.load(directory + folder + '/3x3_a0_Mask.npy')
        bg_mask = np.load(directory + folder + '/3x3_a0_Background.npy')
    else:
        contrast_mask = np.load(directory + folder + '/a0_Mask.npy')
        bg_mask = np.load(directory + folder + '/a0_Background.npy')

    time_pts = np.arange(0.001, 1.001, 0.001)  # Time points from 0.001 s to 10 s by 0.001 s increments
    CNR_pts = np.zeros([10, num_bins, len(time_pts)])  # Collect CNR over 1 s for all 10 files

    for i in np.arange(1, 11):

        if corr3x3:
            total_data = np.zeros([num_bins, 8, 12])  # Holds the current data for all bins plus the sum of all bins
            add_data = np.load(directory + folder + '/3x3 Corrected Data/Run' + '{:03d}'.format(i) + '_a0.npy')
        else:
            total_data = np.zeros([num_bins, 24, 36])  # Holds the current data for all bins plus the sum of all bins
            add_data = np.load(directory + folder + '/Corrected Data/Run' + '{:03d}'.format(i) + '_a0.npy')

        add_data = np.squeeze(add_data)  # Squeeze out the single capture axis

        if CC:
            add_data = add_data[6:12]  # Grab just cc (or sec) bins
        else:
            add_data = add_data[0:6]  # Grab just sec bins

        add_data = add_adj_bins(add_data, bins)  # Add the appropriate bins together

        for j in np.arange(1000):
            single_frame = add_data[:, j]  # Get the next view data
            total_data[0:num_bins-1] = np.add(total_data[0:num_bins-1], single_frame[0:num_bins-1])  # Add to the current total data
            sum_single_frame = np.sum(single_frame, axis=0)  # Sum all bins to get summed cc (or sec)
            total_data[num_bins-1] = np.add(total_data[num_bins-1], sum_single_frame)  # Add to the total summed

            for k, img in enumerate(total_data):
                # Calculate the CNR (i-1 = file, k = bin, j = view/time point)
                CNR_pts[i-1, k, j], err = sct.cnr(img, contrast_mask, bg_mask)

    CNR_pts = np.mean(CNR_pts, axis=0)  # Average over all of the files

    return time_pts, CNR_pts


def get_CNR_over_time_data_corrected_1sec(folder, corr3x3=False, CC=True,
                                          directory='C:/Users/10376/Documents/Phantom Data/Uniformity/'):

    if corr3x3:
        contrast_mask = np.load(directory + folder + '/3x3_a0_Mask.npy')
        bg_mask = np.load(directory + folder + '/3x3_a0_Background.npy')
    else:
        contrast_mask = np.load(directory + folder + '/a0_Mask.npy')
        bg_mask = np.load(directory + folder + '/a0_Background.npy')

    time_pts = np.arange(0.001, 1.001, 0.001)  # Time points from 0.001 s to 10 s by 0.001 s increments
    CNR_pts = np.zeros([10, 6, len(time_pts)])  # Collect CNR over 1 s for all 10 files
    CNR_err = np.zeros([10, 6])

    for i in np.arange(1, 11):

        if corr3x3:
            total_data = np.zeros([6, 8, 12])  # Holds the current data for all bins plus the sum of all bins
            add_data = np.load(directory + folder + '/3x3 Corrected Data/Run' + '{:03d}'.format(i) + '_a0.npy')
        else:
            total_data = np.zeros([6, 24, 36])  # Holds the current data for all bins plus the sum of all bins
            add_data = np.load(directory + folder + '/Corrected Data/Run' + '{:03d}'.format(i) + '_a0.npy')

        add_data = np.squeeze(add_data)  # Squeeze out the single capture axis

        if CC:
            add_data = add_data[6:12]  # Grab just cc (or sec) bins
        else:
            add_data = add_data[0:6]  # Grab just sec bins

        for j in np.arange(1000):
            single_frame = add_data[:, j]  # Get the next view data
            total_data[0:5] = np.add(total_data[0:5], single_frame[0:5])  # Add to the current total data
            sum_single_frame = np.sum(single_frame, axis=0)  # Sum all bins to get summed cc (or sec)
            total_data[5] = np.add(total_data[5], sum_single_frame)  # Add to the total summed

            for k, img in enumerate(total_data):
                # Calculate the CNR (i-1 = file, k = bin, j = view/time point)
                CNR_pts[i-1, k, j], err = sct.cnr(img, contrast_mask, bg_mask)
                if j == 249:
                    CNR_err[i-1, k] = err

    CNR_pts = np.mean(CNR_pts, axis=0)  # Average over all of the files
    err_mean = np.mean(CNR_err, axis=0)
    err_std = np.std(CNR_err, axis=0)

    return time_pts, CNR_pts, err_mean, err_std


def get_CNR_over_time_energy_thresh(folder, threshold, corr3x3=False, CC=True,
                            directory='C:/Users/10376/Documents/Phantom Data/Uniformity/Multiple Energy Thresholds'):

    if corr3x3:
        contrast_mask = np.load(directory + folder + '/3x3_a0_Mask.npy')
        bg_mask = np.load(directory + folder + '/3x3_a0_Background.npy')
    else:
        contrast_mask = np.load(directory + folder + '/a0_Mask.npy')
        bg_mask = np.load(directory + folder + '/a0_Background.npy')

    time_pts = np.arange(0.001, 1.001, 0.001)  # Time points from 0.001 s to 10 s by 0.001 s increments
    CNR_pts = np.zeros([6, len(time_pts)])  # Collect CNR over 1 s for all 10 files

    if corr3x3:
        total_data = np.zeros([6, 8, 12])  # Holds the current data for all bins plus the sum of all bins
        add_data = np.load(directory + folder + '/3x3 Data/Thresholds_' + str(threshold) + '.npy')
    else:
        total_data = np.zeros([6, 24, 36])  # Holds the current data for all bins plus the sum of all bins
        add_data = np.load(directory + folder + '/Data/Thresholds_' + str(threshold) + '.npy')

    add_data = np.squeeze(add_data)  # Squeeze out the single capture axis

    if CC:
        add_data = add_data[6:12]  # Grab just cc (or sec) bins
    else:
        add_data = add_data[0:6]  # Grab just sec bins

    for j in np.arange(1000):
        single_frame = add_data[:, j]  # Get the next view data
        #total_data[0:5] = np.add(total_data[0:5], single_frame[0:5])  # Add to the current total data
        total_data[0:5] = single_frame[0:5]  # switch between the two if looking to average over every frame
        sum_single_frame = np.sum(single_frame, axis=0)  # Sum all bins to get summed cc (or sec)
        #total_data[5] = np.add(total_data[5], sum_single_frame)  # Add to the total summed
        total_data[5] = sum_single_frame

        for k, img in enumerate(total_data):
            # Calculate the CNR (i-1 = file, k = bin, j = view/time point)
            CNR_pts[k, j], err = sct.cnr(img, contrast_mask, bg_mask)

    CNR_err = np.std(CNR_pts, axis=1)
    CNR_pts = np.mean(CNR_pts, axis=1)

    return time_pts, CNR_pts, CNR_err


def plot_CNR_over_time_1s_multiple(time_pts, CNR_pts, CNR_err_mean, CC='CC', title='n/a', save=False,
                       directory='C:/Users/10376/Documents/Phantom Data/Uniformity/'):
    """

    :param time_pts:
    :param CNR_pts:
    :param CNR_err:
    :param title:
    :param save:
    :param directory:
    :return:
    """
    sns.set_style('whitegrid')
    fig, axes = plt.subplots(2, 3, figsize=(8, 6))
    ax1 = fig.add_subplot(111, frameon=False)
    ax1.grid(False)
    ax1.set_xticks([])
    ax1.set_yticks([])
    titles = ['20-30 keV', '30-50 keV', '50-70 keV', '70-90 keV', '90-120 keV', 'Sum ' + CC]

    max_CNR = np.max(CNR_pts) + 0.25
    for i, ax in enumerate(axes.flat):
        ax.plot(time_pts, CNR_pts[i], lw=1)
        ax.set_title(titles[i])
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_ylim([0, max_CNR])
        ax.set_xlabel('CNR error at 0.25 s is \n' + '{:.2f}'.format(CNR_err_mean[i]))

    plt.subplots_adjust(left=0.12, bottom=0.2, right=0.96, top=0.88, wspace=0.31, hspace=0.55)
    ax1.set_xlabel('Acquisition Time (s)', fontsize=14, labelpad=45)
    ax1.set_ylabel('CNR', fontsize=14, labelpad=30)
    ax1.set_title(title, fontsize=15, pad=25)
    plt.show()

    if save:
        plt.savefig(directory + '/Plots/CNR ' + title + '.png', dpi=fig.dpi)
        plt.close()


def plot_CNR_adj_bins(time_pts, CNR_pts, plottitles, title='n/a', save=False,
                      directory='C:/Users/10376/Documents/Phantom Data/Uniformity/'):
    """

    :param time_pts:
    :param CNR_pts:
    :param title:
    :param save:
    :param directory:
    :return:
    """
    sns.set_style('whitegrid')
    fig, axes = plt.subplots(2, 3, figsize=(8, 6))
    ax1 = fig.add_subplot(111, frameon=False)
    ax1.grid(False)
    ax1.set_xticks([])
    ax1.set_yticks([])

    max_CNR = np.max(CNR_pts) + 0.25
    for i, ax in enumerate(axes.flat):
        if i > len(CNR_pts) - 1:
            break
        ax.plot(time_pts, CNR_pts[i][0:len(time_pts)], lw=1)
        ax.set_title(plottitles[i])
        #ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_ylim([0, max_CNR])

    plt.subplots_adjust(left=0.12, bottom=0.2, right=0.96, top=0.88, wspace=0.31, hspace=0.55)
    ax1.set_xlabel('Acquisition Time (s)', fontsize=14, labelpad=45)
    ax1.set_ylabel('CNR', fontsize=14, labelpad=30)
    ax1.set_title(title, fontsize=15, pad=25)
    plt.show()

    if save:
        plt.savefig(directory + '/Plots/' + title + '.png', dpi=fig.dpi)
        plt.close()


def get_CNR_single_adj_bin(folder, bins, threshold, corr3x3=False, CC=True,
                                directory='C:/Users/10376/Documents/Phantom Data/Uniformity/'):

    if corr3x3:
        contrast_mask = np.load(directory + folder + '/3x3_a0_Mask.npy')
        bg_mask = np.load(directory + folder + '/3x3_a0_Background.npy')
    else:
        contrast_mask = np.load(directory + folder + '/a0_Mask.npy')
        bg_mask = np.load(directory + folder + '/a0_Background.npy')

    time_pts = np.arange(0.001, 1.001, 0.001)  # Time points from 0.001 s to 10 s by 0.001 s increments
    CNR_pts = np.zeros(len(time_pts))  # Collect CNR over 1 s for all 10 files

    if corr3x3:
        total_data = np.zeros([8, 12])  # Holds the current data for all bins plus the sum of all bins
        add_data = np.load(directory + folder + '/3x3 Data/Thresholds_' + str(threshold) + '.npy')
    else:
        total_data = np.zeros([24, 36])  # Holds the current data for all bins plus the sum of all bins
        add_data = np.load(directory + folder + '/Data/Thresholds_' + str(threshold) + '.npy')

    add_data = np.squeeze(add_data)  # Squeeze out the single capture axis

    if CC:
        add_data = add_data[6:12]  # Grab just cc (or sec) bins
    else:
        add_data = add_data[0:6]  # Grab just sec bins

    add_data = add_adj_bins(add_data, bins)  # sum the desired bins together
    add_data = add_data[bins[0]]  # Get just the new summed bin

    for j in np.arange(1000):
        single_frame = add_data[j]  # Get the next view data
        #total_data = np.add(total_data, single_frame)  # Add to the current total data
        total_data = single_frame  # Swtich if looking to average over every frame

        # Calculate the CNR (i-1 = file, k = bin, j = view/time point)
        CNR_pts[j], err = sct.cnr(total_data, contrast_mask, bg_mask)

    CNR_err = np.std(CNR_pts)
    CNR_pts = np.mean(CNR_pts)


    return time_pts, CNR_pts, CNR_err


def find_top_10(folder, sub, directory=r'C:\Users\10376\Documents\Phantom Data\Uniformity\Multiple Energy Thresholds/'):

    files = glob.glob(directory + folder + '/' + sub + '/Single Frame Avg/*')

    file_list = []
    num_list = np.zeros(len(files))

    for i, file in enumerate(files):
        file_list.append(file[-20:-4])
        num_list[i] = np.load(file)[0]

    high_files = []
    high_nums = np.zeros(15)
    for j in np.arange(15):
        idx = np.argmax(num_list)
        high_files.append(file_list[idx])
        high_nums[j] = num_list[idx]
        file_list = np.delete(file_list, idx)
        num_list = np.delete(num_list, idx)

    return high_files, high_nums


directory = r'C:\Users\10376\Documents\Phantom Data\Uniformity\Multiple Energy Thresholds/'
bintitles = [['20-30 keV', '30-50 keV', '50-70 keV', '70-90 keV', '90-120 keV', '20-120 keV'],
             ['20-30 keV', '30-40 keV', '40-50 keV', '50-60 keV', '60-70 keV', '20-70 keV'],
             ['20-35 keV', '35-50 keV', '50-65 keV', '65-80 keV', '80-90 keV', '20-90 keV'],
             ['25-35 keV', '35-45 keV', '45-55 keV', '55-65 keV', '65-75 keV', '25-75 keV'],
             ['25-40 keV', '40-55 keV', '55-70 keV', '70-80 keV', '80-95 keV', '25-95 keV'],
             ['30-45 keV', '45-60 keV', '60-75 keV', '75-85 keV', '85-95 keV', '30-95 keV'],
             ['20-30 keV', '30-70 keV', '70-85 keV', '85-100 keV', '100-120 keV', '20-120 keV']]

nbt = [[20, 30, 50, 70, 90, 120],
       [20, 30, 40, 50, 60, 70],
       [20, 35, 50, 65, 80, 90],
       [25, 35, 45, 55, 65, 75],
       [25, 40, 55, 70, 80, 95],
       [30, 45, 60, 75, 85, 95],
       [20, 30, 70, 85, 100, 120]]

files = ['1_20-30 keV.npy', '1_30-50 keV.npy', '1_50-70 keV.npy', '1_70-90 keV.npy', '1_90-120 keV.npy', '1_20-120 keV.npy']
tpts = np.arange(0.001, 0.251, 0.001)
diam = ['1 mm', '330 um']




    #for j, subfolder in enumerate(['3x3 Data', 'Data']):



        #data = np.zeros([6, 250])
        #for i in np.arange(6):
        #    data[i] = np.load(directory + '/' + folder + '/' + subfolder + '/Ranges CNR/' + files[i])[0:250]
        #title = folder + ' ' + diam[j]
       #plot_CNR_adj_bins(tpts, data, bintitles[0], title=title, save=True, directory=directory)

bins = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])

#for folder in ['/1w/', '/3w/']:
#    for three in [False, True]:
#        for i in np.arange(1, 8):
#            t, c, ce = get_CNR_over_time_energy_thresh(folder, i, corr3x3=three)
#            for j in np.arange(6):
#                if three:
#                    np.save(directory + folder + '/3x3 Data/Single Frame Avg/' + str(i) + '_' + bintitles[i-1][j] +
#                            '.npy', np.array([c[j], ce[j]]))
#                else:
#                    np.save(directory + folder + '/Data/Single Frame Avg/' + str(i) + '_' + bintitles[i-1][j] + '.npy',
#                            np.array([c[j], ce[j]]))

#            for b in bins:
#                print(b)
#                tb, cb, ceb = get_CNR_single_adj_bin(folder, b, i, corr3x3=three, directory=directory)
#                ths = nbt[i-1]
#                name = str(i) + '_' + str(ths[b[0]]) + '-' + str(ths[b[1]+1]) + ' kev.npy'
#                print(name)
#                if three:
#                    np.save(directory + folder + '/3x3 Data/Single Frame Avg/' + name, np.array([cb, ceb]))
#                else:
#                    np.save(directory + folder + '/Data/Single Frame Avg/' + name, np.array([cb, ceb]))

for folder in ['1w', '3w']:
    print(folder)
    print(find_top_10(folder, '3x3 Data'))
    print()
#%%
np.set_printoptions(precision=2)
for sub in ['Data', '3x3 Data']:
    for folder in ['/1w/', '/3w/']:
        path = directory + folder + sub + '/Single Frame Avg/'
        files = glob.glob(path + '*25-65*')
        print(folder + sub)
        temp = np.zeros([len(files), 2])
        for i, file in enumerate(files):
            temp[i] = np.load(file)
        print(np.mean(temp, axis=0))
        print()
#idx = np.argmax(num_list)
#print(num_list[idx])
#print(file_list[idx])
#print()

