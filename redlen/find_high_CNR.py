import numpy as np
import matplotlib.pyplot as plt

direct = r'C:\Users\10376\Documents\Phantom Data\UNIFORMITY/'
folders = ['many_thresholds_BB4mm', 'many_thresholds_BB2mm', 'many_thresholds_BB1mm',
           'many_thresholds_glass2mm', 'many_thresholds_glass1mm',
           'many_thresholds_steel07mm', 'many_thresholds_steel2mm',
           'many_thresholds_PP']

names = ['BB4', 'BB2', 'BB1', 'glass2', 'glass1', 'steel07', 'steel2', 'PP']

titles_many = [['20-30', '30-50', '50-70', '70-90', '90-120', 'EC'],
               ['20-40', '40-60', '60-80', '80-100', '100-120', 'EC'],
               ['25-45', '45-65', '65-85', '85-105', '105-120', 'EC'],
               ['20-35', '35-55', '55-75', '75-95', '95-120', 'EC'],
               ['20-50', '50-80', '80-100', '100-110', '110-120', 'EC'],
               ['20-30', '30-60', '60-90', '90-100', '100-120', 'EC'],
               ['20-40', '40-70', '70-100', '100-110', '110-120', 'EC'],
               ['20-60', '60-70', '70-80', '80-90', '90-120', 'EC'],
               ['20-70', '70-80', '80-90', '90-100', '100-120', 'EC'],
               ['20-80', '80-90', '90-100', '100-110', '110-120', 'EC'],
               ['20-90', '90-100', '100-105', '105-110', '110-120', 'EC'],
               ['20-30', '30-70', '70-80', '80-90', '90-120', 'EC'],
               ['20-30', '30-80', '80-90', '90-100', '100-120', 'EC'],
               ['20-30', '30-90', '90-100', '100-110', '110-120', 'EC'],
               ['20-40', '40-80', '80-90', '90-100', '100-120', 'EC'],
               ['20-40', '40-90', '90-100', '100-110', '110-120', 'EC'],
               ['20-50', '50-90', '90-100', '100-110', '110-120', 'EC']]


def plot_values(folder, name, cnr_or_noise=0, time_idx=0):

    data = np.load(direct + folder + '/AHighCNR.npz')
    ec = np.mean(data['ec_cnr'][time_idx, 0])

    time_values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 50, 100, 250, 500, 1000])

    temp_titles = np.ndarray.flatten(np.array(titles_many)[:, 0:5])
    temp_vals = data['cnr'][time_idx, 0]
    temp_uncer = data['cnr'][time_idx, 1]

    vals = []
    uncer = []
    titles = []
    for i in np.arange(85):
        if temp_vals[i] > ec:
            vals.append(temp_vals[i])
            uncer.append(temp_uncer[i])
            titles.append(temp_titles[i])
    if len(vals) > 10:
        fig = plt.figure(figsize=(9, 5))
    else:
        fig = plt.figure(figsize=(8, 5))

    pts = np.arange(len(vals))
    plt.bar(pts, vals, yerr=uncer, capsize=3)
    plt.plot(np.linspace(-2, len(vals)+1), ec*np.ones(len(np.linspace(-2, len(vals)+1))), color='r')
    plt.legend(['EC CNR'], fontsize=16)
    plt.xlabel('Energy bins (keV)', fontsize=16, labelpad=10)
    plt.ylabel('CNR', fontsize=16)
    plt.xticks(pts, titles)
    plt.xlim([-1, len(vals)])
    plt.ylim([0, np.max(vals)*1.5])
    plt.tick_params(labelsize=14)
    plt.title(f'{time_values[time_idx]} ms', fontsize=18)
    plt.subplots_adjust(bottom=0.18, top=0.88, right=0.95)
    plt.plot()
    # plt.savefig(r'C:\Users\10376\Documents\Phantom Data\Report\Ranges/' + name + f'_{time_values[time_idx]}ms.png',
    #             dpi=fig.dpi)
    # plt.close()


def get_values():
    for folder in folders:
        cnr = np.zeros([13, 2, 85])  # Form: <time, value and error, test_num + bin>
        noise = np.zeros([13, 2, 85])
        ec_cnr = np.zeros([13, 2, 17])  # Form: <time, value and error, test_num
        ec_noise = np.zeros([13, 2, 17])

        for i in np.arange(1, 18):
            cnr_path = direct + folder + f'/TestNum{i}_cnr_time.npy'
            noise_path = direct + folder + f'/TestNum{i}_noise_time.npy'

            # Both of the form <pixels, bin, value or error, time>
            temp_cnr = np.load(cnr_path)[0]  # Only 1x1 pixels
            temp_noise = np.load(noise_path)[0]

            ec_cnr[:, :, i-1] = np.transpose(temp_cnr[12, :, 0:13], axes=(1, 0))
            ec_noise[:, :, i-1] = np.transpose(temp_noise[12, :, 0:13], axes=(1, 0))

            cnr[:, :, (i-1)*5:(i-1)*5+5] = np.transpose(temp_cnr[6:11, :, 0:13], axes=(2, 1, 0))
            noise[:, :, (i-1)*5:(i-1)*5+5] = np.transpose(temp_noise[6:11, :, 0:13], axes=(2, 1, 0))

        np.savez(direct+folder+'/AHighCNR.npz', cnr=cnr, noise=noise, ec_cnr=ec_cnr, ec_noise=ec_noise)

# for idx, folder in enumerate(folders):
#     for t in [0, 4, 9, 10]:
#         plot_values(folder, names[idx], time_idx=t)
