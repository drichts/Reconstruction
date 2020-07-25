import numpy as np
import matplotlib.pyplot as plt

direct = r'C:\Users\10376\Documents\Phantom Data\UNIFORMITY/'
folders = ['many_thresholds_BB4mm', 'many_thresholds_BB2mm', 'many_thresholds_BB1mm',
           'many_thresholds_glass2mm', 'many_thresholds_glass1mm',
           'many_thresholds_steel07mm', 'many_thresholds_steel2mm',
           'many_thresholds_PP']

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


def plot_values(folder, cnr_or_noise=0, time_idx=0):
    fig = plt.figure(figsize=(8, 4))

    data = np.load(direct + folder + '/AHighCNR.npz')
    ec = np.mean(data['ec_cnr'][time_idx, 0])

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

    pts = np.arange(len(vals))
    plt.bar(pts, vals, yerr=uncer, capsize=3)
    plt.plot(np.linspace(-2, len(vals)+1), ec*np.ones(len(np.linspace(-2, len(vals)+1))), color='r')
    plt.legend(['EC CNR'], fontsize=12)
    plt.xlabel('Energy bins (keV)', fontsize=12)
    plt.ylabel('CNR', fontsize=12)
    plt.xticks(pts, titles)
    plt.xlim([-1, len(vals)])
    plt.tick_params(labelsize=11)
    plt.subplots_adjust(bottom=0.12, top=0.95)
    plt.plot()


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