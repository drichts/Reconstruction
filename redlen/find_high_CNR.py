import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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


def plot_values(time_val=25):

    time_values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 50, 100, 250, 500, 1000])
    time_idx = np.squeeze(np.argwhere(time_values == time_val))
    fig, axes = plt.subplots(2, 2, figsize=(7, 6))
    plot_titles = ['Blue belt', 'Glass', 'Steel', 'Polypropylene']

    ax = axes.flatten()

    for idx, folder in enumerate([folders[0], folders[3], folders[6], folders[7]]):
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
        vals = np.array(vals)
        uncer = np.array(uncer)
        titles = np.array(titles)

        sort_idx = np.argsort(vals)
        vals = vals[sort_idx]
        uncer = uncer[sort_idx]
        titles = titles[sort_idx]
        if len(vals) < 3:
            pts = np.arange(1, 3)
            ax[idx].bar(pts, vals[-2:], yerr=uncer[-2:], capsize=6)
            ax[idx].set_xticks(pts)
            ax[idx].set_xticklabels(titles[-2:])
            ax[idx].set_xlim([0, 3])
        else:
            pts = np.arange(1, 4)
            ax[idx].bar(pts, vals[-3:], yerr=uncer[-3:], capsize=6)
            ax[idx].set_xticks(pts)
            ax[idx].set_xticklabels(titles[-3:])
            ax[idx].set_xlim([0, 4])


        ax[idx].plot(np.arange(-2, 5), ec * np.ones(7), color='r')
        ax[idx].legend(['EC CNR'], fontsize=13)
        ax[idx].set_ylim([0, np.max(vals)*1.5])
        ax[idx].tick_params(labelsize=13)
        ax[idx].set_title(plot_titles[idx], fontsize=14)
    plt.subplots_adjust(top=0.95, wspace=0.29, hspace=0.37)
    fig.text(0.5, 0.02, 'Energy bins (keV)', ha='center', fontsize=16)
    fig.text(0.02, 0.5, 'CNR', va='center', rotation='vertical', fontsize=16)
    plt.plot()
    plt.savefig(r'C:\Users\10376\Documents\Phantom Data\Report\Ranges/' + f'_{time_values[time_idx]}ms.png', dpi=fig.dpi)
    plt.close()


def get_values():
    sns.set(style='white')
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
