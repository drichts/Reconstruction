import numpy as np
import matplotlib.pyplot as plt


#%% Flat field air

def flatfield_air():
    folder = 'C:/Users/10376/Documents/IEEE Abstract/Analysis Data/Flat Field/'

    print('1w           4w')
    w1 = np.load(folder + '/flatfield_1wA0.npy')
    w4 = np.load(folder + '/flatfield_4wA0.npy')

    w1 = np.squeeze(w1)
    w4 = np.squeeze(w4)

    w1 = np.sum(w1, axis=1)
    w4 = np.sum(w4, axis=1)

    for i in np.arange(5):
        print('Bin' + str(i))
        #print(np.nanstd(w1[i+6])/np.mean(w1[i+6]))
        #print(np.nanstd(w4[i+6])/np.mean(w4[i+6]))
        print(np.mean(w1[i+6]))
        print(np.mean(w4[i+6]))

    return w1
#x = flatfield_air()


#%% Flatfield phantoms (correct for air)

def flatfield_phantoms():
    directory = 'C:/Users/10376/Documents/IEEE Abstract/'
    load_folder = 'Raw Data/Flat Field\\'
    save_folder = 'Analysis Data/Flat Field/'

    load_path = directory + load_folder
    air1w = np.squeeze(np.load(load_path + '/flatfield_1w.npy'))
    air4w = np.squeeze(np.load(load_path + '/flatfield_4w.npy'))

    blue1w = np.squeeze(np.load(load_path + 'bluebelt_1w.npy'))
    blue4w = np.squeeze(np.load(load_path + 'bluebelt_4w.npy'))

    plexi1w = np.squeeze(np.load(load_path + 'plexiglass_1w.npy'))
    plexi4w = np.squeeze(np.load(load_path + 'plexiglass_4w.npy'))

    blue1w = np.sum(blue1w, axis=1)
    blue4w = np.sum(blue4w, axis=1)
    plexi1w = np.sum(plexi1w, axis=1)
    plexi4w = np.sum(plexi4w, axis=1)
    air1w = np.sum(air1w, axis=1)
    air4w = np.sum(air4w, axis=1)

    blue1w = np.divide(blue1w, air1w)
    blue4w = np.divide(blue4w, air4w)

    plexi1w = np.divide(plexi1w, air1w)
    plexi4w = np.divide(plexi4w, air4w)

    blue1w = -1*np.log(blue1w)
    blue4w = -1*np.log(blue4w)

    plexi1w = -1*np.log(plexi1w)
    plexi4w = -1*np.log(plexi4w)

    #np.save(directory + save_folder + 'bluebelt_1w.npy', blue1w)
    #np.save(directory + save_folder + 'bluebelt_4w.npy', blue4w)

    #np.save(directory + save_folder + 'plexiglass_1w.npy', plexi1w)
    #np.save(directory + save_folder + 'plexiglass_4w.npy', plexi4w)


    return blue1w, blue4w, plexi1w, plexi4w

#b1, b4, p1, p4 = flatfield_phantoms()

#%% Counts for multiple tube currents
def cc_sec_counts():
    directory = 'X:/Devon_UVic/ACS_Data/'
    a0 = np.load(directory + 'a0.npy')
    a1 = np.load(directory + 'a1.npy')
    currents = np.array([0.,  1.,  2.,  3.,  4.,  5., 10., 15., 20., 25])

    a0 = np.sum(a0, axis=2)
    a1 = np.sum(a1, axis=2)

    a0_cc = np.sum(a0[:, 7:12, :, :], axis=1)
    a0_sec = np.sum(a0[:, 0:6, :, :], axis=1)
    a1_cc = np.sum(a1[:, 7:12, :, :], axis=1)
    a1_sec = np.sum(a1[:, 0:6, :, :], axis=1)

    a0_cc_med = np.median(a0_cc, axis=(1, 2))
    a0_sec_med = np.median(a0_sec, axis=(1, 2))
    a1_cc_med = np.median(a1_cc, axis=(1, 2))
    a1_sec_med = np.median(a1_sec, axis=(1, 2))

    plt.scatter(currents, a0_cc_med)
    plt.scatter(currents, a1_cc_med)
    plt.show()

#%% Counts for multiple tube currents (Elik's data)
from scipy.interpolate import interp1d
directory = 'X:/Devon_UVic/ACS_Data/'
currents = np.array([0.,  1.,  2.,  3.,  4.,  5., 10., 15., 20., 25])

#a0_1w = loadmat(directory + 'uniformity1w.mat')
#a0_4wa = loadmat(directory + 'uniformity4wa.mat')
#a0_4wb = loadmat(directory + 'uniformity4wb.mat')

#a0_1w = a0_1w['cc_struct']['data'][0][0][0][0][0]
#a0_4wa = a0_4wa['cc_struct']['data'][0][0][0][0][0]
#a0_4wb = a0_4wb['cc_struct']['data'][0][0][0][0][0]

#np.save(directory + 'uniformity1w.npy', a0_1w)
#np.save(directory + 'uniformity4wa.npy', a0_4wa)
#np.save(directory + 'uniformity4wb.npy', a0_4wb)

a0_1w = np.load(directory + 'uniformity1w.npy')
a0_4wa = np.load(directory + 'uniformity4wa.npy')
a0_4wb = np.load(directory + 'uniformity4wb.npy')

a0_1w = np.sum(a0_1w, axis=2)
a0_4wa = np.sum(a0_4wa, axis=2)
a0_4wb = np.sum(a0_4wb, axis=2)

a0_1w_cc = np.sum(a0_1w[:, 7:12, :, :], axis=1)
a0_1w_sec = np.sum(a0_1w[:, 0:5, :, :], axis=1)
a0_4wa_cc = np.sum(a0_4wa[:, 7:12, :, :], axis=1)
a0_4wa_sec = np.sum(a0_4wa[:, 0:5, :, :], axis=1)
a0_4wb_cc = np.sum(a0_4wb[:, 7:12, :, :], axis=1)
a0_4wb_sec = np.sum(a0_4wb[:, 0:5, :, :], axis=1)

a0_1w_cc_med = np.median(a0_1w_cc, axis=(1, 2))
a0_1w_sec_med = np.median(a0_1w_sec, axis=(1, 2))
a0_4wa_cc_med = np.median(a0_4wa_cc, axis=(1, 2))
a0_4wa_sec_med = np.median(a0_4wa_sec, axis=(1, 2))
a0_4wb_cc_med = np.median(a0_4wb_cc, axis=(1, 2))
a0_4wb_sec_med = np.median(a0_4wb_sec, axis=(1, 2))

coeffs_1w_cc = np.polyfit(currents[0:6], a0_1w_cc_med[0:6], 1)
p_1w_cc = np.poly1d(coeffs_1w_cc)

coeffs_4w_cc = np.polyfit(currents[0:6], a0_4wa_cc_med[0:6], 1)
p_4w_cc = np.poly1d(coeffs_4w_cc)


#%% Curves for SEC, CC for 1w and 4w
xpts = np.linspace(0, 25, 25)
f1w_cc = interp1d(currents, a0_1w_cc_med, kind='quadratic')
cc_1w = f1w_cc(xpts)

f4w_sec = interp1d(currents, a0_1w_sec_med, kind='quadratic')
sec_1w = f4w_sec(xpts)

f4w_cc = interp1d(currents, a0_4wa_cc_med, kind='quadratic')
cc_4w = f4w_cc(xpts)

f4w_sec = interp1d(currents, a0_4wa_sec_med, kind='quadratic')
sec_4w = f4w_sec(xpts)
fig = plt.figure(figsize=(6, 6))
plt.plot(xpts, cc_1w, color='blue')
plt.plot(xpts, sec_1w, color='blue', ls='--')
plt.plot(xpts, p_1w_cc(xpts), color='blue', ls=':')
plt.plot(xpts, cc_4w, color='red')
plt.plot(xpts, sec_4w, color='red', ls='--')
plt.plot(xpts, p_4w_cc(xpts), color='red', ls=':')
plt.legend(['cc 1w', 'sec 1w', 'theoretical 1w', 'cc 4w', 'sec 4w', 'theoretical 4w'])
plt.xlim([0, 25])
plt.ylim([0, 3E7])
plt.ylabel('Counts')
plt.xlabel('mA')

plt.show()
#plt.savefig('X:/Devon_UVic/Figures/curves.png', dpi=fig.dpi)

#%% Heat map data

med_1w = np.array([a0_1w_cc_med[5], a0_1w_cc_med[6], a0_1w_cc_med[9]])
med_4w = np.array([a0_4wa_cc_med[5], a0_4wa_cc_med[6], a0_4wa_cc_med[9]])

theoretical_1w = p_1w_cc([5, 10, 25])
theoretical_4w = p_4w_cc([5, 10, 25])

# Get the ratio of measured counts/median counts
map5_1w = np.divide(a0_1w_cc[5], theoretical_1w[0])
map5_4w = np.divide(a0_4wa_cc[5], theoretical_4w[0])

map10_1w = np.divide(a0_1w_cc[6], theoretical_1w[1])
map10_4w = np.divide(a0_4wa_cc[6], theoretical_4w[1])

map25_1w = np.divide(a0_1w_cc[9], theoretical_1w[2])
map25_4w = np.divide(a0_4wa_cc[9], theoretical_4w[2])

#%% 1w plots
fig, ax = plt.subplots(2, 3, figsize=(8, 4))
ax1 = fig.add_subplot(111, frameon=False)
ax1.grid(False)
# Hide axes ticks
ax1.set_xticks([])
ax1.set_yticks([])

im0 = ax[0, 0].imshow(map5_1w, vmin=0.8, vmax=1.2, cmap='jet')
ax[0, 0].set_xticks([])
ax[0, 0].set_yticks([])
ax[0, 1].imshow(map10_1w, vmin=0.8, vmax=1.2, cmap='jet')
ax[0, 1].set_xticks([])
ax[0, 1].set_yticks([])
ax[0, 1].set_title('1w CC Counter (Measured/Theoretical OCR)', fontsize=15)
ax[0, 2].imshow(map25_1w, vmin=0.8, vmax=1.2, cmap='jet')
ax[0, 2].set_xticks([])
ax[0, 2].set_yticks([])

cbar_ax = fig.add_axes([0.25, 0.55, 0.5, 0.03])
#cbar_ax.tick_params(labelsize=14)
fig.colorbar(im0, cax=cbar_ax, orientation='horizontal')

ax[1, 0].hist(np.ndarray.flatten(map5_1w))
ax[1, 0].set_ylim([0, 650])
ax[1, 0].set_title('5 mA O/O > 20 keV')
ax[1, 1].hist(np.ndarray.flatten(map10_1w))
ax[1, 1].set_yticks([])
ax[1, 1].set_ylim([0, 650])
ax[1, 1].set_title('10 mA O/O > 20 keV')
ax[1, 2].hist(np.ndarray.flatten(map25_1w))
ax[1, 2].set_yticks([])
ax[1, 2].set_ylim([0, 650])
ax[1, 2].set_title('25 mA O/O > 20 keV')

plt.subplots_adjust(left=0.1, bottom=0.13, right=0.9, top=0.9, wspace=0.25, hspace=0.7)

plt.show()
#plt.savefig('X:/Devon_UVic/Figures/1w_heatmaps_OCR.png', dpi=fig.dpi)

#%% 4w plots
fig, ax = plt.subplots(2, 3, figsize=(8, 4))
ax1 = fig.add_subplot(111, frameon=False)
ax1.grid(False)
# Hide axes ticks
ax1.set_xticks([])
ax1.set_yticks([])

im0 = ax[0, 0].imshow(map5_4w, vmin=0.8, vmax=1.2, cmap='jet')
ax[0, 0].set_xticks([])
ax[0, 0].set_yticks([])
ax[0, 1].imshow(map10_4w, vmin=0.8, vmax=1.2, cmap='jet')
ax[0, 1].set_xticks([])
ax[0, 1].set_yticks([])
ax[0, 1].set_title('4w CC Counter (Measured/Theoretical OCR)', fontsize=15)
ax[0, 2].imshow(map25_4w, vmin=0.8, vmax=1.2, cmap='jet')
ax[0, 2].set_xticks([])
ax[0, 2].set_yticks([])

cbar_ax = fig.add_axes([0.25, 0.55, 0.5, 0.03])
#cbar_ax.tick_params(labelsize=14)
fig.colorbar(im0, cax=cbar_ax, orientation='horizontal')

ax[1, 0].hist(np.ndarray.flatten(map5_4w))
ax[1, 0].set_ylim([0, 650])
ax[1, 0].set_title('5 mA O/O > 20 keV')
ax[1, 1].hist(np.ndarray.flatten(map10_4w))
ax[1, 1].set_yticks([])
ax[1, 1].set_ylim([0, 650])
ax[1, 1].set_title('10 mA O/O > 20 keV')
ax[1, 2].hist(np.ndarray.flatten(map25_4w))
ax[1, 2].set_yticks([])
ax[1, 2].set_ylim([0, 650])
ax[1, 2].set_title('25 mA O/O > 20 keV')

plt.subplots_adjust(left=0.1, bottom=0.13, right=0.9, top=0.9, wspace=0.25, hspace=0.7)

plt.show()
#plt.savefig('X:/Devon_UVic/Figures/4w_heatmaps_OCR.png', dpi=fig.dpi)

