import numpy as np
import matplotlib.pyplot as plt

dir = r'C:\Users\10376\Documents\Phantom Data\UNIFORMITY'

folder = '/many_thresholds_steel07mm/'
airfolder = '/many_thresholds_airscan/'

data = np.load(dir + folder + 'TestNum1_DataA0.npy')
airdata = np.load(dir + airfolder + 'TestNum1_DataA0.npy')

airdata = np.sum(airdata, axis=1) / 40

cnr = np.zeros([7, 40])
noise = np.zeros([7, 40])
signal = np.zeros([7, 40])

bg = np.zeros([7, 4])
roi = np.zeros([7, 2])

airbg = np.zeros([7, 4])
airroi = np.zeros([7, 2])

brtwo = 8
brthree = brtwo+6
brfour = brthree+6

bcone = 0
bctwo = bcone+6
bcthree = bctwo+6
bcfour = bcthree+6
bcfive = bcfour+6

rrtwo = 6
rrthree = rrtwo+6
rrfour = rrthree+6

rcone = 0
rctwo = rcone+6
rcthree = rctwo+6
rcfour = rcthree+6
rcfive = rcfour+6


for idx, i in enumerate(np.arange(0, 1000, 25)):
    tdata = np.sum(data[:, i:i+25], axis=1)

    for j, b in enumerate([6, 7, 8, 9, 10, 11, 12]):
        bg[j, 0] = np.sum(tdata[b, brtwo:brthree, bcone:bctwo])
        airbg[j, 0] = np.sum(airdata[b, brtwo:brthree, bcone:bctwo])
        # roi[j, 0] = np.sum(tdata[b, rrtwo:rrthree, rcthree:rcfour])
        # airroi[j, 0] = np.sum(airdata[b, rrtwo:rrthree, rcthree:rcfour])

        bg[j, 1] = np.sum(tdata[b, brtwo:brthree, bctwo:bcthree])
        airbg[j, 1] = np.sum(airdata[b, brtwo:brthree, bctwo:bcthree])
        roi[j, 0] = np.sum(tdata[b, rrtwo:rrthree, rcfour:rcfive])
        airroi[j, 0] = np.sum(airdata[b, rrtwo:rrthree, rcfour:rcfive])

        bg[j, 2] = np.sum(tdata[b, brthree:brfour, bcone:bctwo])
        airbg[j, 2] = np.sum(airdata[b, brthree:brfour, bcone:bctwo])
        # roi[j, 2] = np.sum(tdata[b, rrthree:rrfour, rcthree:rcfour])
        # airroi[j, 2] = np.sum(airdata[b, rrthree:rrfour, rcthree:rcfour])

        bg[j, 3] = np.sum(tdata[b, brthree:brfour, bctwo:bcthree])
        airbg[j, 3] = np.sum(airdata[b, brthree:brfour, bctwo:bcthree])
        roi[j, 1] = np.sum(tdata[b, rrthree:rrfour, rcfour:rcfive])
        airroi[j, 1] = np.sum(airdata[b, rrthree:rrfour, rcfour:rcfive])

    bg = -1*np.log(bg/airbg)
    roi = -1*np.log(roi/airroi)
    #print(np.mean(bg, axis=1) - np.mean(roi, axis=1))

    for b in np.arange(7):
        cnr[b, idx] = np.abs(np.mean(roi[b]) - np.mean(bg[b])) / np.std(bg[b])
        noise[b, idx] = np.std(bg[b])
        signal[b, idx] = np.mean(bg[b])

#cnr[2, 26] = np.nan
#noise = np.divide(noise, signal) * 100
print(np.nanmean(cnr, axis=1))
print(np.nanstd(cnr, axis=1))
print()
print(np.load(dir + folder + 'TestNum1_cnr_time.npy')[4, 6:13, 0, 12])
print(np.load(dir + folder + 'TestNum1_cnr_time.npy')[4, 6:13, 1, 12])
# print()
# print(np.mean(noise, axis=1))
# print(np.std(noise, axis=1))
# print()
# print(np.load(dir + folder + 'TestNum1_noise_time.npy')[4, 6:13, 0, 12])
# print(np.load(dir + folder + 'TestNum1_noise_time.npy')[4, 6:13, 1, 12])


x = np.load(dir + folder + 'TestNum1_cnr_time.npy')
y = np.load(dir + folder + 'TestNum1_noise_time.npy')
z = np.load(dir + folder + 'TestNum1_signal.npy')
save = True
if save:
    x[4, 6:, 0, 12] = np.nanmean(cnr, axis=1)
    x[4, 6:, 1, 12] = np.nanstd(cnr, axis=1)

    y[4, 6:, 0, 12] = np.mean(noise, axis=1)
    y[4, 6:, 1, 12] = np.std(noise, axis=1)

    z[4, 6:, 0, 12] = np.mean(signal, axis=1)
    z[4, 6:, 1, 12] = np.std(signal, axis=1)

    np.save(dir + folder + 'TestNum1_cnr_time.npy', x)
    np.save(dir + folder + 'TestNum1_noise_time.npy', y)
    np.save(dir + folder + 'TestNum1_signal.npy', z)
