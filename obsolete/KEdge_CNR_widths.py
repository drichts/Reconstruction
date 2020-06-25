import numpy as np
import matplotlib.pyplot as plt
from sCT_Analysis import cnr

directory = 'D:/Research/Python Data/Spectral CT/'
folder1 = 'AuGd_width_5_12-2-19/'
folder2 = 'AuGd_width_10_12-2-19/'

start, stop = 11, 19
rng = stop - start

path5 = directory + folder1
path10 = directory + folder2

vials5 = np.load(path5 + 'Vial_Masks.npy')
vials10 = np.load(path10 + 'Vial_Masks.npy')

Au3w5 = np.zeros(rng)
Au05w5 = np.zeros(rng)
AuMw5 = np.zeros(rng)
Au3w10 = np.zeros(rng)
Au05w10 = np.zeros(rng)
AuMw10 = np.zeros(rng)

Gd3w5 = np.zeros(rng)
Gd05w5 = np.zeros(rng)
GdMw5 = np.zeros(rng)
Gd3w10 = np.zeros(rng)
Gd05w10 = np.zeros(rng)
GdMw10 = np.zeros(rng)

for i in np.arange(start, stop):
    imgGd5 = np.load(path5 + 'K-Edge/Bin1-0_Slice' + str(i) + '.npy')
    imgGd10 = np.load(path10 + 'K-Edge/Bin1-0_Slice' + str(i) + '.npy')
    imgAu5 = np.load(path5 + 'K-Edge/Bin4-3_Slice' + str(i) + '.npy')
    imgAu10 = np.load(path10 + 'K-Edge/Bin4-3_Slice' + str(i) + '.npy')

    Au3w5[i-start] = cnr(imgAu5, vials5[1], vials5[0])
    Au05w5[i-start] = cnr(imgAu5, vials5[5], vials5[0])
    AuMw5[i-start] = cnr(imgAu5, vials5[3], vials5[0])
    Au3w10[i-start] = cnr(imgAu10, vials10[1], vials10[0])
    Au05w10[i-start] = cnr(imgAu10, vials10[5], vials10[0])
    AuMw10[i-start] = cnr(imgAu10, vials10[3], vials10[0])

    Gd3w5[i-start] = cnr(imgGd5, vials5[4], vials5[0])
    Gd05w5[i-start] = cnr(imgGd5, vials5[2], vials5[0])
    GdMw5[i-start] = cnr(imgGd5, vials5[3], vials5[0])
    Gd3w10[i-start] = cnr(imgGd10, vials10[4], vials10[0])
    Gd05w10[i-start] = cnr(imgGd10, vials10[2], vials10[0])
    GdMw10[i-start] = cnr(imgGd10, vials10[3], vials10[0])

#np.save(path5 + 'Gd_3P_CNR.npy', Gd3w5)
#np.save(path5 + 'Gd_0.5P_CNR.npy', Gd05w5)
#np.save(path5 + 'Gd_Mixed_CNR.npy', GdMw5)
#print(Au3w5, Au05w5, AuMw5, Au3w10, Au05w10, AuMw10, Gd3w5, Gd05w5, GdMw5, Gd3w10, Gd05w10, GdMw10)
Au3w5 = np.mean(Au3w5)
Au05w5 = np.mean(Au05w5)
AuMw5 = np.mean(AuMw5)
Au3w10 = np.mean(Au3w10)
Au05w10 = np.mean(Au05w10)
AuMw10 = np.mean(AuMw10)

Gd3w5 = np.mean(Gd3w5)
Gd05w5 = np.mean(Gd05w5)
GdMw5 = np.mean(GdMw5)
Gd3w10 = np.mean(Gd3w10)
Gd05w10 = np.mean(Gd05w10)
GdMw10 = np.mean(GdMw10)

print(Au3w5, Au05w5, AuMw5, Au3w10, Au05w10, AuMw10, Gd3w5, Gd05w5, GdMw5, Gd3w10, Gd05w10, GdMw10)

#%%

folder = 'AuGd_width_5_12-2-19/'
start, stop = 11, 20
path = directory + folder

vials = np.load(path + 'Vial_Masks.npy')

mask = np.array([vials[4], vials[2], vials[3], vials[1]])
CNR = np.zeros(4)

for i in np.arange(4):
    temp = np.zeros(stop-start)
    for j in np.arange(start, stop):
        file = np.load(path + 'K-Edge/Bin' + str(i+1) + '-' + str(i) + '_Slice' + str(j) + '.npy')
        temp[j-start] = cnr(file, mask[i], vials[0])
    CNR[i] = np.mean(temp)

print(CNR)