import numpy as np
import matplotlib.pyplot as plt

folder = r'D:\Research\Python Data\Redlen\Attenuation'
fig = plt.figure(figsize=(5, 5))

steel = np.load(folder + '/steel.npy')
glass = np.load(folder + '/sodalimeglass.npy')
pmma = np.load(folder + '/PMMA.npy')
pp = np.load(folder + '/PP.npy')
water = np.load(folder + '/solid water.npy')
pvc = np.load(folder + '/PVC.npy')

steel[:, 1] = steel[:, 1] * 7.85
glass[:, 1] = glass[:, 1] * 2.52
pmma[:, 1] = pmma[:, 1] * 1.18
pp[:, 1] = pp[:, 1] * 0.855
water[:, 1] = water[:, 1] * 1.013
pvc[:, 1] = pvc[:, 1] * 1.38

plt.semilogy(pmma[:, 0]*1000, pmma[:, 1], color='b')
plt.semilogy(pp[:, 0]*1000, pp[:, 1], color='g')
plt.legend(['Acrylic', 'Polypropylene'], fontsize=12)

# plt.semilogy(steel[:, 0]*1000, steel[:, 1], color='r')
# plt.semilogy(glass[:, 0]*1000, glass[:, 1], color='g')
# plt.semilogy(pvc[:, 0]*1000, pvc[:, 1], color='b')
# plt.semilogy(water[:, 0]*1000, water[:, 1], color='k')
plt.xlim([20, 120])
plt.ylim([1E-1, 1E0])
# plt.legend(['Steel', 'Glass', 'Blue belt', 'Solid water'], fontsize=12)

plt.xlabel('Energy (keV)', fontsize=12)
plt.ylabel(r'$\mu$ (cm$^{-1}$)', fontsize=12)
plt.subplots_adjust(left=0.20, right=0.95)
plt.tick_params(labelsize=11)
plt.plot()
plt.savefig(folder + '/Acrylic.png', dpi=fig.dpi)

#%%
# file = '/PVC.txt'
#
# f = open(folder + file, 'rt')
#
# matrix = []
# for line in f:
#     col = line.split()
#     col = np.array(col)
#     matrix.append(col)
#
# matrix = np.array(matrix, dtype='float')
#
# np.save(folder + file[:-4] + '.npy', matrix)
