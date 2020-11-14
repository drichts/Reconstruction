import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from glob import glob

sns.set_style('whitegrid')

folder = r'D:\OneDrive - University of Victoria\Research\Attenuation Data\Soft Tissue'

files = glob(os.path.join(folder, '*.npy'))

fig = plt.figure(figsize=(6, 6))
leg = []
for file in files:
    data = np.load(file)
    leg.append(file.replace(folder, '')[1:-4])
    plt.semilogy(data[:, 0]*1000, data[:, 1])

plt.ylabel(r'$\mu/\rho$ cm$^2$/g', fontsize=17)
plt.xlabel('Energy (kev)', fontsize=17)
plt.xlim([10, 160])
plt.legend(leg, fontsize=14)
plt.tick_params(labelsize=13)
plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95)
plt.savefig(folder + '/plotted.png', dpi=fig.dpi)
plt.show()
