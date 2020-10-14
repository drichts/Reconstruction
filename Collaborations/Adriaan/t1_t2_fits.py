import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
import seaborn as sns

from scipy.optimize import curve_fit


def t_eq(x, A1, T1, y0):
    return A1 * np.exp(-(x/T1)) + y0


folder = r'D:\OneDrive - University of Victoria\Files\Grad School\Collaborations\Adriaan/'

t1_files = glob(folder + '\*t1*')
t2_files = glob(folder + '\*t2*')

t1_data = np.array([[0.001, 0.01, 0.1, 0.25, 0.5, 1, 1.5, 2, 3, 4, 5, 7.5, 10, 15, 20, 30],
                   [-1.3967E10, -1.387E10, -1.2926E10, -1.1422E10, -9.0922E09, -5.0239E09, -1.6343E09, 1.1914E09,
                    5.5216E09, 8.5292E09, 1.0624E10, 1.3479E10, 1.4631E10, 1.5283E10, 1.5388E10, 1.5407E10],
                   [-1.3976E10, -1.3837E10, -1.248E10, -1.0351E10, -7.1574E09, -4.3586E09, -1.9082E09, 2.1171E09,
                    5.2116E09, 9.4153E09, 1.1891E10, 1.3353E10, 1.4894E10, 1.5304E10, 1.5442E10, 1.5444E10],
                    [0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 7.5, 10, 15, 22]])


t2_data = np.array([[2, 4, 6, 8, 10, 12, 16, 24, 32, 40, 48, 56, 64, 96, 128, 160],
                    [4.2628E10, 4.0448E10, 3.8056E10, 3.5697E10, 3.3673E10, 3.1975E10, 2.8861E10, 2.2885E10, 1.8313E10,
                     1.4697E10, 1.1701E10, 9.4002E09, 7.5372E09, 3.1453E09, 1.3325E09, 5.6286E08],
                    [3.9719E10, 3.676E10, 3.3923E10, 3.1109E10, 2.8648E10, 2.6532E10, 2.3008E10, 1.6714E10, 1.2387E10,
                     9.0212E09, 6.6817E09, 4.906E09, 3.6278E09, 2.6949E09, 2.0058E09, 9.7081E08],
                    [2, 4, 6, 8, 10, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 100]])

sns.set_style('whitegrid')

fig, ax = plt.subplots(1, 2, figsize=(8, 3))
tpts = np.linspace(0.001, 30, 1000)

popt_b, pcov_b = curve_fit(t_eq, t1_data[0], t1_data[1], p0=[-1E10, 1, 1E10])
popt_a, pcov_a = curve_fit(t_eq, t1_data[3], t1_data[2], p0=[-1E10, 1, 1E10])

res_b = t1_data[1] - t_eq(t1_data[0], *popt_b)
res_a = t1_data[2] - t_eq(t1_data[3], *popt_a)

ss_res_b = np.sum(res_b**2)
ss_res_a = np.sum(res_a**2)

ss_tot_b = np.sum((t1_data[1] - np.mean(t1_data[1]))**2)
ss_tot_a = np.sum((t1_data[2] - np.mean(t1_data[2]))**2)

r_sq_b = 1 - (ss_res_b / ss_tot_b)
r_sq_a = 1 - (ss_res_a / ss_tot_a)

perr_b = np.sqrt(np.diag(pcov_b))
perr_a = np.sqrt(np.diag(pcov_a))

ax[0].scatter(t1_data[0], t1_data[1], color='r')
ax[0].plot(tpts, t_eq(tpts, *popt_b), color='b')
ax[0].set_title('T1 Before', fontsize=14)
ax[0].set_xlabel('Time (s)', fontsize=12)
ax[0].set_ylabel(r'Integrated counts, H$_2$O peak', fontsize=12)
ax[0].annotate(r'Eq: $y = A1*exp(-x/T1) + y0$', (8, 0.5E10), xycoords='data', fontsize=9)
ax[0].annotate(r'$A1 = $ %.4e $\pm$ %.4e' % (popt_b[0], perr_b[0]), (8, 0.2E10), xycoords='data', fontsize=9)
ax[0].annotate(r'$T1 = $ %.4e $\pm$ %.4e' % (popt_b[1], perr_b[1]), (8, -0.1E10), xycoords='data', fontsize=9)
ax[0].annotate(r'$y0 = $ %.4e $\pm$ %.4e' % (popt_b[2], perr_b[2]), (8, -0.4E10), xycoords='data', fontsize=9)
ax[0].annotate(r'$R^2 = $ %.3f' % r_sq_b, (10, -0.7E10), xycoords='data', fontsize=9)

ax[1].scatter(t1_data[3], t1_data[2], color='r')
ax[1].plot(tpts, t_eq(tpts, *popt_a), color='b')
ax[1].set_title('T1 After', fontsize=14)
ax[1].set_xlabel('Time (s)', fontsize=12)
ax[1].set_ylabel(r'Integrated counts, H$_2$O peak', fontsize=12)
ax[1].annotate(r'Eq: $y = A1*exp(-x/T1) + y0$', (8, 0.5E10), xycoords='data', fontsize=9)
ax[1].annotate(r'$A1 = $ %.4e $\pm$ %.4e' % (popt_a[0], perr_a[0]), (8, 0.2E10), xycoords='data', fontsize=9)
ax[1].annotate(r'$T1 = $ %.4e $\pm$ %.4e' % (popt_a[1], perr_a[1]), (8, -0.1E10), xycoords='data', fontsize=9)
ax[1].annotate(r'$y0 = $ %.4e $\pm$ %.4e' % (popt_a[2], perr_a[2]), (8, -0.4E10), xycoords='data', fontsize=9)
ax[1].annotate(r'$R^2 = $ %.3f' % r_sq_a, (8, -0.7E10), xycoords='data', fontsize=9)

plt.subplots_adjust(bottom=0.17, top=0.9, right=0.95, wspace=0.33)
plt.savefig(folder + 'T1.png', dpi=fig.dpi)
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(8, 3))
tpts = np.linspace(2, 160, 1000)

popt_b, pcov_b = curve_fit(t_eq, t2_data[0], t2_data[1], p0=[1E10, 1, 1E8])
popt_a, pcov_a = curve_fit(t_eq, t2_data[3], t2_data[2], p0=[1E10, 1, 1E8])

res_b = t2_data[1] - t_eq(t2_data[0], *popt_b)
res_a = t2_data[2] - t_eq(t2_data[3], *popt_a)

ss_res_b = np.sum(res_b**2)
ss_res_a = np.sum(res_a**2)

ss_tot_b = np.sum((t2_data[1] - np.mean(t2_data[1]))**2)
ss_tot_a = np.sum((t2_data[2] - np.mean(t2_data[2]))**2)

r_sq_b = 1 - (ss_res_b / ss_tot_b)
r_sq_a = 1 - (ss_res_a / ss_tot_a)

perr_b = np.sqrt(np.diag(pcov_b))
perr_a = np.sqrt(np.diag(pcov_a))

ax[0].scatter(t2_data[0], t2_data[1], color='r')
ax[0].plot(tpts, t_eq(tpts, *popt_b), color='b')
ax[0].set_title('T2 Before', fontsize=14)
ax[0].set_xlabel('Time (s)', fontsize=12)
ax[0].set_ylabel(r'Integrated counts, H$_2$O peak', fontsize=12)
ax[0].annotate(r'Eq: $y = A1*exp(-x/T2) + y0$', (45, 3.7E10), xycoords='data', fontsize=9)
ax[0].annotate(r'$A1 = $ %.4e $\pm$ %.4e' % (popt_b[0], perr_b[0]), (45, 3.3E10), xycoords='data', fontsize=9)
ax[0].annotate(r'$T2 = $ %.4e $\pm$ %.4e' % (popt_b[1], perr_b[1]), (45, 2.9E10), xycoords='data', fontsize=9)
ax[0].annotate(r'$y0 = $ %.4e $\pm$ %.4e' % (popt_b[2], perr_b[2]), (45, 2.5E10), xycoords='data', fontsize=9)
ax[0].annotate(r'$R^2 = $ %.3f' % r_sq_b, (45, 2.1E10), xycoords='data', fontsize=9)

ax[1].scatter(t2_data[3], t2_data[2], color='r')
ax[1].plot(tpts, t_eq(tpts, *popt_a), color='b')
ax[1].set_title('T2 After', fontsize=14)
ax[1].set_xlabel('Time (s)', fontsize=12)
ax[1].set_ylabel(r'Integrated counts, H$_2$O peak', fontsize=12)
ax[1].annotate(r'Eq: $y = A1*exp(-x/T2) + y0$', (45, 3.7E10), xycoords='data', fontsize=9)
ax[1].annotate(r'$A1 = $ %.4e $\pm$ %.4e' % (popt_a[0], perr_a[0]), (45, 3.3E10), xycoords='data', fontsize=9)
ax[1].annotate(r'$T2 = $ %.4e $\pm$ %.4e' % (popt_a[1], perr_a[1]), (45, 2.9E10), xycoords='data', fontsize=9)
ax[1].annotate(r'$y0 = $ %.4e $\pm$ %.4e' % (popt_a[2], perr_a[2]), (45, 2.5E10), xycoords='data', fontsize=9)
ax[1].annotate(r'$R^2 = $ %.3f' % r_sq_a, (45, 2.1E10), xycoords='data', fontsize=9)

plt.subplots_adjust(bottom=0.17, top=0.9, right=0.95, wspace=0.33)
plt.savefig(folder + 'T2.png', dpi=fig.dpi)
plt.show()

