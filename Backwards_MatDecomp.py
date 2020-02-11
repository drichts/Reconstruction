import numpy as np
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate

# Z eff value of water
Zw = 8.11
Zw4 = Zw*Zw*Zw*Zw
atten_folder = 'D:/Research/Attenuation Data/NPY Attenuation/'
spectra_folder = 'D:/Research/Attenuation Data/NPY Spectra/'

NA = 6.022E23

# Matrices for the scattering data based on Z (rows) and E (columns)
example_file = np.load(atten_folder + 'Z04.npy')
Z_values = np.arange(4, 31, 1)
num_rows = len(Z_values)
num_cols = len(example_file)

# Initialize the scattering and photoelectric matrices
scatter_matrix = np.empty([num_rows, num_cols])
PE_matrix = np.empty([num_rows, num_cols])

# Load the beam energies and weights
spectra_energies = example_file[:, 0]
wts_low = np.load(spectra_folder + '40kVp_weights.npy')
wts_high = np.load(spectra_folder + '80kVp_weights.npy')

num_energies = len(spectra_energies)

# Populate the matrices with the correct data
for z in Z_values:

    # Load the file of a specific set of Z attenuation data
    if z < 10:
        data = np.load(atten_folder+'Z0' + str(z) + '.npy')
    else:
        data = np.load(atten_folder+'Z' + str(z) + '.npy')

    scatter_matrix[z-4] = np.add(data[:, 1], data[:, 2])
    PE_matrix[z-4] = data[:, 3]

# Normalize the scatter and PE values to F and G
atomic_mass = np.array([9.01218, 10.81, 12.011, 14.0067, 15.9994, 18.998403, 20.179, 22.98977, 24.305, 26.98154,
                        28.0855, 30.97376, 32.06, 35.453, 39.948, 39.0983, 40.08, 44.9559, 47.90, 50.9415, 51.996,
                        54.9380, 55.847, 58.9332, 58.70, 63.546, 65.38])

# Calculate the scatter correction for each Z-value
scatter_corr = np.divide(atomic_mass, Z_values)
PE_corr = np.divide(atomic_mass, np.power(Z_values, 5))

for i in np.arange(len(scatter_matrix[0])):
    scatter_matrix[:, i] = np.multiply(scatter_corr, scatter_matrix[:, i])
    PE_matrix[:, i] = np.multiply(PE_corr, PE_matrix[:, i])

#%%

F = interpolate.interp2d(spectra_energies, Z_values, PE_matrix, kind='cubic')

G = interpolate.interp2d(spectra_energies, Z_values, scatter_matrix, kind='cubic')

Fw = F(spectra_energies, Zw)
Gw= G(spectra_energies, Zw)

Fw = np.multiply(Fw, Zw4)

FGw= np.add(Fw, Gw)

muw_low = np.dot(wts_low, FGw)
muw_high = np.dot(wts_high, FGw)

muw_low = muw_low*NA*0.563
muw_high = muw_high*NA*0.563

#%%


def find_mu_rel(Z, rho, energy):
    """

    :param Z:
    :param rho:
    :param energy:
    :return:
    """
    if energy is 'low':
        weights = wts_low
        mu_water = muw_low
    else:
        weights = wts_high
        mu_water = muw_high

    Z4 = Z*Z*Z*Z
    F_Eji = F(spectra_energies, Z)
    G_Eji = G(spectra_energies, Z)

    F_Eji = np.multiply(F_Eji, Z4)
    FG = np.add(F_Eji, G_Eji)

    mu = np.dot(weights, FG)
    mu = mu * rho

    return mu#/mu_water

x = find_mu_rel(5.74, 0.194, 'high')
#x = x/np.dot(wts_low, FGw)
print(x)

#%%
import numpy as np
folder = 'D:/Research/Attenuation Data/SARRP Attenuation/'

air = np.load(folder + 'Air_attenuation.npy')
pa = 0.0013
G457 = np.load(folder + 'Gammex457_attenuation.npy')
p57 = 1.06
G450 = np.load(folder + 'Gammex450_SB3_attenuation.npy')
p50 = 1.92
eth = np.load(folder + 'C2H4_attenuation_polyethylene.npy')
peth = 0.95
sty = np.load(folder + 'C8H8_attenuation_polystyrene.npy')
psty = 0.24

mu1 = np.dot(wts_low[0:200], air)
mu1 = mu1*pa
mu2 = np.dot(wts_high[0:200], air)
mu2 = mu2*pa

mu3 = np.dot(wts_low[0:200], eth)
mu3 = mu3*peth
mu4 = np.dot(wts_high[0:200], eth)
mu4 = mu4*peth

mu5 = np.dot(wts_low[0:200], sty)
mu5 = mu5*psty
mu6 = np.dot(wts_high[0:200], sty)
mu6 = mu6*psty

mu7 = np.dot(wts_low[0:200], G457)
mu7 = mu7*p57
mu8 = np.dot(wts_high[0:200], G457)
mu8 = mu8*p57

mu9 = np.dot(wts_low[0:200], G450)
mu9 = mu9*p50
mu10 = np.dot(wts_high[0:200], G450)
mu10 = mu10*p50

print(mu1/mu7, mu2/mu8)
print(mu3/mu7, mu4/mu8)
print(mu5/mu7, mu6/mu8)
print(mu7/mu7, mu8/mu8)
print(mu9/mu7, mu10/mu8)

print(mu1, mu2)
print(mu3, mu4)
print(mu5, mu6)
print(mu7, mu8)
print(mu9, mu10)

#%%
import numpy as np
import matplotlib.pyplot as plt
my_x = [0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 4]

my_40 = [-999, -699, -879, -989, -847, -853, -669, -743, -832, 0, 6395]
my_80 = [-999, -694, -851, -986, -805, -811, -644, -709, -790, 0, 5099]

nist_x = np.arange(5)
nist_40 = [-999, -496, -872, 0, 7248]
nist_80 = [-999, -383, -848, 0, 5619]

plt.scatter(my_x, my_80, color='blue', s=200)
plt.scatter(nist_x, nist_80, color='red', s=100)
plt.title('80 kVp Comparison', fontsize=40)
plt.legend(['Backwards data', 'NIST data'], fontsize=30)
plt.ylabel('HU', fontsize=35)
plt.xticks(np.arange(5), ('Air', 'Polystyrene', 'Polyethylene', 'Solid Water', 'SB3 Bone'), fontsize=30)
plt.yticks(fontsize=30)
plt.show()
