import numpy as np
import matplotlib.pyplot as plt
import os


class ContrastMaterial:

    def __init__(self, material, density, directory=r'C:\Users\10376\Documents\Attenuation'):
        self.dir = directory
        self.material = material
        self.file = material + '.txt'
        self.path = os.path.join(self.dir, self.file)

        self.density = density
        self.values = self.load_att_txt()

    def load_att_txt(self):
        """This function loads the energies and mass attenuation values for the material"""
        f = open(self.path, 'rt')

        values = []
        for line in f:
            col = line.split()
            col = np.array(col)
            values.append(col)

        values = np.array(values, dtype='float')
        values[:, 0] = values[:, 0]*1000  # Convert to keV

        return values

    def interp_val(self, energy):
        """This function will return the mass attenuation value at the energy(ies) desired"""
        return np.interp(energy, self.values[:, 0], self.values[:, 1])
