import numpy as np
import matplotlib.pyplot as plt

# Things to enter
save_nm = 'I_Au_5-17'  # Save name to append
extra_append = '_Gold'  # Can be left blank
folder = 'I_Au_5_17_19'  # Folder name
lower_bin, upper_bin = 0, 1  # Upper and lower bins (upper-lower)
vials = 6  # Number of ROIs
x = [0, 1, 1, 5]  # Concentrations

x = np.array(['a', 'b', 'c'])
print(np.core.defchararray.add('hello_', x))

#%%
def update_angle(direct, angle, eta):
    '''
    Define the function to update the direction of the particle
    :param direct: direction vector of the particle coming into the interaction
    :param angle: angle calculated from Compton, Rayleigh, or Pair Production
    :param eta: a random angle sampled within [0, 2pi]
    :return:
    '''
    # Get each component of the direction vector
    x = direct[0]
    y = direct[1]
    costheta = direct[2]
    chi = angle  # Chi = calculated angle from interaction

    # Calculate sin and cos for chi and eta
    sinchi = np.sin(chi)
    coschi = np.cos(chi)
    sineta = np.sin(eta)
    coseta = np.cos(eta)

    theta = np.arccos(costheta)  # Calculate the initial azimuthal angle

    # If theta = 0, we will get a divide by zero error
    if theta == 0:
        theta = 0.00000000001

    sintheta = np.sin(theta)

    # Get sin(phi) and cos(phi)
    cosphi = x / sintheta
    sinphi = y / sintheta

    # Create the matrix to update the direction
    matrix = np.array([[costheta * cosphi, -sinphi, sintheta * cosphi],
                       [costheta * sinphi, costheta, sintheta * sinphi],
                       [-sintheta, 0, costheta]])

    new_vector = np.array([sinchi * coseta, sinchi * sineta, coschi])

    # Calculate the updated direction
    new_direction = np.dot(matrix, new_vector)

    return new_direction