import os
import numpy as np


def cart_to_polar(data):

    center = (287, 287)
    r_max = 287

    c_shape = np.shape(data)  # Shape of the regular data

    # Shape of the polar data
    p_shape = c_shape
    p_shape[-3] = c_
