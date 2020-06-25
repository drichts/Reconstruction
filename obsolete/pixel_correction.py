'''
This code does an air and dark field correction to the CZT
it outputs a mask of all bad pixels for the CZT detector, and performs interpolation of neighbouring pixels
and subtracts off the (corrected/interpolated) dark field from the CZT projection data
'''
import numpy as np
import matplotlib.pyplot as plt
from CZTmodule import clickAwayTheBadPixels
import os


def main():
    '''
    This is the main method to run for this dead pixel correction.
    Set where the directory of the air and dark field scans are
    Specify the names of the air scans in an array, as well as the dark field scan
    Then specify the directory of where the projection images are that you wish to do the correction for
    This function will correct for dead pixels and then subtract each projection data image by the dark field for
    the corresponding energy bin.
    To correct for dead pixels, plots will come up, and you single-left-click where it's bad. You may click the same
    spot multiple times and it won't break the code.
    After this, the corrected projection images will be saved as .npy files, readable in Python as NumPy arrays
    The side edges of the pixel images are quite underresponding; this is taken care of in the stitching step so no need
    to click every pixel along the side.
    :return:
    '''

    # Directory where the air scan files are located
    directory_air = "C:/Users/drich/PycharmProjects/Reconstruction/sCT Scan Data/Au_4_29_19/Air/"


    # The filenames of the air scan files (raw data, .csv format) in array
    translations = np.array(["10", "0.5", "-9", "-18.5", "0.5"])
    air_files = np.array(["air_" + translations[0] + ".csv", "air_" + translations[1] + ".csv",
                          "air_" + translations[2] + ".csv", "air_" + translations[3] + ".csv",
                          "air_" + translations[4] + ".csv"])


    # The filename of the dark field scan file (raw data, .csv format), in same directory as air scan files
    darkfield_file = "dark.csv"

    size = np.array([24, 36])  # The CZT outputs 24 pixels high and 36 pixels wide, don't change this

    # Directory where all your projection data files are located
    directory_dataproj = "C:/Users/drich/PycharmProjects/Reconstruction/sCT Scan Data/Au_4_29_19/"

    # The filenames of all projection data files (raw data, .csv format) in array
    # Depending on your structure, you may find it useful to code this in a loop in a separate function
    proj_files = getDataFiles(directory_dataproj, translations)

    # Specify the energy bin of interest. One of "EC", "SEC0" ... "SEC5"
    # This code can loop over multiple energy bins, but will reset the mask of bad pixels you clicked after each loop.
    energy_name = "EC"

    # want to play with image windowing? Adjust these parameters (will display log scale)
    window_min = 1e2
    window_max = 1e7

    # Here's where the magic happens. If you want saved images of air and dark field scans change save=True
    new_proj_images = clickAwayTheBadPixels(energy_name, directory_air, air_files, darkfield_file, directory_dataproj,
                                            proj_files, size, (window_min, window_max), save=False)

    # This will loop over all your projection images and save as a numpy array to be loaded later in the same directory
    for ind, pj in enumerate(new_proj_images):
        np.save(directory_dataproj+proj_files[ind][0:-4]+".npy", pj)


def getDataFiles(dir, trans):
    '''
    Function that returns an array of the filenames of your CZT projection data
    You may wish to edit this function according to your file structure
    :param dir: directory where files are located
    :return: files: list of filenames
    '''

    files1 = os.listdir(dir + "Rot_" + trans[0] + "/")
    files1 = np.core.defchararray.add("Rot_" + trans[0] + "/", files1)
    files2 = os.listdir(dir + "Rot_" + trans[1] + "/")
    files2 = np.core.defchararray.add("Rot_" + trans[1] + "/", files2)
    files3 = os.listdir(dir + "Rot_" + trans[2] + "/")
    files3 = np.core.defchararray.add("Rot_" + trans[2] + "/", files3)
    files4 = os.listdir(dir + "Rot_" + trans[3] + "/")
    files4 = np.core.defchararray.add("Rot_" + trans[3] + "/", files4)
    # files5 = os.listdir(dir_home + "Rot_" + translations[4] + "/")

    files = np.concatenate((files1, files2))
    files = np.concatenate((files, files3))
    files = np.concatenate((files, files4))
    return files

if __name__ == "__main__":
    main()