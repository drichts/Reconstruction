import numpy as np
import pydicom as pyd
import os
from obsolete import general_OS_functions as gof
from scipy.ndimage import rotate as rotate
import matplotlib.pyplot as plt


def dcm_to_npy(folder, start=1, stop=280, mid_slice=190, dim=120, energies=['40kVp', '80kVp'],
               load_directory='D:/Research/CBCT Scans/', save_directory='D:/Research/Python Data/CBCT/'):
    """
    This function takes the dcm files from the SARRP and converts them to .npy matrices ( Also crops the images to the
    dim (dimension) desired
    :param folder: The folder with the specific scans in it
    :param start: The slice number to start with
    :param stop: The slice number to stop at
    :param mid_slice: the slice that shows the desired view of the phantom
    :param dim: The dim desired to crop to
    :param energies: The energies used in the scan (for dual energy)
    :param load_directory: The directory where all the raw SARRP scan folders live
    :param save_directory: The directory where all the converted raw SARRP scan folders live
    :return:
    """
    path = load_directory + folder + '/'
    save_path = save_directory + folder + '/'

    # Create the folder in the save_directory
    gof.create_folder(folder_name=folder, directory_path=save_directory)

    # Create the 'RawMatrices' folder
    gof.create_folder(folder_name='RawMatrices', directory_path=save_path)

    save_path = save_path + 'RawMatrices/'

    # Save each slice as .npy matrix
    for energy in energies:

        dirs3 = os.listdir(save_path)

        # Create the energy folder in the RawMatrices folder
        gof.create_folder(folder_name=energy, directory_path=save_path)

        save_path = save_path + energy + '/'

        # Sub file path
        subpath = energy + '/Mouse_Cropped.xst/'
        # Load the mid_slice view to find the edges of the phantom
        data = pyd.dcmread(path + subpath + 'volume0' + mid_slice + '.dcm')
        s6 = data.pixel_array

        # Center the image for cropping
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(s6)
        ax.set_title('Click the edges of the phantom in the order: top, bottom, left, right. '
                     '\n Left-click: add point, Right-click: remove point, Enter: stop collecting')

        # Array to hold the coordinates of the center of the ROI and its radius
        # Left-click to add point, right-click to remove point, press enter to stop collecting
        # First point
        coords = plt.ginput(n=-1, timeout=-1, show_clicks=True)
        coords = np.array(coords)
        coords = np.round(coords, decimals=0)
        top = coords[0]
        bottom = coords[1]
        left = coords[2]
        right = coords[3]
        # Round the center coordinates to index numbers
        x = int(round((right[0]+left[0])/2))
        y = int(round((bottom[1]+top[1])/2))

        for i in np.arange(start, stop):
            if i < 10:
                filename = 'volume000' + str(i) + '.dcm'
                savename = 'volume000' + str(i) + '.npy'
            elif i < 100 and i >= 10:
                filename = 'volume00' + str(i) + '.dcm'
                savename = 'volume00' + str(i) + '.npy'
            else:
                filename = 'volume0' + str(i) + '.dcm'
                savename = 'volume0' + str(i) + '.npy'

            # Crop image
            crop = dim/2
            data = pyd.dcmread(path+subpath+filename)
            matrix = data.pixel_array
            matrix = matrix[y-crop:y+crop, x-crop:x+crop]
            np.save(save_path+savename, matrix)


def shift_to_match(folder, x=0, y=0, z=0, angle=0, dim=120, energies=['40kVp', '80kVp'],
                   directory='D:/Research/Python Data/CBCT/'):
    """
    Updates the higher energy images so that they directly overlap with the lower energy images. Takes the images from
    the RawMatrices folder and saves them in the Shifted Matrices folder
    :param folder: the specific scan folder
    :param x: The desired pixel shift left or right (must be int)
    :param y: The desired pixel shift up or down (within the image plane) (must be int)
    :param z: The desired pixel shift between images (need to shift all slices up or down to match) (must be int)
    :param angle: The desired angle to roll to rotate the higher energy image (can be float or int)
    :param dim: The dimension of the images (square)
    :param energies: the two energies the object was scanned at
    :param directory: the directory in which all the scan folders live
    :return:
    """
    path = directory + folder + '/'

    for energy in energies:

        load_path = path + energy

        gof.create_folder(folder_name='Shifted Matrices', directory_path=load_path)

        load_path = load_path + '/RawMatrices/'
        save_path = path + energy + '/Shifted Matrices/'

        # Get all the slices to shift
        files = os.listdir(load_path)

        for file in files:
            temp = np.load(load_path + file)

            if energy is '40kVp':
                # Don't need to do anything for 40 kVp images
                np.save(save_path + file, temp)
            else:
                savefile = file
                # Shift within XY plane (the slice plane)
                if y is not 0:
                    temp = np.roll(temp, y, axis=0)  # Y shift
                if x is not 0:
                    temp = np.roll(temp, x, axis=1)  # X shift

                # Rotation
                if angle is not 0:
                    index = np.round(np.abs(angle), decimals=0)
                    index = int(index)
                    temp = rotate(temp, angle)
                    temp = temp[index:index + dim, index:index + dim]

                # Shift slices in the z (rename files)
                if z is not 0:
                    file = file.replace('.npy', '')
                    file = file.replace('volume0', '')
                    file = int(file) + z
                    if file < 10:
                        savefile = 'volume000' + str(file) + '.npy'
                    elif file < 100 and file >= 10:
                        savefile = 'volume00' + str(file) + '.npy'
                    else:
                        savefile = 'volume0' + str(file) + '.npy'

                np.save(save_path + savefile, temp)


def normalize(folder, air_vial, subfolder, energies=['40kVp', '80kVp'], one_norm=False, mid_slice=180,
              directory='D:/Research/Python Data/CBCT/'):
    """
    This function takes the slice files that have been shifted (moved slightly so structures match in each image from
    different energies) and normalizes them to the water vial, it can handle normalizing each slice separately, or from
    one normalization values, saves in the Slices folder
    :param folder: the specific scan folder
    :param air_vial:
    :param subfolder:
    :param energies: the two energies the object was scanned at
    :param one_norm: True if one-normalization value is needed
    :param mid_slice: The slice to take the normalization value from for one_norm
    :param directory: the directory in which all the scan folders live
    :return:
    """
    for energy in energies:
        path = directory + folder + '/'
        load_path = path + energy + subfolder

        # Create the Slices subfolder within
        save_path = path + energy + '/Slices/'



        # Load the air and water ROI
        water_ROI = np.load(path + 'Vial0_MaskMatrix.npy')
        air_ROI = np.load(path + 'Vial' + str(air_vial) + '_MaskMatrix.npy')

        files = os.listdir(load_path)

        if one_norm:
            # Find the value for water and air
            mid_slice = str(mid_slice)
            norm_slice = np.load(load_path + 'volume0' + mid_slice + '.npy')
            water = np.nanmean(norm_slice * water_ROI)
            air = np.nanmean(norm_slice * air_ROI)
            water_air = water - air

            # Normalize each file
            for file in files:
                if 'volume' not in file:
                    continue

                temp = np.load(load_path + file)
                temp = np.subtract(temp, water)
                temp = np.divide(temp, water_air)
                temp = np.multiply(temp, 1000)
                np.save(save_path + file, temp)

        else:
            # Normalize each file
            for file in files:
                if 'volume' not in file:
                    continue

                temp = np.load(load_path + file)
                water = np.nanmean(temp * water_ROI)
                air = np.nanmean(temp * air_ROI)
                water_air = water - air
                temp = np.subtract(temp, water)
                temp = np.divide(temp, water_air)
                temp = np.multiply(temp, 1000)
                np.save(save_path + file, temp)


def find_HU(folder, subfolder, slice, show_image=False, energies=['40kVp', '80kVp'],
            directory='D:/Research/Python Data/CBCT/'):
    """
    :param folder:
    :param subfolder:
    :param slice:
    :param show_image:
    :param energies:
    :param directory:
    :return:
    """

    if slice < 10:
        file = 'volume000' + str(slice) + '.npy'
    elif slice < 100 and slice >= 10:
        file = 'volume00' + str(slice) + '.npy'
    else:
        file = 'volume0' + str(slice) + '.npy'

    print(energies[0])
    path = directory + folder + '/'
    subpath = path + energies[0] + '/' + subfolder + '/'

    for i in np.arange(5):
        vial = np.load(path + 'Vial' + str(i) + '_MaskMatrix.npy')
        image = np.load(subpath + file)
        print('Vial', i, ', HU: ', np.nanmean(vial * image))

    if show_image:
        plt.imshow(image, cmap='grey')
        plt.axis('off')
        plt.show()
        plt.pause(2)
        plt.close()

    subpath = path + energies[1] + '/' + subfolder + '/'
    print(energies[1])
    for i in np.arange(5):
        vial = np.load(path + 'Vial' + str(i) + '_MaskMatrix.npy')
        image = np.load(subpath + file)
        print('Vial', i, ', HU: ', np.nanmean(vial * image))

    if show_image:
        plt.imshow(image, cmap='grey')
        plt.axis('off')
        plt.show()
        plt.pause(2)
        plt.close()

