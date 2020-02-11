import numpy as np
import matplotlib.pyplot as plt
from generateROImask import generateROImask


# Take an image and compute the statistics of the number of ROIs given
def image_statistics(num_of_ROIs, radius, folder='', filename=''):

    #folder = 'D:/Research/Python Data/CBCT/Mouse_6-4-19/40kVp/'
    #filename = 'volume0160.npy'

    image = np.load(folder+filename)

    # Create arrays to hold the desired statistics
    Means = np.empty(num_of_ROIs)
    Medians = np.empty(num_of_ROIs)
    Std_Dev = np.empty(num_of_ROIs)

    # Create each of the masks (remember the order you select the ROIs)
    masks, coords = generateROImask(folder=folder, filename=filename, radius=radius, save=True)

    for i in np.arange(num_of_ROIs):

        # Get the desired data from the image
        data = np.multiply(image, masks[i])

        mean = np.nanmean(data)
        median = np.nanmedian(data)
        std = np.nanstd(data)

        Means[i] = mean
        Medians[i] = median
        Std_Dev[i] = std

    return Means, Medians, Std_Dev


# Function to go over multiple slices of the same images
def multiple_slice_stats(num_of_ROIs, radius, start_num, stop_num, folder='', energies=[]):

    length = stop_num - start_num + 1
    #folder = 'D:/Research/Python Data/CBCT/Au_6-5-19/'
    energies = ['40kVp/', '80kVp/']

    for energy in energies:

        path = folder + energy

        Mean = np.empty([length, num_of_ROIs])
        Median = np.empty([length, num_of_ROIs])
        Std = np.empty([length, num_of_ROIs])

        # Get the initial mean, median, and mode
        file = 'volume0'+str(stop_num)+'.npy'
        mean, median, std = image_statistics(num_of_ROIs, radius, folder=path, filename=file)
        Mean[length-1] = mean
        Median[length-1] = median
        Std[length-1] = std

        # Load the masks and center coordinates
        image = np.load(path+file)
        num_rows, num_cols = np.shape(image)
        masks = np.empty([num_of_ROIs, num_rows, num_cols])
        coords = np.load(path+'center-coords.npy')

        for i in np.arange(num_of_ROIs):
            masks[i] = np.load(path + 'Vial' + str(i) + '_MaskMatrix.npy')

        for j in np.arange(start_num, stop_num):

            file = 'volume0' + str(j) + '.npy'

            image = np.load(path+file)

            # Check if the ROIs match on the image
            # Plot to verify the ROI's
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.imshow(image, cmap='gray')

            for k in np.arange(num_of_ROIs):
                center = coords[k]
                circ = plt.Circle(center, radius=radius, fill=False, edgecolor='red')
                ax.add_artist(circ)

                temp = np.multiply(image, masks[k])
                mean = np.nanmean(temp)
                median = np.nanmedian(temp)
                std = np.nanstd(temp)

                Mean[j - start_num, k] = mean
                Median[j - start_num, k] = median
                Std[j - start_num, k] = std

            plt.show()
            plt.pause(0.5)
            plt.close()

        np.save(path+'Means.npy', Mean)
        np.save(path+'Medians.npy', Median)
        np.save(path+'Std_Devs.npy', Std)

    np.save(path+'ROI-Names.npy', np.array(['Water', '0.5%', '1%', '2%', '3%', '4%']))
    #np.save(path+'ROI-Names.npy', np.array(['Water', 'Au', 'Gd', 'Lu']))

folder = 'D:/Research/Python Data/CBCT/Au_6-5-19/'
multiple_slice_stats(6, 6, 155, 180, folder=folder)




