import numpy as np
import matplotlib.pyplot as plt


def phantom_ROIs(image, radius=6):
    """
    This function generates the number of circular ROIs corresponding to user input center points of each ROI
    It will output as many ROIs as coordinates clicked
    The radius is also set by the user and may need some fine tuning
    Each mask will have 1's inside the ROI and nan everywhere else
    :param image: The image as a numpy array
    :param radius: The desired ROI radius (all ROIs will have this radius)
    :return: the saved masks as a single numpy array (individual masks callable by masks[i]
    """

    # Open the image and click the 6 ROIs
    coords = click_image(image, message_num=3)

    # Array to hold the saved masks
    num_rows, num_cols = np.shape(image)
    num_of_ROIs = len(coords)
    masks = np.empty([num_of_ROIs, num_rows, num_cols])

    # Plot to verify the ROI's
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(image, cmap='gray')

    for idx, center in enumerate(coords):

        # Make the mask for those center coordinates
        masks[idx] = circular_mask(center, radius, (num_rows, num_cols))

        # Verify the ROI
        circ = plt.Circle(center, radius=radius, fill=False, edgecolor='red')
        ax.add_artist(circ)

    plt.show()
    plt.pause(2)
    plt.close()

    return masks


def background_ROI(image):
    """
    This function takes a single ROI representing the background of an image, you will click the center of the ROI and
    a point along its radius
    :param image: The image as a numpy array
    :return: mask of the ROI containing 1's inside the ROI and nan elsewhere
    """

    # Open the image and click the single background ROI
    coords = click_image(image, message_num=1)

    # Array to hold the saved mask
    num_rows, num_cols = np.shape(image)

    # Plot to verify the ROI's
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    ax.imshow(image, cmap='gray')

    # Get the center point and the point on the edge of the desired ROI
    center = coords[0]
    point = coords[1]
    x1 = center[0]
    y1 = center[1]
    x2 = point[0]
    y2 = point[1]
    radius = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    mask = circular_mask(center, radius, (num_rows, num_cols))

    # Verify the ROI
    circ = plt.Circle(center, radius=radius, fill=False, edgecolor='red')
    ax.add_artist(circ)

    plt.show()
    plt.pause(2)
    plt.close()

    return mask


def noise_ROIs(image, radius=4):
    """
    This function generates the number of circular ROIs corresponding to user input center points of each ROI
    It will output as many ROIs as coordinates clicked
    The radius is also set by the user and may need some fine tuning
    Each mask will have 1's inside the ROI and nan everywhere else
    :param image: The image as a numpy array
    :param radius: The desired ROI radius (all ROIs will have this radius)
    :return: the saved masks as a single numpy array (individual masks callable by masks[i]
    """

    # Open the image and click the ROIs within the desired vial
    coords = click_image(image, message_num=4)

    # Array to hold the saved masks
    num_rows, num_cols = np.shape(image)
    num_of_ROIs = len(coords)
    masks = np.empty([num_of_ROIs, num_rows, num_cols])

    # Plot to verify the ROI's
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(image, cmap='gray')

    for idx, center in enumerate(coords):

        # Make the mask for those center coordinates
        masks[idx] = circular_mask(center, radius, (num_rows, num_cols))

        # Verify the ROI
        circ = plt.Circle(center, radius=radius, fill=False, edgecolor='red')
        ax.add_artist(circ)

    plt.show()
    plt.pause(3)
    plt.close()

    return masks


def entire_phantom(image, radii=13):

    ## OUTER PHANTOM
    coords1 = click_image(image, message_num=5)

    # Array to hold the saved mask
    num_rows, num_cols = np.shape(image)

    # Plot to verify the ROI's
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.imshow(image, cmap='gray')

    # Get the center point and the point on the edge of the desired ROI
    center = coords1[0]
    point = coords1[1]
    x1 = center[0]
    y1 = center[1]
    x2 = point[0]
    y2 = point[1]
    radius = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    outer_mask = circular_mask(center, radius, (num_rows, num_cols))
    circ = plt.Circle(center, radius=radius, fill=False, edgecolor='red')
    ax.add_artist(circ)

    ## VIALS
    coords2 = click_image(image, message_num=2)
    num_of_ROIs = len(coords2)
    masks = np.empty([num_of_ROIs, num_rows, num_cols])

    for idx, center in enumerate(coords2):
        # Make the mask for those center coordinates
        masks[idx] = circular_mask(center, radii, (num_rows, num_cols))

        # Verify the ROI
        circ = plt.Circle(center, radius=radii, fill=False, edgecolor='red')
        ax.add_artist(circ)

    plt.show()
    plt.pause(3)
    plt.close()
    # Create the full mask
    for mask in masks:
        inner = np.zeros([num_cols, num_rows])
        inner[mask != 1.0] = 1
        inner[mask == 1.0] = np.nan
        outer_mask = np.multiply(outer_mask, inner)

    return outer_mask


def click_image(image, message_num=0):
    """

    :param image:
    :param message_num:
    :return:
    """
    # These are the possible instructions that will be set as the title of how to collect the desired points
    instructions = {0: 'Click the center of the phantom first, then a point giving the desired radius from the center'
                    '\n Left-click: add point, Right-click: remove point, Enter: stop collecting',
                    1: 'Click the center of the desired ROI, then the desired radius (relative to the center)'
                    '\n Left-click: add point, Right-click: remove point, Enter: stop collecting',
                    2: 'Click the center of each ROI in order from water to highest concentration.'
                    '\n Left-click: add point, Right-click: remove point, Enter: stop collecting',
                    3: 'Click the center of each ROI from water vial, then move counter-clockwise.'
                    '\n Left-click: add point, Right-click: remove point, Enter: stop collecting',
                    4: 'Click the centers of the desired ROIs'
                    '\n Left-click: add point, Right-click: remove point, Enter: stop collecting',
                    5: 'Click the center of the phantom and the edge of the phantom'}

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    ax.imshow(image)
    ax.set_title(instructions[message_num])

    # Array to hold the coordinates of the center of the ROI and its radius
    # Left-click to add point, right-click to remove point, press enter to stop collecting
    coords = plt.ginput(n=-1, timeout=-1, show_clicks=True)
    coords = np.array(coords)
    coords = np.round(coords, decimals=0)
    plt.close()

    return coords


def circular_mask(center, radius, img_dim):
    """
    Creates a mask matrix of a circle at the specified location and with the specified radius
    :param center:
    :param radius:
    :param img_dim:
    :return:
    """
    # Create meshgrid of values from 0 to img_dim in both dimension
    xx, yy, = np.mgrid[:img_dim[1], :img_dim[0]]

    # Define the equation of the circle that we would like to create
    circle = (xx - center[1])**2 + (yy - center[0])**2

    # Create the mask of the circle
    arr = np.ones(img_dim)
    mask = np.ma.masked_where(circle < radius**2, arr)
    mask = mask.mask

    arr = np.zeros([img_dim[0], img_dim[1]])
    arr[mask] = 1
    arr[arr == 0] = np.nan

    return arr
