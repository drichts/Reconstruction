import numpy as np
import matplotlib.pyplot as plt

# Things to enter
save_nm = 'I_Au_5-17'  # Save name to append
folder = 'I_Au_5_17_19'  # Folder name
r = 7  # Enter the desired radius
vial = 0  # Number of vials (Set to zero if just getting the background)

# Load the EC bin matrix (Make sure to have the correct files in the current folder)
s6 = np.load(folder + '\Bin6_Matrix_' + save_nm + '.npy')  # bin 6 (summed bin)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(s6)
ax.set_title('Click the center of each ROI in order from highest to lowest, background last. '
             '\n Left-click: add point, Right-click: remove point, Enter: stop collecting')

# Array to hold the coordinates of the center of the ROI and its radius
# Left-click to add point, right-click to remove point, press enter to stop collecting
# First point
coords = plt.ginput(n=-1, timeout=-1, show_clicks=True)
coords = np.array(coords)
coords = coords.astype(int)

# Plot to verify the ROI's on
fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(s6, cmap='gray')

# Set up the mask matrix
arr = np.empty(np.shape(s6))
xpts = np.arange(0, np.size(s6, 0))
ypts = np.arange(0, np.size(s6, 1))

# Set up the masks and plot the ROIs to verify
for center in coords:

    arr.fill(0)

    # Plot
    circ = plt.Circle(center, radius=r, fill=False, edgecolor='red')
    ax.add_artist(circ)

    # Create mask matrix
    cx = center[0]
    cy = center[1]
    mask = (xpts[np.newaxis, :] - cx)**2 + (ypts[:, np.newaxis] - cy)**2 < r**2
    arr[mask] = 1
    arr[arr == 0] = np.nan

    # Save the mask matrix
    if vial > 0:
        np.save(folder + '\Vial' + str(vial) + '_MaskMatrix_' + save_nm + '.npy', arr)
    else:
        # Save the background matrix
        np.save(folder + '\BackgroundMaskMatrix_' + save_nm + '.npy', arr)

    vial = vial - 1

plt.show()
