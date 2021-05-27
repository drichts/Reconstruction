import os
import numpy as np

DIRECTORY = r'D:\OneDrive - University of Victoria\Research\LDA Data'

# Dead pixel mask <row, column>
# Set: dead_pixel_mask[r, c] = np.nan
DEAD_PIXEL_MASK = np.load(os.path.join(DIRECTORY, 'dead_pixel_mask_width10.npy'))

"""
CT Reconstruction Parameters
"""

NUM_ASICS = 16  # Number of ASICs (2 per module)

PS = 0.330  # Pixel size of the physical detector

# Number of voxels in the reconstructed image in the specified direction
NX = 36*NUM_ASICS  # x-direction
NY = 36*NUM_ASICS  # y-direction
NZ = 24  # z-direction (axial direction)

# Physical size of the entire reconstructed image in the specified direction
SX = NX*PS  # x-dir (mm)
SY = NY*PS  # y-dir (mm)
SZ = 8  # Axial direction (z) (mm)

# Number of pixels of the physical detector along each direction
NU = 36*NUM_ASICS  # Longer (horizontal) direction
NV = 24  # Shorter (vertical) direction

# Physical detector size
SU = NU*PS  # mm (horizontal dir)
SV = NV*PS  # mm (vertical dir)

DSD = 578  # Distance from the x-ray source to detector (mm)  (At horiz. x-ray = 0, DSD = 578; x-ray = 300, DSD = 876)
DSO = 322  # Distance from x-ray source to the axis of rotation (isocenter) (mm)
           # (At horiz. x-ray = 0, DSO = 322; x-ray = 300, DSO = 625

# Angle settings
DIRECTION = -1  # Rotation direction (gantry rotation direction) (1 or -1)
DANG = 2  # Angle between captures (degrees)
DEG = np.arange(0, 359, DANG)  # List of all capture angles (0-359 by steps of dang)
DEG = DEG * DIRECTION  # Move along angles in the correct rotation direction
NUM_PROJ = len(DEG)

# filter options: 'ram-lak', 'cosine', 'hamming', 'hann'
FILTER = 'hamming'  # High-pass filter

# Single voxel size (in mm)
DX = SX/NX  # Reconstructed image x-dir voxel
DY = SY/NY  # Reconstructed image y-dir voxel
DZ = SZ/NZ  # Reconstructed image z-dir (axial) voxel

# This is correction for the detector rotation shift (real size, i.e. mm)
OFF_U, OFF_V = 0, 0  # Horizontal, vertical

# Spline interpolation order for remapping in backprojection (function numpy.ndimage.map_coordinates)
SPLINE_ORDER = 1  # Options: 0-5  CAUTION: reconstruction time increases significantly with order > 3

# Geometry calculations for center of each voxel in each direction (in mm measured from the center of the reconstructed
# image or the physical detector)
XS = np.arange(-(NX-1)/2, NX/2, 1) * DX  # Center of all voxels in x-dir
YS = np.arange(-(NY-1)/2, NY/2, 1) * DY  # Center of all voxels in y-dir
ZS = np.arange(-(NZ-1)/2, NZ/2, 1) * DZ  # Center of all voxels in the axial direction

US = np.arange(-(NU-1)/2, NU/2, 1) * PS + OFF_U  # Center of pixels in the physical detector (horizontally)
VS = np.arange(-(NV-1)/2, NV/2, 1) * PS + OFF_V  # Center of pixels in the physical detector (vertically)

# If running the MATLAB CTbackprojection
MAT = False  # True if running the Matlab CTbackprojection
INTERPTYPE = 'linear'  # 'linear' or 'nearest'