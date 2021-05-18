import os
import numpy as np

directory = r'D:\OneDrive - University of Victoria\Research\LDA Data'
folder = '21-05-12_CT_metal'

air_folder = 'airscan_65s'
dark_folder = 'darkscan_65s'

air = np.load(os.path.join(directory, folder, air_folder, 'Data', 'data.npy'))
dark = np.load(os.path.join(directory, folder, dark_folder, 'Data', 'data.npy'))

save_air = os.path.join(directory, folder, 'airscan_60s', 'Data')
save_dark = os.path.join(directory, folder, 'darkscan_60s', 'Data')

os.makedirs(save_dark, exist_ok=True)
os.makedirs(save_air, exist_ok=True)

air = np.sum(air[1:], axis=0)
dark = np.sum(dark[1:], axis=0)

np.save(os.path.join(save_air, 'data.npy'), air)
np.save(os.path.join(save_dark, 'data.npy'), dark)
